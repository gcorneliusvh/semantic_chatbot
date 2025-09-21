import streamlit as st
import pandas as pd
import io
import os
from pydantic import BaseModel, Field
from typing import Literal

# --- LLM Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Langchain Core Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.tools import render_text_description

# --- Tool Imports ---
from tools.looker_tool import (
    get_all_looks, 
    get_looker_query_payload, 
    run_looker_query,
    get_visualization_embed_url,
    create_and_run_inline_query  # <-- NEW TOOL IMPORTED
)
from tools.knowledge_tool import get_census_data_definition
from tools.social_tool import get_profile_data
from tools.cache_tool import load_df_from_cache, save_data_to_cache

# ==============================================================================
# 1. Initialize LLM
# ==============================================================================

# Get the API key from Streamlit secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing or empty in secrets.toml")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True # Gemini prefers this
    )
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()


# ==============================================================================
# 2. Pydantic Model for Routing
# ==============================================================================

class RouteQuery(BaseModel):
    """Classifies the user's query to the appropriate agent."""
    destination: Literal["looker", "python_agent", "social", "general"] = Field(
        ..., 
        description=(
            "The agent to route the query to. "
            "'looker' for data retrieval/visualization about US census data. "
            "'python_agent' for analysis, math, or manipulation on *already cached* data. "
            "'social' for questions about LinkedIn profiles. "
            "'general' for conversation, definitions, or anything else."
        )
    )

# ==============================================================================
# 3. Router Chain
# ==============================================================================

router_prompt_template = """
You are an expert dispatcher routing user queries to the correct agent.
Based on the user's query and chat history, you must classify it into one of the following destinations.
Your response must be a JSON object matching the 'RouteQuery' Pydantic schema.

<SCHEMA>
{schema}
</SCHEMA>

CHAT HISTORY:
{chat_history}

USER QUERY:
{input}

CLASSIFICATION:
"""

parser = PydanticOutputParser(pydantic_object=RouteQuery)

router_prompt = PromptTemplate(
    template=router_prompt_template,
    input_variables=["input", "chat_history"],
    partial_variables={"schema": parser.get_format_instructions()}
)

# The router chain itself
router = router_prompt | llm | parser

# ==============================================================================
# 4. Agent Definitions
# ==============================================================================

# --- 4a. Python Agent ---
PYTHON_AGENT_PROMPT_TEMPLATE = """
You are an expert Python data analyst. You have access to a Python REPL tool.
A pandas DataFrame named `df` has already been loaded into the REPL's memory.
DO NOT try to load any data or use any other variable names; use `df` directly.

The `df` DataFrame contains data from a recent Looker query.
Here is the schema of `df` (from df.info()):
{schema}

Your task is to write and execute Python code to answer the user's question.
- Only use the `df` dataframe.
- Perform any calculations or manipulations needed.
- You MUST end your final code block with a `print()` statement
  containing the result or answer.
- Do not install any packages. Pandas is already available.

Here are the available tools:
{tools}

Use the following format:
Thought: I need to write Python code to answer the user's question.
Action: The action to take, should be one of [{tool_names}]
Action Input: 
```python
# Your python code here
print(df.some_method())
```
Observation: The result of your code.
... (this thought/action/observation can repeat)
Thought: I now have the final answer.
Final Answer: The final answer to the user

USER INPUT: {input}
CHAT HISTORY: {chat_history}

Begin!
Thought:
{agent_scratchpad}
"""

python_agent_prompt = PromptTemplate(
    template=PYTHON_AGENT_PROMPT_TEMPLATE,
    input_variables=["input", "chat_history", "agent_scratchpad", "schema", "tools", "tool_names"]
)

def create_python_agent_chain():
    """Creates the Python agent executor with the data_cache injected."""
    data_cache = load_df_from_cache.func() 
    
    if data_cache is None or data_cache.empty:
        return (lambda x: "There is no data cached to analyze. Please run a Looker query first.")

    buffer = io.StringIO()
    data_cache.info(buf=buffer)
    schema = buffer.getvalue()

    tools = [PythonREPLTool(locals={"df": data_cache})]
    
    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])

    agent = create_react_agent(llm, tools, python_agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "PythonAgent"})
    
    return agent_executor.bind(schema=schema, tools=tools_description, tool_names=tool_names)


# --- 4b. Looker Agent ---
# --- COMPLETELY REWRITTEN PROMPT ---
LOOKER_AGENT_PROMPT_TEMPLATE = """
You are an expert in US Census data and a master of the Looker platform.
Your goal is to answer the user's question using the Looker tools.

**EXPLORE INFORMATION:**
All US Census data is in a single Looker Explore with the following details:
- Model: `us_census`
- View: `population_data`
- Key Fields: 
  - `population_data.count` (Measure)
  - `population_data.average_age` (Measure)
  - `gender.name` (Dimension)
  - `state.name` (Dimension)
  - `age.group` (Dimension)
  - `education.level` (Dimension)
  - `income.bracket` (Dimension)

**YOUR STRATEGY:**
You must follow this plan:
1.  **Check Looks:** First, call `get_all_looks()` to see if a pre-built report (Look) matches the user's request.
2.  **Run Look:** If you find a Look with a title that is a *perfect match* (e.g., user asks for 'gender breakdown' and a Look is titled 'Gender Demographics'), then use `get_looker_query_payload` and `run_looker_query` to run it.
3.  **Build Inline Query:** If NO Look title is a good match, DO NOT try to run a Look. Instead, use `create_and_run_inline_query` to build a new query from scratch. Use the 'Explore Information' above to get the correct model, view, and field names.
4.  **Visualize:** If the user asks to "see" a chart or visualization, use `get_visualization_embed_url` with the `look_id` *after* you have determined a good Look exists.

**AVAILABLE TOOLS:**
{tools}

**FORMAT:**
Use the following format:
Thought: I need to decide my strategy.
Action: The action to take, should be one of [{tool_names}]
Action Input: The input for the tool.
Observation: The result from the tool.
... (this thought/action/observation can repeat)
Thought: I now have the final answer.
Final Answer: The final answer to the user

USER INPUT: {input}
CHAT HISTORY: {chat_history}

Begin!
Thought:
{agent_scratchpad}
"""

looker_agent_prompt = PromptTemplate(
    template=LOOKER_AGENT_PROMPT_TEMPLATE,
    input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
)

def create_looker_agent_chain():
    tools = [
        get_all_looks,
        get_looker_query_payload,
        run_looker_query,
        get_visualization_embed_url,
        get_census_data_definition,
        create_and_run_inline_query  # <-- NEW TOOL ADDED
    ]
    
    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    
    agent = create_react_agent(llm, tools, looker_agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "LookerAgent"})
    
    return agent_executor.bind(tools=tools_description, tool_names=tool_names)


# --- 4c. Social Agent ---
SOCIAL_AGENT_PROMPT_TEMPLATE = """
You are a social media research assistant.
Your only task is to get data about LinkedIn profiles using the provided tool.
Do not make up information.

Here are the available tools:
{tools}

Use the following format:
Thought: I need to use the get_profile_data tool.
Action: The action to take, should be one of [{tool_names}]
Action Input: The LinkedIn profile URL from the user input.
Observation: The result from the tool.
Thought: I now have the final answer.
Final Answer: The profile data or an error message.

USER INPUT: {input}
CHAT HISTORY: {chat_history}

Begin!
Thought:
{agent_scratchpad}
"""

social_agent_prompt = PromptTemplate(
    template=SOCIAL_AGENT_PROMPT_TEMPLATE,
    input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
)

def create_social_agent_chain():
    tools = [get_profile_data]

    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    
    agent = create_react_agent(llm, tools, social_agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "SocialAgent"})
    
    return agent_executor.bind(tools=tools_description, tool_names=tool_names)


# --- 4d. General Agent (Fallback) ---
GENERAL_AGENT_PROMPT_TEMPLATE = """
You are a helpful assistant.
Your goal is to provide helpful, conversational answers to the user's question.
You can also provide definitions for US Census terms using your tools.

Here are the available tools:
{tools}

Use the following format:
Thought: Do I need to use a tool?
Action: The action to take, should be one of [{tool_names}] (or 'No' if no tool is needed).
Action Input: The input for the tool (e.g., 'poverty line')
Observation: The result from the tool.
Thought: I now have the final answer.
Final Answer: The final answer to the user

USER INPUT: {input}
CHAT HISTORY: {chat_history}

Begin!
Thought:
{agent_scratchpad}
"""

general_agent_prompt = PromptTemplate(
    template=GENERAL_AGENT_PROMPT_TEMPLATE,
    input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
)

def create_general_agent_chain():
    tools = [get_census_data_definition]
    
    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    
    agent = create_react_agent(llm, tools, general_agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "GeneralAgent"})
    
    return agent_executor.bind(tools=tools_description, tool_names=tool_names)

# ==============================================================================
# 5. Main Graph / Chain
# ==============================================================================

branch = RunnableBranch(
    (lambda x: x['route'].destination == "looker", create_looker_agent_chain()),
    (lambda x: x['route'].destination == "python_agent", create_python_agent_chain()),
    (lambda x: x['route'].destination == "social", create_social_agent_chain()),
    create_general_agent_chain() 
)

chain = (
    RunnablePassthrough.assign(route=router)
    | branch
)

# ==============================================================================
# 6. Streamlit UI
# ==============================================================================

def setup_sidebar():
    """Configures and displays the Streamlit sidebar with cached data."""
    st.sidebar.title("Cached Datasource")
    
    try:
        # Use the tool's function to load the data
        df = load_df_from_cache.func() 
        
        if df is not None and not df.empty:
            st.sidebar.info(f"**data.csv** loaded ({len(df)} rows)")
            st.sidebar.dataframe(df.head())
            
            # Convert DataFrame to CSV string for downloading
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            st.sidebar.download_button(
                label="📥 Download data.csv",
                data=csv_data,
                file_name="data.csv",
                mime="text/csv",
            )
        else:
            st.sidebar.info("No data cached. Run a Looker query in the chatbot to load data.")
    except FileNotFoundError:
        st.sidebar.info("No data.csv file found. Run a Looker query to create one.")
    except Exception as e:
        st.sidebar.error(f"Error loading cache: {e}")

# --- Main Page ---
st.set_page_config(page_title="Semantic Chatbot", layout="wide")
st.title("Semantic Chatbot 🤖")

# --- Setup Sidebar ---
# This will be displayed on the main "app.py" page
setup_sidebar()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chat_history_str = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]]
                )
                
                chain_input = {"input": prompt, "chat_history": chat_history_str}
                
                response = chain.invoke(chain_input)
                
                final_answer = response.get("output", "I'm sorry, I encountered an error.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                final_answer = "I'm sorry, I'm having trouble processing that request."

        st.markdown(final_answer)
    
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

