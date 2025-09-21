import streamlit as st
import pandas as pd
import io
import os
from pydantic import BaseModel, Field
from typing import Literal
import json # <-- ADDED IMPORT

# --- LLM Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI # <-- FIX: Corrected typo

# --- Langchain Core Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.output_parsers.pydantic import PydanticOutputParser
# --- FIX 1: ADDED create_structured_chat_agent and hub ---
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain import hub
# --- END FIX ---
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.tools import render_text_description
# --- FIX: Import Message types ---
from langchain_core.messages import HumanMessage, AIMessage
# --- END FIX ---

# --- Tool Imports ---
from tools.looker_tool import looker_data_tool
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
# --- FIX: Updated prompt to instruct agent to load data ---
PYTHON_AGENT_PROMPT_TEMPLATE = """
You are an expert Python data analyst. You have access to a Python REPL tool.
Your first step is to load the cached data. The data is stored in a file named "data.csv".
You MUST load this file into a pandas DataFrame named `df`.
The pandas library is available to you as `pd`.

Example loading code:
```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
```

After loading the data, your task is to write and execute Python code to answer the user's question using the `df` dataframe.
- You MUST end your final code block with a `print()` statement
  containing the result or answer.
- Do not install any packages.

Here are the available tools:
{tools}

Use the following format:
Thought: I need to load the data, then write Python code to answer the user's question.
Action: The action to take, should be one of [{tool_names}]
Action Input: 
```python
# Your python code here
# (e.g., df = pd.read_csv("data.csv"))
# (e.g., print(df.some_method()))
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

# --- FIX: Removed 'schema' from input_variables ---
python_agent_prompt = PromptTemplate(
    template=PYTHON_AGENT_PROMPT_TEMPLATE,
    input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
)

def create_python_agent_chain():
    """Creates the Python agent executor."""
    # --- FIX: Check for file existence, but don't load it ---
    if not os.path.exists("data.csv"):
        return (lambda x: "There is no data cached to analyze. Please run a Looker query first.")
    # --- END FIX ---

    # --- FIX: Inject 'pd' instead of 'df' ---
    tools = [PythonREPLTool(locals={"pd": pd})]
    # --- END FIX ---
    
    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    
    # --- FIX: Partial the prompt to inject variables (no schema) ---
    prompt_with_vars = python_agent_prompt.partial(
        tools=tools_description, 
        tool_names=tool_names
    )
    # --- END FIX ---
    
    agent = create_react_agent(llm, tools, prompt_with_vars) # Use the partial'd prompt
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "PythonAgent"})
    
    return agent_executor


# --- 4b. Looker Agent ---
def create_looker_agent_chain():
    tools = [
        looker_data_tool,
        get_census_data_definition
    ]
    
    agent_prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True # This is very important
    ).with_config({"run_name": "LookerAgent"})
    
    return agent_executor


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
    
    # --- FIX: Partial the prompt to inject variables ---
    prompt_with_vars = social_agent_prompt.partial(
        tools=tools_description, 
        tool_names=tool_names
    )
    # --- END FIX ---
    
    agent = create_react_agent(llm, tools, prompt_with_vars) # Use the partial'd prompt
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "SocialAgent"})
    
    # --- FIX: No need to bind variables here ---
    return agent_executor


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
    
    # --- FIX: Partial the prompt to inject variables ---
    prompt_with_vars = general_agent_prompt.partial(
        tools=tools_description, 
        tool_names=tool_names
    )
    # --- END FIX ---
    
    agent = create_react_agent(llm, tools, prompt_with_vars) # Use the partial'd prompt
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "GeneralAgent"})
    
    # --- FIX: No need to bind variables here ---
    return agent_executor

# ==============================================================================
# 5. Main Graph / Chain
# ==============================================================================

branch = RunnableBranch(
    (lambda x: x['route'].destination == "looker", create_looker_agent_chain()),
    (lambda x: x['route'].destination == "python_agent", create_python_agent_chain()),
    (lambda x: x['route'].destination == "social", create_social_agent_chain()),
    create_general_agent_chain() 
)

# --- MODIFIED: The router is now run *before* the branch in the UI section ---
# chain = (
#     RunnablePassthrough.assign(route=router)
#     | branch
# )

# ==============================================================================
# 6. Streamlit UI
# ==============================================================================

def setup_sidebar():
    """Configures and displays the Streamlit sidebar with cached data."""
    st.sidebar.title("Cached Datasource")
    
    try:
        # --- FIX: Explicitly load 'data.csv' ---
        df = load_df_from_cache.func(file_path="data.csv") 
        
        if df is not None and not df.empty:
            # --- FIX: Correct label ---
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
        # --- FIX: Correct error message ---
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
    # --- FIX: Read from .content attribute ---
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)
        # --- MODIFICATION: Check if viz_url is in history and display ---
        # This part is complex, let's skip for now to avoid saving complex objects.
        # The user will just see the text on rerun, which is fine.

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # --- FIX: Append HumanMessage object ---
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            final_answer = ""
            viz_url = None # Variable to hold the URL
            
            try:
                # --- FIX: Pass the list of messages directly ---
                # Get all messages *except* the new one
                chat_history_list = st.session_state.messages[:-1]
                
                chain_input = {"input": prompt, "chat_history": chat_history_list}
                
                # --- MODIFICATION: Run router first to check destination ---
                route = router.invoke(chain_input)
                chain_input["route"] = route
                
                response = branch.invoke(chain_input)
                # --- END MODIFICATION ---
                
                # --- FIX: Handle string response from python_agent guard clause ---
                if isinstance(response, str):
                    final_answer = response
                else:
                    final_answer = response.get("output", "I'm sorry, I encountered an error.")
                # --- END FIX ---
                
                # --- NEW IFRAME LOGIC ---
                if (route and route.destination == 'looker' and
                    response.get("intermediate_steps") and
                    len(response["intermediate_steps"]) > 0):
                    
                    # Get the last tool call's observation
                    last_step = response["intermediate_steps"][-1]
                    observation = last_step[1] # This is the JSON string from looker_tool
                    
                    if isinstance(observation, str):
                        try:
                            obs_json = json.loads(observation)
                            viz_url = obs_json.get("viz_url")
                        except (json.JSONDecodeError, AttributeError) as e:
                            # This is a soft fail, so we just print to console
                            print(f"Could not parse viz_url from observation: {e}")
                # --- END NEW LOGIC ---
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                final_answer = "I'm sorry, I'm having trouble processing that request."

        # --- MODIFIED: Render text and iframe separately ---
        st.markdown(final_answer)
        
        if viz_url:
            st.iframe(viz_url, height=500, scrolling=True)
        # --- END MODIFICATION ---
    
    # --- FIX: Append AIMessage object (only text) ---
    st.session_state.messages.append(AIMessage(content=final_answer))

