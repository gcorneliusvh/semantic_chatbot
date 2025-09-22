import streamlit as st
import pandas as pd
import io
import os
from pydantic import BaseModel, Field
# --- FIX: Add List, Dict, Optional ---
from typing import Literal, List, Dict, Optional
import json 
# --- ADDED: We need StructuredTool ---
from langchain_core.tools import StructuredTool
# --- FIX: Import Streamlit components ---
from streamlit import components

# --- LLM Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Langchain Core Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.tools import render_text_description
from langchain_core.messages import HumanMessage, AIMessage

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
    
    # --- MODIFICATION: llm_pro for heavy tasks ---
    llm_pro = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True
    )
    
    # --- MODIFICATION: llm_flash for lighter, faster tasks ---
    llm_flash = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Using the exact model name
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True
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

# --- MODIFICATION: Use llm_flash for the router ---
router = router_prompt | llm_flash | parser

# ==============================================================================
# 4. Agent Definitions
# ==============================================================================

# --- 4a. Python Agent ---
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

python_agent_prompt = PromptTemplate(
    template=PYTHON_AGENT_PROMPT_TEMPLATE,
    input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
)

def create_python_agent_chain():
    """Creates the Python agent executor."""
    if not os.path.exists("data.csv"):
        return (lambda x: "There is no data cached to analyze. Please run a Looker query first.")

    tools = [PythonREPLTool(locals={"pd": pd})]
    
    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    
    prompt_with_vars = python_agent_prompt.partial(
        tools=tools_description, 
        tool_names=tool_names
    )
    
    # --- MODIFICATION: Use llm_pro for Python Agent ---
    agent = create_react_agent(llm_pro, tools, prompt_with_vars)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "PythonAgent"})
    
    return agent_executor


# --- 4b. Looker Agent ---
LOOKER_AGENT_PROMPT_TEMPLATE = """
You are an expert assistant for US Census data. You have two tools:
1.  `LookerDataQuery`: To get data and visualizations.
2.  `get_census_data_definition`: To define terms.

**STRATEGY FOR `LookerDataQuery`:**
-   You MUST provide a `vis_config_string` (a JSON string).
-   **CHART PREFERENCE:** Prefer `'{{\"type\": \"looker_column\"}}'` or `'{{\"type\": \"looker_bar\"}}'` for most queries that compare values (e.g., "population by state").
-   Use `'{{\"type\": \"table\"}}'` only if the user specifically asks for a table or if it's a long list of data.
-   Use `'{{\"type\": \"single_value\"}}'` for single-number answers (e.g., "total population of US").

**STRATEGY FOR FINAL ANSWER:**
-   Your tool will return a JSON object with a "summary" and a "viz_url".
-   Your "Final Answer" to the user should be a friendly, conversational confirmation.
-   Start with the "summary" text (e.g., "Successfully queried 52 rows...").
-   Then, add a one-sentence insight or prompt for the next step (e.g., "This data is now cached, so you can ask me to analyze it.")
-   **DO NOT** include the "viz_url" or any markdown links. The app will display the visualization automatically.

Here are the tools you must use:
{tools}

Here are the names of your tools: {tool_names}

Use the following format for your thoughts and actions.
(You MUST use this format. Do not just output the final answer.)

Thought:
The user is asking for...
I need to use the `LookerDataQuery` tool.
I will select the fields...
For the visualization, the user wants to compare values, so I will use a column chart: `'{{\"type\": \"looker_column\"}}'`.
(or)
For the visualization, the user wants a single number, so I will use: `'{{\"type\": \"single_value\"}}'`.

Action:
```json
{{
  "action": "LookerDataQuery",
  "action_input": {{
    "fields": ["field_name_1", "field_name_2"],
    "filters": {{"some_field": "some_value"}},
    "sorts": ["field_name_1"],
    "vis_config_string": "{{\"type\": \"looker_column\"}}"
  }}
}}
```
Observation:
(The tool's JSON output, e.g., {{"summary": "Successfully queried 52 rows.", "viz_url": "https://..."}})

Thought:
The tool ran successfully and provided a summary. My job is to take that summary and make it more conversational and insightful.
The user's original query was: {input}
The summary is: (e.g., "Successfully queried 52 rows.")

This query implies an interest in comparisons (e.g., male vs. female, or values by state).
I will confirm I've retrieved the data for their query and suggest a *specific* next-step analysis that is relevant to their question.

Action:
```json
{{
  "action": "Final Answer",
  "action_input": "I've successfully retrieved the data for '{input}' ({{"summary"}}). This data is now cached and the chart is displayed below. Since the data is ready, you can now ask me to perform analysis, like 'which state has the highest population' or 'what is the male-to-female ratio'!"
}}
```

CHAT HISTORY:
{chat_history}

USER INPUT: {input}

Begin!
Thought:
{agent_scratchpad}
"""

# --- NEW: Wrapper function to capture viz_url ---
# --- FIX: The signature MUST match the original tool's args_schema (LookerQueryInput) ---
def run_looker_tool_and_save_url(
    fields: List[str], 
    filters: Optional[Dict[str, str]] = None, 
    sorts: Optional[List[str]] = None, 
    limit: Optional[str] = "500",
    vis_config_string: str = '{"type": "table"}'
):
    """
    A wrapper that calls the looker_data_tool and saves 
    the viz_url to the session state.
    """
    # --- FIX: Handle default Nones for filters and sorts ---
    if filters is None:
        filters = {}
    if sorts is None:
        sorts = []

    # --- FIX: Call the original tool's function with the keyword arguments ---
    result_str = looker_data_tool.func(
        fields=fields,
        filters=filters,
        sorts=sorts,
        limit=limit,
        vis_config_string=vis_config_string
    )
    
    viz_url = None
    if isinstance(result_str, str):
        try:
            result_json = json.loads(result_str)
            viz_url = result_json.get("viz_url")
            
            # Save the URL to a temporary spot in session state
            if viz_url:
                st.session_state["temp_viz_url"] = viz_url
                
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Wrapper could not parse viz_url: {e}")
            
    # Return the original string result to the agent
    return result_str
# --- END FIX ---

def create_looker_agent_chain():
    
    # --- MODIFICATION: Wrap the tool ---
    tools = [
        StructuredTool.from_function(
            func=run_looker_tool_and_save_url,
            name=looker_data_tool.name,
            description=looker_data_tool.description,
            args_schema=looker_data_tool.args_schema
        ),
        get_census_data_definition
    ]
    # --- END MODIFICATION ---
    
    agent_prompt = PromptTemplate(
        template=LOOKER_AGENT_PROMPT_TEMPLATE,
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
    )
    
    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    
    agent_prompt = agent_prompt.partial(
        tools=tools_description,
        tool_names=tool_names
    )

    # --- MODIFICATION: Use llm_flash for Looker Agent ---
    agent = create_structured_chat_agent(llm_flash, tools, agent_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
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
    
    prompt_with_vars = social_agent_prompt.partial(
        tools=tools_description, 
        tool_names=tool_names
    )
    
    # --- MODIFICATION: Use llm_flash for Social Agent ---
    agent = create_react_agent(llm_flash, tools, prompt_with_vars)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "SocialAgent"})
    
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
    
    prompt_with_vars = general_agent_prompt.partial(
        tools=tools_description, 
        tool_names=tool_names
    )
    
    # --- MODIFICATION: Use llm_pro for General/Knowledge Agent ---
    agent = create_react_agent(llm_pro, tools, prompt_with_vars)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    ).with_config({"run_name": "GeneralAgent"})
    
    return agent_executor

# ==============================================================================
# 5. Main Graph / Chain
# ==============================================================================

branch = RunnableBranch(
    (lambda x: x['route'].destination == "looker", create_looker_agent_chain()),
    (lambda x: x['route'].destination == "python_agent", create_python_agent_chain()),
    (lambda x: x['route'].destination == "social", create_social_agent_chain()),
    (lambda x: x['route'].destination == "general", create_general_agent_chain()), 
    create_general_agent_chain() # Default fallback
)

# ==============================================================================
# 6. Streamlit UI
# ==============================================================================

def setup_sidebar():
    """Configures and displays the Streamlit sidebar with cached data."""
    st.sidebar.title("Cached Datasource")
    
    try:
        df = load_df_from_cache.func(file_path="data.csv") 
        
        if df is not None and not df.empty:
            st.sidebar.info(f"**data.csv** loaded ({len(df)} rows)")
            st.sidebar.dataframe(df.head())
            
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
setup_sidebar()

# --- FIX: Store messages as dictionaries to include viz_url ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# --- NEW: Clear temp viz_url on page load ---
if "temp_viz_url" not in st.session_state:
    st.session_state.temp_viz_url = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # --- NEW: If message has a viz_url, display it ---
        if message.get("viz_url"):
            # --- FIX: Use st.components.v1.iframe and remove invalid arg ---
            st.components.v1.iframe(message["viz_url"], height=600)
# --- END FIX ---

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # --- FIX: Store as dictionary ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    # --- NEW: Clear the temp viz_url before every new run ---
    st.session_state.temp_viz_url = None
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            final_answer = ""
            viz_url = None # Variable to hold the URL
            route = None   # Variable to hold the route
            
            try:
                # --- FIX: Reconstruct message objects for the agent ---
                chat_history_list = []
                for msg in st.session_state.messages[:-1]: # All but the new one
                    if msg["role"] == "user":
                        chat_history_list.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history_list.append(AIMessage(content=msg["content"]))
                # --- END FIX ---
                
                chain_input = {"input": prompt, "chat_history": chat_history_list}
                
                route = router.invoke(chain_input)
                chain_input["route"] = route
                
                response = branch.invoke(chain_input)
                
                if isinstance(response, str):
                    final_answer = response
                else:
                    final_answer = response.get("output", "I'm sorry, I encountered an error.")
                
                # --- NEW IFRAME LOGIC ---
                # Check if the wrapper function saved a URL
                if st.session_state.temp_viz_url:
                    viz_url = st.session_state.temp_viz_url
                    st.session_state.temp_viz_url = None # Clear it
                # --- END NEW IFRAME LOGIC ---
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                final_answer = "I'm sorry, I'm having trouble processing that request."

        # Render text and iframe
        st.markdown(final_answer)
        if viz_url:
            # --- FIX: Use st.components.v1.iframe and remove invalid arg ---
            st.components.v1.iframe(viz_url, height=600)
        
    
    # --- FIX: Append the new dictionary to session state ---
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_answer, 
        "viz_url": viz_url # This will be None if it's not a Looker query
    })
    # --- END FIX ---

    # Rerun if it was a successful Looker query to update sidebar
    if (route and route.destination == 'looker' and 
        not final_answer.startswith("I'm sorry") and 
        not final_answer.startswith("There is no data")):
        st.rerun()

