import streamlit as st
import pandas as pd
import io
import os
import ast  # <-- NEW: Added for parsing follow-up questions
from pydantic import BaseModel, Field
# --- FIX: Add List, Dict, Optional ---
from typing import Literal, List, Dict, Optional, Any, Union
import json 
# --- ADDED: We need StructuredTool ---
from langchain_core.tools import StructuredTool
# --- FIX: Import Streamlit components ---
from streamlit import components

# --- LLM Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI # <-- FIX: Corrected typo

# --- Langchain Core Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.tools import render_text_description
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# --- NEW: Import for Callback Handler ---
from langchain_core.callbacks.base import BaseCallbackHandler
# --- NEW: Import for Parsing Fallback ---
from langchain_core.exceptions import OutputParserException

# --- Tool Imports ---
from tools.looker_tool import looker_data_tool
from tools.knowledge_tool import get_census_data_definition
from tools.social_tool import get_profile_data
from tools.cache_tool import load_df_from_cache, save_data_to_cache

# ==============================================================================
# 0. Custom Callback Handler
# ==============================================================================

class StreamlitCallbackHandler(BaseCallbackHandler):
    """A callback handler that writes agent thoughts to a list."""
    
    def __init__(self):
        super().__init__()
        self.thoughts = []

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log the start of a chain."""
        # --- vvv FIX: Add check for serialized ---
        # Only log the start of the main agent chains
        if serialized and "Agent" in serialized.get('name', ''):
             self.thoughts.append(f"> Entering new {serialized.get('name', 'chain')}...")
        # --- ^^^ FIX ---

    # --- vvv MODIFIED: Fixed code block formatting vvv ---
    def on_agent_action(
        self, action: Any, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Log the agent's action."""
        # Add newlines and language hint for proper markdown rendering
        self.thoughts.append(f"Action: {action.tool}\nAction Input:\n```python\n{action.tool_input}\n```")
    # --- ^^^ MODIFIED ^^^ ---

    def on_tool_end(
        self, output: str, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Log the tool's output."""
        self.thoughts.append(f"Observation: {output}")
        
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log the LLM's thought process."""
        try:
            if response.generations:
                # This usually contains the 'Thought:'
                generation_text = response.generations[0][0].text
                if generation_text:
                    self.thoughts.append(generation_text)
        except Exception:
            pass # Ignore if we can't parse

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
# --- vvv MODIFIED: Reverted to pre-loading 'df' and updated prompt vvv ---
PYTHON_AGENT_PROMPT_TEMPLATE = """
You are an expert Python data analyst. You have access to a Python REPL tool.
Your primary task is to answer the user's question by analyzing a pandas DataFrame named `df`.

**CRITICAL CONTEXT:**
- A pandas DataFrame named `df` has **ALREADY BEEN LOADED** into memory for you.
- Your **ONLY** job is to write Python code that *uses* this existing `df` variable.
- **DO NOT** use `pd.read_csv("data.csv")`. It is unnecessary and will fail.
- The pandas library is available as `pd`.
- Your final answer should be conversational, explaining what you found.

Here are the tools you must use:
{tools}

Here are the names of your tools: {tool_names}

Use the following format for your thoughts and actions.
(You MUST use this format. Do not just output the final answer.)

Thought:
The user is asking for...
My first step is to inspect the `df` DataFrame to see the column names, as I must not assume them.
Action:
```json
{{
  "action": "Python_REPL",
  "action_input": "print(df.columns)"
}}
```
Observation:
(The tool's output, e.g., "Index(['state.state_name', ...])")

Thought:
Okay, I have the correct column names (e.g., 'state.state_name', 'blockgroup.bachelors_degree').
Now I will write the code to perform the calculation and print the result.
Action:
```json
{{
  "action": "Python_REPL",
  "action_input": "result = df['blockgroup.bachelors_degree'].mean()\nprint(result)"
}}
```
Observation:
(The tool's output, e.g., "12345.67")

Thought:
I have the final numerical result. Now I will provide a conversational answer.
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "I've finished the analysis. The average number of residents with a bachelor's degree is 12,345.67."
}}
```

CHAT HISTORY:
{chat_history}

USER INPUT: {input}

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

    # --- vvv MODIFIED: Pre-load df and add to locals vvv ---
    df = load_df_from_cache.func(file_path="data.csv")
    
    if df is None:
        return (lambda x: "Could not load data.csv. The file might be empty or corrupt. Please run a Looker query first.")
    
    # Provide the loaded df to the tool's local scope
    tools = [PythonREPLTool(locals={"pd": pd, "df": df})]
    # --- ^^^ MODIFIED ^^^ ---
    
    tools_description = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    
    prompt_with_vars = python_agent_prompt.partial(
        tools=tools_description, 
        tool_names=tool_names
    )
    
    # --- MODIFICATION: Use llm_pro and create_structured_chat_agent ---
    agent = create_structured_chat_agent(llm_pro, tools, prompt_with_vars)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=False, # <-- MODIFIED: Set to False
        return_intermediate_steps=True 
    ).with_config({"run_name": "PythonAgent"})
    
    return agent_executor
# --- ^^^ MODIFIED: Python agent logic and error handling ^^^ ---


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
-   Your tool will return a JSON object with:
    1.  "summary" (e.g., "Successfully queried 52 rows...")
    2.  "viz_url" (e.g., "https://...")
    3.  "data_preview" (a JSON string of the first 5 rows, e.g., "[{{\"state.name\": \"California\", ...}}]")
    4.  "data_stats" (a JSON string of `df.describe()`, e.g., "{{\"blockgroup.total_pop\": {{\"mean\": 600000, \"min\": 100, \"max\": 39000000, ...}} }}")
-   Your "Final Answer" to the user must be a 1-2 paragraph response, formatted in MARKDOWN with two headings: '### Summary' and '### Insights'.
-   Under '### Summary', describe the data that was retrieved, incorporating the "summary" text from the tool.
-   Under '### Insights', use the "data_stats" (for min/max/mean) and "data_preview" (for string examples) to provide a brief, *specific* analysis.
-   **DO NOT** include the "viz_url", "data_preview", or "data_stats" in your final answer. The app will display the visualization automatically.
-   **DO NOT** suggest follow-up questions. This will be handled by another part of the system.

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
(The tool's JSON output, e.g., {{"summary": "Successfully queried 52 rows.", "viz_url": "https://...", "data_preview": "[...]", "data_stats": "{{\"blockgroup.total_pop\": {{\"mean\": 600000, ...}} }}" }})

Thought:
The tool ran successfully. My job is to generate a detailed summary and insight.
The user's query was: '{input}'
The tool's summary is: 'Successfully queried 52 rows.'
The data preview is: '[{{\"state.name\": \"California\", ...}}]'
The data stats are: '{{\"blockgroup.total_pop\": {{\"mean\": 600000, \"min\": 100, \"max\": 39000000, ...}} }}'
I will use this data_stats and data_preview to generate a specific insight.

Action:
```json
{{
  "action": "Final Answer",
  "action_input": "### Summary\n\nI've successfully retrieved the data for '{input}' ({{"summary"}}). This data, which is now cached, provides a breakdown of [describe what the data is, e.g., population by state] across the United States. The chart is displayed below.\n\n### Insights\n\nBased on a statistical summary of the data, I can see [describe the specific insight from data_stats, e.g., 'the average population per state is 600,000, with a maximum of 39,000,000']. This highlights the wide variation in population density. The data is cached, so you can now ask analysis questions."
}}
```

CHAT HISTORY:
{chat_history}

USER INPUT: {input}

Begin!
Thought:
{agent_scratchpad}
"""

def run_looker_tool_and_save_url(
    fields: List[str], 
    filters: Optional[Dict[str, str]] = None, 
    sorts: Optional[List[str]] = None, 
    limit: Optional[str] = "500",
    vis_config_string: str = '{"type": "table"}'
):
    """
    A wrapper that calls the looker_data_tool and saves 
    the viz_url and data_summary to the session state.
    It also reads the newly cached data and passes a preview and stats back to the agent.
    """
    if filters is None:
        filters = {}
    if sorts is None:
        sorts = []

    # 1. Call the original tool. This saves "data.csv"
    result_str = looker_data_tool.func(
        fields=fields,
        filters=filters,
        sorts=sorts,
        limit=limit,
        vis_config_string=vis_config_string
    )
    
    viz_url = None
    data_summary = None
    result_json = {} # Start with an empty dict
    
    if isinstance(result_str, str):
        try:
            # 2. Parse the tool's output (summary and viz_url)
            result_json = json.loads(result_str)
            viz_url = result_json.get("viz_url")
            data_summary = result_json.get("summary")
            
            # 3. Save to session state for the UI
            if viz_url:
                st.session_state["temp_viz_url"] = viz_url
            if data_summary:
                st.session_state["temp_data_summary"] = data_summary
                
            # --- vvv NEW: Load cache and add data preview AND stats vvv ---
            # 4. Read the cache that the tool just saved
            df = load_df_from_cache.func(file_path="data.csv")
            if df is not None:
                # 5. Get a preview and add it to the JSON for the agent
                data_preview = df.head().to_json(orient='records')
                result_json["data_preview"] = data_preview
                
                # 6. Get stats and add them to the JSON for the agent
                data_stats = df.describe().to_json()
                result_json["data_stats"] = data_stats
                # --- Save stats for the followup generator ---
                st.session_state["temp_data_stats"] = data_stats
            # --- ^^^ END NEW ^^^ ---
                
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Wrapper could not parse viz_url/summary: {e}")
            result_json["error"] = str(e) # Add error to JSON
            
    # 6. Return the *enhanced* JSON (with preview and stats) back to the agent
    return json.dumps(result_json)

def create_looker_agent_chain():
    
    tools = [
        StructuredTool.from_function(
            func=run_looker_tool_and_save_url,
            name=looker_data_tool.name,
            description=looker_data_tool.description,
            args_schema=looker_data_tool.args_schema
        ),
        get_census_data_definition
    ]
    
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
        handle_parsing_errors=False, # <-- MODIFIED: Set to False
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
Action:
```json
{{
  "action": "get_profile_data",
  "action_input": {{"profile_url": "..."}}
}}
```
Observation: The result from the tool.
Thought: I now have the final answer.
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "The profile data or an error message."
}}
```

USER INPUT: {input}
CHAT HISTORY:
{chat_history}

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
    
    # --- MODIFICATION: Use llm_flash and create_structured_chat_agent ---
    agent = create_structured_chat_agent(llm_flash, tools, prompt_with_vars)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=False, # <-- MODIFIED: Set to False
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
Action:
```json
{{
  "action": "get_census_data_definition",
  "action_input": {{"term": "..."}}
}}
```
(or)
Thought: I don't need a tool.
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "The helpful answer to the user."
}}
```
Observation: The result from the tool.
Thought: I now have the final answer.
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "The final answer to the user"
}}
```

USER INPUT: {input}
CHAT HISTORY:
{chat_history}

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
    
    # --- MODIFICATION: Use llm_pro and create_structured_chat_agent ---
    agent = create_structured_chat_agent(llm_pro, tools, prompt_with_vars)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=False, # <-- MODIFIED: Set to False
    ).with_config({"run_name": "GeneralAgent"})
    
    return agent_executor

# --- vvv MODIFIED FUNCTION vvv ---
def get_followup_questions(user_query: str, chat_history: List, data_summary: str, data_stats: str) -> List[str]:
    """Generates 3 follow-up questions using llm_flash."""
    
    followup_prompt_template = """
    You are an AI assistant. Based on the user's last query, the chat history, and the data that was just retrieved, generate 3 relevant follow-up questions.
    
    - The questions should be insightful and encourage further exploration of the data.
    - The data has been cached, so questions can be about analysis (e.g., "what is the highest...", "compare X and Y...").
    - Use the DATA STATS to ask specific questions about min/max/mean values.
    - Return *only* a Python list of strings, e.g., ["question 1", "question 2", "question 3"].
    - Do not add any other text, titles, or markdown.
    
    CHAT HISTORY:
    {chat_history}
    
    USER'S LAST QUERY:
    {user_query}
    
    DATA SUMMARY:
    {data_summary}
    
    DATA STATS (df.describe()):
    {data_stats}
    
    FOLLOW-UP QUESTIONS (as a Python list of strings):
    """
    
    followup_prompt = PromptTemplate(
        template=followup_prompt_template,
        input_variables=["user_query", "chat_history", "data_summary", "data_stats"]
    )
    
    # Use the fast model for this
    chain = followup_prompt | llm_flash
    
    try:
        response = chain.invoke({
            "user_query": user_query,
            "chat_history": chat_history, # Pass the list of Message objects
            "data_summary": data_summary,
            "data_stats": data_stats
        })
        
        # Check if response has 'content' attribute (AIMessage) or is just a string
        content = response.content if hasattr(response, 'content') else response
        
        # Safely evaluate the string to a list
        questions = ast.literal_eval(content.strip())
        if isinstance(questions, list):
            return questions
        return []
    except Exception as e:
        print(f"Error generating follow-up questions: {e}\nResponse was: {content}")
        return []
# --- ^^^ MODIFIED FUNCTION ^^^ ---

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

# --- MODIFICATION: Store messages as dictionaries to include viz_url AND followup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# --- NEW: Clear temp session state vars on page load ---
if "temp_viz_url" not in st.session_state:
    st.session_state.temp_viz_url = None
if "temp_data_summary" not in st.session_state:
    st.session_state.temp_data_summary = None
if "temp_data_stats" not in st.session_state: # <-- NEW
    st.session_state.temp_data_stats = None
# --- NEW: Add state for clickable prompts ---
if "clicked_prompt" not in st.session_state:
    st.session_state.clicked_prompt = None

# --- vvv MODIFIED HISTORY RENDER vvv ---
# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages): # <-- Add enumerate
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # --- NEW: If message has agent_thoughts, display them ---
        if message.get("agent_thoughts"):
            with st.expander("See my thought process..."):
                st.code("\n".join(message["agent_thoughts"]), language="text")

        # If message has a viz_url, display it
        if message.get("viz_url"):
            st.components.v1.iframe(message["viz_url"], height=600)
            
        # --- NEW: If message has followup questions, display them as buttons ---
        if message.get("followup"):
            st.markdown("---")
            st.markdown("**Suggested follow-up questions:**")
            # Use columns for a cleaner layout
            if len(message.get("followup")) > 0:
                cols = st.columns(min(len(message.get("followup")), 3)) # Max 3 columns
                for j, q in enumerate(message.get("followup")):
                    with cols[j % 3]:
                        if st.button(q, key=f"followup_{i}_{j}"): # Unique key
                            st.session_state.clicked_prompt = q
                            st.rerun()
# --- ^^^ MODIFIED HISTORY RENDER ^^^ ---


# --- vvv MODIFIED: Handle prompt from chat_input OR button click vvv ---

# Always render the chat input box at the bottom
prompt_from_input = st.chat_input("What would you like to know?")

# Check if a button was clicked in the last run
prompt_from_button = st.session_state.clicked_prompt
st.session_state.clicked_prompt = None # Clear it immediately

# Decide which prompt to use
prompt = prompt_from_input or prompt_from_button

# --- ^^^ MODIFIED ^^^ ---


# React to user input
if prompt:
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # --- NEW: Clear *all* temp keys before every new run ---
    st.session_state.temp_viz_url = None
    st.session_state.temp_data_summary = None
    st.session_state.temp_data_stats = None # <-- NEW
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # --- vvv MODIFIED: Show routing and expandable thoughts vvv ---
        callback_handler = StreamlitCallbackHandler()
        config = {"callbacks": [callback_handler]}
        
        with st.spinner("Routing query..."):
            try:
                # Reconstruct message objects for the agent
                chat_history_list = []
                for msg in st.session_state.messages[:-1]: # All but the new one
                    if msg["role"] == "user":
                        chat_history_list.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history_list.append(AIMessage(content=msg["content"]))
                
                chain_input = {"input": prompt, "chat_history": chat_history_list}
                
                route = router.invoke(chain_input, config=config)
                chain_input["route"] = route
            
            except Exception as e:
                st.error(f"Error during routing: {e}")
                route = None # Set route to None to stop
        
        if route:
            st.success(f"Routing to: {route.destination}")
            
            with st.spinner(f"Running {route.destination}..."):
                
                final_answer = ""
                viz_url = None        # Variable to hold the URL
                data_summary = None   # Variable to hold the summary
                data_stats = None     # <-- NEW: Variable to hold the stats
                followup_questions = [] # Variable to hold questions
                
                # --- vvv MODIFIED: Add try/except for OutputParserException vvv ---
                try:
                    response = branch.invoke(chain_input, config=config)
                    
                    if isinstance(response, str):
                        final_answer = response
                    else:
                        # This handles output from all agents now
                        final_answer = response.get("output", "I'm sorry, I encountered an error.")
                
                except OutputParserException as e:
                    st.warning("The agent's response wasn't formatted perfectly, but I recovered this answer:")
                    # This is the fallback: use the raw LLM output
                    final_answer = str(e.llm_output) if hasattr(e, 'llm_output') else str(e)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    final_answer = "I'm sorry, I'm having trouble processing that request."
                # --- ^^^ MODIFIED ^^^ ---

                # --- vvv MODIFIED: Check for viz_url, summary, and stats vvv ---
                if st.session_state.temp_viz_url:
                    viz_url = st.session_state.temp_viz_url
                    st.session_state.temp_viz_url = None # Clear it
                    
                if st.session_state.temp_data_summary:
                    data_summary = st.session_state.temp_data_summary
                    st.session_state.temp_data_summary = None # Clear it
                
                if st.session_state.temp_data_stats: # <-- NEW
                    data_stats = st.session_state.temp_data_stats
                    st.session_state.temp_data_stats = None # Clear it
                # --- ^^^ MODIFIED ^^^ ---

                # --- vvv MODIFIED: Generate follow-up questions with stats vvv ---
                if (route and route.destination == 'looker' and 
                    data_summary and data_stats and # <-- Check for stats
                    not final_answer.startswith("I'm sorry")):
                    
                    with st.spinner("Generating follow-up questions..."):
                        followup_questions = get_followup_questions(
                            user_query=prompt,
                            chat_history=chat_history_list,
                            data_summary=data_summary,
                            data_stats=data_stats # <-- Pass stats
                        )
                # --- ^^^ MODIFIED ^N^ ---
                    
            # --- vvv MODIFIED: Render text, then thoughts, then iframe, then follow-ups vvv ---
            
            # 1. Render the main answer
            st.markdown(final_answer)
            
            # 2. Render the agent's thoughts
            with st.expander("See my thought process..."):
                st.code("\n".join(callback_handler.thoughts), language="text")
            
            # 3. Render the visualization
            if viz_url:
                st.components.v1.iframe(viz_url, height=600)
            
            # 4. Render the follow-up questions
            if followup_questions:
                st.markdown("---") # Add a separator
                st.markdown("**Suggested follow-up questions:**")
                # Use columns for a cleaner layout
                if len(followup_questions) > 0:
                    cols = st.columns(min(len(followup_questions), 3)) # Max 3 columns
                    for i, q in enumerate(followup_questions):
                        with cols[i % 3]:
                            if st.button(q, key=f"followup_new_{i}"): # Use a unique key
                                st.session_state.clicked_prompt = q
                                # This will cause the rerun
            # --- ^^^ MODIFIED ^^^ ---
        
            # --- vvv MODIFIED: Append new dictionary with followups AND thoughts vvv ---
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer, 
                "viz_url": viz_url,
                "followup": followup_questions,
                "agent_thoughts": callback_handler.thoughts # <-- NEW
            })
            # --- ^^^ MODIFIED ^^^ ---

            # Rerun if it was a successful Looker query to update sidebar
            if (route and route.destination == 'looker' and 
                not final_answer.startswith("I'm sorry") and 
                not final_answer.startswith("There is no data")):
                st.rerun()

