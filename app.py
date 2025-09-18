# In app.py
import streamlit as st
import streamlit.components.v1 as components
import json 
import pandas as pd
from io import StringIO

# --- (All imports from previous step remain the same) ---
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_core.runnables import RunnableBranch, RunnableLambda
from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool
from pydantic import BaseModel, Field
from typing import Literal
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from tools.looker_tool import looker_data_tool
from tools.social_tool import social_tool
from tools.knowledge_tool import general_knowledge_tool
from tools.cache_tool import save_to_cache 
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from typing import Any, Dict, List

# --- CALLBACK HANDLER CLASS (Unchanged) ---
class StreamlitCallback(BaseCallbackHandler):
    # (This class is unchanged)
    def __init__(self, status_container, log_container):
        self.status = status_container
        self.log = log_container
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.status.update(label="Agent is thinking...")
        self.log.write(f"🤔 **Thinking...** ({serialized.get('name', 'LLM')})")
        self.log.code(prompts[0], language="text")
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        self.status.update(label=f"Calling tool: `{action.tool}`...")
        tool_input_str = action.tool_input
        if isinstance(tool_input_str, dict): 
            tool_input_str = json.dumps(action.tool_input, indent=2)
        self.log.markdown(f"🛠️ **Calling Tool:** `{action.tool}` with input:\n```json\n{tool_input_str}\n```")
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.status.update(label="Got tool result. Thinking...")
        self.log.markdown(f"👀 **Observation:**\n```text\n{output}\n```")
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.status.update(label="Formulating final answer...")
        self.log.write("✅ **Agent Finished:** Returning final response.")


# --- 1. SPECIALIST CHAINS & TOOLS (Unchanged) ---
def create_looker_agent():
    # (This function is unchanged)
    llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)
    looker_tools = [looker_data_tool, save_to_cache] 
    looker_prompt = hub.pull("hwchase17/openai-tools-agent")
    patch_instructions = (
        "\n\n**CRITICAL WORKFLOW:**\n"
        "1.  Call `LookerDataQuery` to get data and viz URL.\n"
        "2.  The tool returns 'data' (JSON) and 'viz_url'.\n"
        "3.  Invent a descriptive dataset_name and call `save_to_cache`.\n"
        "4.  Construct a 'Final Answer' with a data summary, the dataset_name, and the 'VIZ_URL_TO_RENDER:' string.\n"
    )
    original_system_message = looker_prompt.messages[0].prompt.template
    patched_system_message = original_system_message + patch_instructions
    looker_prompt.messages[0] = SystemMessagePromptTemplate.from_template(patched_system_message)
    agent = create_tool_calling_agent(llm, looker_tools, looker_prompt)
    return AgentExecutor(
        agent=agent, tools=looker_tools, verbose=True, handle_parsing_errors=True
    )
looker_agent_chain = create_looker_agent()

def create_python_agent_chain():
    # (This function is unchanged)
    llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)
    data_cache = st.session_state.get("data_cache", {})
    python_tool_locals = {"pd": pd, "data_cache": data_cache, "StringIO": StringIO, "json": json}
    python_tool = PythonREPLTool(locals=python_tool_locals)
    agent_prompt = hub.pull("hwchase17/openai-tools-agent")
    PYTHON_AGENT_RULES = """
You are an expert Python data analyst... (full rules unchanged: load from data_cache, save plot.png, print PLOT_GENERATED)...
Available datasets: {dataset_names}
"""
    dataset_names = list(data_cache.keys())
    system_message_prefix = PYTHON_AGENT_RULES.format(dataset_names=str(dataset_names))
    original_system_message = agent_prompt.messages[0].prompt.template
    patched_system_message = system_message_prefix + "\n\n" + original_system_message
    agent_prompt.messages[0] = SystemMessagePromptTemplate.from_template(patched_system_message)
    agent = create_tool_calling_agent(llm, [python_tool], agent_prompt)
    return AgentExecutor(agent=agent, tools=[python_tool], verbose=True, handle_parsing_errors=True)

knowledge_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0).bind_tools([VertexTool(google_search={})])
knowledge_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on history and your knowledge. Use search if needed."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
knowledge_chain = knowledge_prompt | knowledge_llm | StrOutputParser()

social_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0.7)
social_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly, conversational AI. Respond to the user's social message."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
social_chain = social_prompt | social_llm | StrOutputParser()


# --- 2. THE ROUTER (PROMPT RE-ORDERED FOR EMPHASIS) ---

# 1. Define the Pydantic schema (Unchanged)
class RouteQuery(BaseModel):
    """The classification decision for routing a user query."""
    intent: Literal['looker', 'python_analysis', 'knowledge', 'social'] = Field(
        ..., 
        description="The classified intent based on the user's question and history."
    )

# 2. Create the prompt (THIS IS THE FIX)
# We are moving the HARD RULES to the BOTTOM of the prompt, just before the response.
router_prompt_template = """Your sole task is to classify the user's intent from their latest question, based on the chat history.
You must select one of four intents: ['looker', 'python_analysis', 'social', 'knowledge'].

Chat History:
{chat_history}
---
User Question: {input}
---
Examples (Review these patterns):
User Question: What is the total population? -> classify as 'looker'
User Question: population in california -> classify as 'looker'
User Question: what is the total male vs female population in the US? -> classify as 'looker'
User Question: Male vs female population rates in the US from census data in Looker? -> classify as 'looker'
User Question: how many high school students are in California? -> classify as 'looker'
User Question: Now plot the data I just got -> classify as 'python_analysis'
User Question: calculate the average of `female_pop_by_state` -> classify as 'python_analysis'
User Question: hi -> classify as 'social'
User Question: what is the capital of france? -> classify as 'knowledge'
User Question: are there telus offices in the us? -> classify as 'knowledge'
User Question: how does that compare to the world? -> classify as 'knowledge'
---
**CRITICAL ROUTING RULES (You MUST follow these):**
1.  **'social'**: Does the question look like a greeting, farewell, or simple chit-chat?
2.  **'python_analysis'**: Does the user ask to 'plot', 'calculate', 'analyze', or refer to a saved dataset name?
3.  **'looker'**: Does the question ask about ANY US census data, population, demographics (race, gender, age), or income?
4.  **RULE 3 OVERRIDE:** ALL questions about US demographics MUST go to 'looker', even if Google Search could also answer them. This is the primary data source. NO EXCEPTIONS.
5.  **'knowledge'**: This is the DEFAULT fallback ONLY if the query does not match social, python, or looker.

Now, classify the user's question based on these rules and examples.
"""
router_prompt = ChatPromptTemplate.from_template(router_prompt_template)
router_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)

# 3. Create the router chain (Unchanged)
router_chain = router_prompt | router_llm.with_structured_output(RouteQuery)
# --- END ROUTER REWRITE ---


# --- 3. THE MAIN CHAIN (RunnableBranch - Unchanged) ---
def route(info):
    intent = info.get("intent")
    user_input = info.get("input") 
    chat_history_messages = info.get("chat_history_messages") 
    
    if intent == 'looker':
        return looker_agent_chain.invoke({"input": user_input, "chat_history": chat_history_messages})
    elif intent == 'python_analysis':
        python_agent = create_python_agent_chain() 
        return python_agent.invoke({"input": user_input}) # No history passed to Python agent
    elif intent == 'social':
        return social_chain.invoke({"input": user_input, "chat_history": chat_history_messages})
    else: 
        return knowledge_chain.invoke({"input": user_input, "chat_history": chat_history_messages})

full_chain = (
    {
        "intent_obj": router_chain,
        "input": lambda x: x["input"],
        "chat_history_messages": lambda x: x["chat_history"]
    }
    | RunnableLambda(lambda x: {"intent": x["intent_obj"].intent, "input": x["input"], "chat_history_messages": x["chat_history_messages"]})
    | RunnableLambda(route)
    | RunnableLambda(lambda result: result.get("output") if isinstance(result, dict) else result) 
)


# --- 4. STREAMLIT APP (Unchanged) ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}

st.title("🤖 Looker AI Chatbot")
st.caption("I can answer questions about US Census data... or anything else!")

with st.sidebar:
    st.header("Cached Datasets 💾")
    if not st.session_state.data_cache:
        st.info("No datasets cached yet. Ask the bot to query Looker!")
    else:
        st.caption("Available for the Python Analyst:")
        for name, data_json in st.session_state.data_cache.items():
            try:
                data = json.loads(data_json)
                if isinstance(data, list) and len(data) > 0:
                    num_rows = len(data)
                    columns = list(data[0].keys())
                    with st.expander(f"**{name}** ({num_rows} rows)"):
                        st.code(", ".join(columns), language="text")
                elif isinstance(data, list) and len(data) == 0:
                     st.expander(f"**{name}** (0 rows)")
                else: 
                    with st.expander(f"**{name}** (Scalar Value)"):
                        st.json(data)
            except Exception as e:
                st.error(f"Error reading {name}")

msgs = st.session_state.memory.chat_memory.messages
for msg in msgs:
    with st.chat_message(msg.type):
        if "VIZ_URL_TO_RENDER:" in msg.content:
            text_answer, viz_url = msg.content.split("VIZ_URL_TO_RENDER:", 1) 
            st.markdown(text_answer.strip())
            st.markdown(f"_[Test Visualization Link]({viz_url.strip()})_")
            components.iframe(viz_url.strip(), height=500)
        elif "PLOT_GENERATED:" in msg.content:
            text_answer, plot_path = msg.content.split("PLOT_GENERATED:", 1)
            st.markdown(text_answer.strip())
            st.image(plot_path.strip())
        else:
            st.markdown(msg.content)

if prompt := st.chat_input("What would you like to know?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        log_expander = st.expander("View Agent Thought Process")
        status_container = st.status("Agent is routing...", expanded=True)
        
        chat_history = st.session_state.memory.chat_memory.messages
        
        with log_expander:
            st.write("🧠 **Routing...**")
            router_output_obj = router_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            intent = router_output_obj.intent
            st.markdown(f"**Intent Classified:** `{intent}`")
        
        with status_container:
            if intent == 'looker':
                status_container.update(label="Querying Looker Agent...")
                with log_expander:
                    callback_container = st.container()
                    callback = StreamlitCallback(status_container, callback_container)
                response = looker_agent_chain.invoke(
                    {"input": prompt, "chat_history": chat_history},
                    config={"callbacks": [callback]}
                )
                answer = response.get("output")

            elif intent == 'python_analysis':
                status_container.update(label="Running Python Data Analyst...")
                with log_expander:
                    st.write("🐍 **Invoking Python Analyst**")
                    st.write(f"Available datasets: {list(st.session_state.data_cache.keys())}")
                    python_log_container = st.container()
                    callback = StreamlitCallback(status_container, python_log_container)
                
                python_agent = create_python_agent_chain()
                response = python_agent.invoke(
                    {"input": prompt}, # We pass ONLY the prompt, no history
                    config={"callbacks": [callback]}
                )
                answer = response.get("output")

            elif intent == 'social':
                status_container.update(label="Generating social response...")
                answer = social_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })

            else: # Default to knowledge
                status_container.update(label="Searching knowledge base...")
                answer = knowledge_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
        
        status_container.update(label="Task Complete!", state="complete", expanded=False)
        
        if "VIZ_URL_TO_RENDER:" in answer:
            text_answer, viz_url = answer.split("VIZ_URL_TO_RENDER:", 1) 
            st.markdown(text_answer.strip())
            st.markdown(f"_[Test Visualization Link]({viz_url.strip()})_")
            components.iframe(viz_url.strip(), height=500)
        elif "PLOT_GENERATED:" in answer:
            text_answer, plot_path = answer.split("PLOT_GENERATED:", 1)
            st.markdown(text_answer.strip())
            st.image(plot_path.strip())
        else:
            st.markdown(answer)
        
        st.session_state.memory.chat_memory.add_user_message(prompt)
        st.session_state.memory.chat_memory.add_ai_message(answer)
        
        st.rerun()