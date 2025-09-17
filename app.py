# In app.py
import streamlit as st
import streamlit.components.v1 as components
import json 
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain_core.runnables import RunnableBranch, RunnableLambda
from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool

# Import all tool files
from tools.looker_tool import looker_data_tool
from tools.social_tool import social_tool
from tools.knowledge_tool import general_knowledge_tool

# --- CALLBACK HANDLER CLASS (Unchanged) ---
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from typing import Any, Dict, List

class StreamlitCallback(BaseCallbackHandler):
    def __init__(self, status_container, log_container):
        self.status = status_container
        self.log = log_container

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.status.update(label="Looker Agent is thinking...")
        self.log.write("🤔 **Thinking...**")
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
# --- END CALLBACK CLASS ---


# --- 1. SPECIALIST CHAINS & TOOLS (Unchanged) ---

def create_looker_agent():
    llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)
    looker_tools = [looker_data_tool]
    looker_prompt = hub.pull("hwchase17/structured-chat-agent")
    patch_instructions = (
        "\n\n**CRITICAL VIZ RULES:**\n"
        "1. The LookerDataQuery tool will return a JSON string with two keys: 'data' (the data results) and 'viz_url' (an embeddable URL).\n"
        "2. Your job is to create a final text answer. First, summarize the 'data' in a natural language response.\n"
        "3. After the text answer, you MUST add a new line and include the 'magic string' `VIZ_URL_TO_RENDER:` followed IMMEDIATELY by the URL from the 'viz_url' key.\n"
        "**Example:**\n"
        "The total population is 326,289,971.\n"
        "VIZ_URL_TO_RENDER:https://my.looker.com/embed/explore/model/explore?qid=abcdefg"
    )
    original_system_message = looker_prompt.messages[0].prompt.template
    patched_system_message = original_system_message + patch_instructions
    looker_prompt.messages[0] = SystemMessagePromptTemplate.from_template(patched_system_message)
    agent = create_structured_chat_agent(llm, looker_tools, looker_prompt)
    return AgentExecutor(
        agent=agent, tools=looker_tools, verbose=True, handle_parsing_errors=True
    )

looker_agent_chain = create_looker_agent()

knowledge_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0).bind_tools([VertexTool(google_search={})])
knowledge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question based on the chat history and your knowledge. Use your search tool if you don't know the answer."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
knowledge_chain = knowledge_prompt | knowledge_llm | StrOutputParser()

social_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0.7)
social_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly, conversational AI. Respond to the user's social message, keeping the chat history in mind."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
social_chain = social_prompt | social_llm | StrOutputParser()


# --- 2. THE ROUTER (WITH NEW EXAMPLE ADDED) ---

# This is the updated prompt template
router_prompt_template = """Your sole task is to classify the user's intent from their latest question, based on the chat history.
You MUST respond with ONLY a valid JSON object containing a single key, "intent".
The possible values for "intent" are: 'looker', 'knowledge', or 'social'.

Here are the classification rules:
- 'looker': Use for ANY questions about US population, census data, demographics, income, etc. This is the primary data source.
- 'knowledge': Use for general knowledge, company facts (like Telus), questions that are NOT about US census data, OR follow-up questions to compare census data to the wider world.
- 'social': Use for greetings, goodbyes, and conversational chit-chat.

---
Chat History:
{chat_history}
---
User Question: {input}
---
Examples:
User Question: What is the total population? -> {{"intent": "looker"}}
User Question: population in california -> {{"intent": "looker"}}
User Question: median income by county in texas -> {{"intent": "looker"}}
User Question: what is the female population by state in the US? -> {{"intent": "looker"}}
User Question: hi -> {{"intent": "social"}}
User Question: what is the capital of france? -> {{"intent": "knowledge"}}
User Question: are there telus offices in the us? -> {{"intent": "knowledge"}}
User Question: how does that compare to the world? -> {{"intent": "knowledge"}}
---
JSON Response:
"""

router_prompt = ChatPromptTemplate.from_template(router_prompt_template)
router_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)
router_chain = router_prompt | router_llm | JsonOutputParser()


# --- 3. THE MAIN CHAIN (RunnableBranch - Unchanged) ---
def route(info):
    intent = info.get("intent")
    user_input = info.get("input") 
    chat_history_messages = info.get("chat_history_messages") 
    
    if intent == 'looker':
        return looker_agent_chain.invoke({"input": user_input, "chat_history": chat_history_messages})
    elif intent == 'social':
        return social_chain.invoke({"input": user_input, "chat_history": chat_history_messages})
    else: # Default to knowledge
        return knowledge_chain.invoke({"input": user_input, "chat_history": chat_history_messages})

full_chain = (
    {
        "intent_json": router_chain,
        "input": lambda x: x["input"],
        "chat_history_messages": lambda x: x["chat_history"]
    }
    | RunnableLambda(lambda x: {"intent": x["intent_json"].get("intent"), "input": x["input"], "chat_history_messages": x["chat_history_messages"]})
    | RunnableLambda(route)
    | RunnableLambda(lambda result: result.get("output") if isinstance(result, dict) else result) 
)


# --- 4. STREAMLIT APP (Unchanged) ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

st.title("🤖 Looker AI Chatbot")
st.caption("I can answer questions about US Census data... or anything else!")

msgs = st.session_state.memory.chat_memory.messages
for msg in msgs:
    with st.chat_message(msg.type):
        if "VIZ_URL_TO_RENDER:" in msg.content:
            text_answer, viz_url = msg.content.split("VIZ_URL_TO_RENDER:", 1) 
            st.markdown(text_answer.strip())
            st.markdown(f"_[Test Visualization Link]({viz_url.strip()})_")
            components.iframe(viz_url.strip(), height=500)
        else:
            st.markdown(msg.content)

if prompt := st.chat_input("What would you like to know?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        log_expander = st.expander("View Agent Thought Process")
        status_container = st.status("Agent is routing...", expanded=True)
        
        with log_expander:
            st.write("🧠 **Routing...**")
            router_output = router_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.memory.chat_memory.messages
            })
            intent = router_output.get("intent", "knowledge") 
            st.markdown(f"**Intent Classified:** `{intent}`")
        
        with status_container:
            if intent == 'looker':
                status_container.update(label="Querying Looker Agent...")
                with log_expander:
                    callback = StreamlitCallback(status_container, st.container())
                response = looker_agent_chain.invoke(
                    {"input": prompt, "chat_history": st.session_state.memory.chat_memory.messages},
                    config={"callbacks": [callback]}
                )
                answer = response.get("output")

            elif intent == 'social':
                status_container.update(label="Generating social response...")
                answer = social_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.memory.chat_memory.messages
                })

            else: # Default to knowledge
                status_container.update(label="Searching knowledge base...")
                answer = knowledge_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.memory.chat_memory.messages
                })
        
        status_container.update(label="Task Complete!", state="complete", expanded=False)
        
        if "VIZ_URL_TO_RENDER:" in answer:
            text_answer, viz_url = answer.split("VIZ_URL_TO_RENDER:", 1) 
            st.markdown(text_answer.strip())
            st.markdown(f"_[Test Visualization Link]({viz_url.strip()})_")
            components.iframe(viz_url.strip(), height=500)
        else:
            st.markdown(answer)
        
        st.session_state.memory.chat_memory.add_user_message(prompt)
        st.session_state.memory.chat_memory.add_ai_message(answer)