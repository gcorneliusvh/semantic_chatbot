# In app.py
import streamlit as st
import json 
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain_core.runnables import RunnableBranch, RunnableLambda
from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool

# Import all tool files (we need their objects)
from tools.looker_tool import looker_data_tool
from tools.social_tool import social_tool, _social_chat  # We need the function for the new chain
from tools.knowledge_tool import general_knowledge_tool, _run_general_knowledge # We need the function

# --- CALLBACK HANDLER CLASS (This is back!) ---
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from typing import Any, Dict, List

class StreamlitCallback(BaseCallbackHandler):
    """Callback handler to display agent thoughts in Streamlit UI."""
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


# --- 1. SPECIALIST CHAINS & TOOLS (NEW ARCHITECTURE) ---

# LOOKER SPECIALIST (This is our full-power agent)
def create_looker_agent():
    llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)
    looker_tools = [looker_data_tool]
    looker_prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, looker_tools, looker_prompt)
    return AgentExecutor(
        agent=agent, tools=looker_tools, verbose=True, handle_parsing_errors=True
    )
looker_agent_chain = create_looker_agent()

# KNOWLEDGE SPECIALIST (New history-aware chain)
knowledge_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0).bind_tools([VertexTool(google_search={})])
knowledge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question based on the chat history and your knowledge. Use your search tool if you don't know the answer."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
knowledge_chain = knowledge_prompt | knowledge_llm | StrOutputParser()

# SOCIAL SPECIALIST (New history-aware chain)
social_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0.7)
social_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly, conversational AI. Respond to the user's social message, keeping the chat history in mind."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
social_chain = social_prompt | social_llm | StrOutputParser()


# --- 2. THE ROUTER (Unchanged) ---
router_prompt_template = """
You are an expert dispatcher. Your job is to classify a user's new question into one of three categories based on the question AND the chat history.
Your output MUST be a JSON object with a single key, "intent".
The possible intents are: 'looker', 'knowledge', or 'social'.

- 'looker': Use for ANY questions about US population, census data, demographics, income, etc. This is the primary data source.
- 'knowledge': Use for general knowledge, company facts (like Telus), questions that are NOT about US census data, OR follow-up questions to compare census data to the wider world.
- 'social': Use for greetings, goodbyes, and conversational chit-chat.

Chat History:
{chat_history}

User Question: {input}

Examples:
User Question: What is the total population? -> {{"intent": "looker"}}
User Question: population in california -> {{"intent": "looker"}}
User Question: hi -> {{"intent": "social"}}
User Question: what is the capital of france? -> {{"intent": "knowledge"}}
User Question: how are you? -> {{"intent": "social"}}
User Question: median income by county in texas -> {{"intent": "looker"}}
User Question: are there telus offices in the us? -> {{"intent": "knowledge"}}
User Question: how does that compare to the world? -> {{"intent": "knowledge"}}

Classification:
"""
router_prompt = ChatPromptTemplate.from_template(router_prompt_template)
router_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)
router_chain = router_prompt | router_llm | JsonOutputParser()


# --- 3. STREAMLIT APP ---
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
        st.markdown(msg.content)

# --- 4. NEW CHAT LOOP (MANUAL ROUTING + LIVE THINKING LOG) ---
if prompt := st.chat_input("What would you like to know?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_history = st.session_state.memory.chat_memory.messages
        
        # We replace the simple spinner with our full logging UI
        log_expander = st.expander("View Agent Thought Process")
        status_container = st.status("Agent is routing...", expanded=True)
        
        # 1. Run the Router to get the intent
        with log_expander:
            st.write("🧠 **Routing...**")
            router_output = router_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            intent = router_output.get("intent", "knowledge") # Default to knowledge if router fails
            st.markdown(f"**Intent Classified:** `{intent}`")
        
        # 2. Run the correct specialist based on the intent
        with status_container:
            if intent == 'looker':
                status_container.update(label="Querying Looker Agent...")
                with log_expander:
                    # We can use our callback handler ONLY for the Looker agent
                    callback = StreamlitCallback(status_container, st.container())
                
                response = looker_agent_chain.invoke(
                    {"input": prompt, "chat_history": chat_history},
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
        
        # 3. Show the final UI
        status_container.update(label="Task Complete!", state="complete", expanded=False)
        st.markdown(answer)
        
        # 4. Save to memory
        st.session_state.memory.chat_memory.add_user_message(prompt)
        st.session_state.memory.chat_memory.add_ai_message(answer)