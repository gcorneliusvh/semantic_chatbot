import streamlit as st
import pandas as pd
import os
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage

# --- Import the new LangGraph agentic chain ---
from core_logic_langgraph import agentic_chain

# ==============================================================================
# Streamlit UI
# ==============================================================================

def setup_sidebar():
    """Configures the sidebar to show the new multi-dataset cache."""
    st.sidebar.title("Cached Datasets")
    cache_dir = "multi_dataset_cache"
    if not os.path.exists(cache_dir):
        st.sidebar.info("No cache directory found.")
        return

    cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".csv")]
    if not cached_files:
        st.sidebar.info("No datasets have been cached yet.")
        return

    for file_name in cached_files:
        with st.sidebar.expander(f"📄 {file_name}"):
            try:
                df = pd.read_csv(os.path.join(cache_dir, file_name))
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Could not load {file_name}: {e}")

# --- Main Page Setup ---
st.set_page_config(page_title="Agentic Semantic Chatbot", layout="wide")
st.title("Looker & Gemini Agentic Analyst 🤖")

setup_sidebar()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask a complex, multi-step data question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create a container for the agent's thought process
        thinking_expander = st.expander("🤔 Agent Thought Process...")
        log_container = thinking_expander.container()
        
        # Prepare inputs for the LangGraph chain
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        chain_input = {"original_question": prompt}

        final_answer = ""
        
        try:
            # Use write_stream to display the agent's process in real-time
            def stream_agent_output():
                for event in agentic_chain.stream(chain_input, config=config, stream_mode="values"):
                    # event is a dictionary representing the current state of the graph
                    if "planner" in event:
                        plan = event["planner"].get("plan", "")
                        if plan:
                            log_container.markdown("### 📝 **Plan**")
                            log_container.markdown(plan)
                    
                    if "messages" in event and event["messages"]:
                        last_message = event["messages"][-1]
                        if last_message.type == "tool":
                            log_container.markdown(f"### 🛠️ **Tool Output: `{last_message.name}`**")
                            try:
                                # Prettify the JSON output for readability
                                tool_output = json.loads(last_message.content)
                                log_container.json(tool_output)
                            except json.JSONDecodeError:
                                log_container.text(last_message.content)

                    if "final_answer" in event and event["final_answer"]:
                        # Yield the final answer to the main chat window
                        yield event["final_answer"]

            # Stream the final answer to the main chat message window
            final_answer = st.write_stream(stream_agent_output)

        except Exception as e:
            st.error(f"An error occurred while running the agent: {e}")
            final_answer = "I'm sorry, I encountered a critical error. Please check the logs."

    # Append the final answer to the message history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    st.rerun()