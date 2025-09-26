import streamlit as st
import pandas as pd
import os
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- Import the new LangGraph agentic chain and feedback handler ---
from core_logic_langgraph import agentic_chain
from feedback_handler import save_feedback

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
        with st.sidebar.expander(f"📄 {file_name.replace('.csv', '')}"):
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

# --- Display Chat History & Feedback Buttons ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            feedback_key_base = f"feedback_{i}"
            
            # Use columns to place buttons side-by-side
            col1, col2, _ = st.columns([1, 1, 10]) 
            
            with col1:
                if st.button("👍", key=f"{feedback_key_base}_pos"):
                    save_feedback(st.session_state.session_id, st.session_state.messages[i-1]['content'], message['content'], "positive")
                    st.toast("Thanks for your feedback!")

            with col2:
                if st.button("👎", key=f"{feedback_key_base}_neg"):
                    save_feedback(st.session_state.session_id, st.session_state.messages[i-1]['content'], message['content'], "negative")
                    st.toast("Thanks for your feedback!")

# --- Handle User Input ---
if prompt := st.chat_input("Ask a complex, multi-step data question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking_expander = st.expander("🤔 Agent Thought Process...")
        log_container = thinking_expander.container()
        
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        # Prepare the initial input for the graph
        # For a new question, the message history is empty for the graph's perspective
        chain_input = {
            "original_question": prompt,
            "messages": [HumanMessage(content=prompt)]
        }

        final_answer_container = st.empty()
        final_answer = ""
        
        try:
            # Stream events from the graph to update the UI in real-time
            for event in agentic_chain.stream(chain_input, config=config, stream_mode="values"):
                if "plan" in event:
                    log_container.markdown("### 📝 **Plan**")
                    log_container.markdown(event["plan"])
                
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if isinstance(last_message, AIMessage) and last_message.tool_calls:
                        log_container.markdown(f"### 📞 **Calling Tool: `{last_message.tool_calls[0]['name']}`**")
                        log_container.json(last_message.tool_calls[0]['args'])
                        
                    elif isinstance(last_message, ToolMessage):
                        log_container.markdown(f"### 🛠️ **Tool Output: `{last_message.name}`**")
                        try:
                            tool_output = json.loads(last_message.content)
                            log_container.json(tool_output)
                        except json.JSONDecodeError:
                            log_container.text(last_message.content)

                if "final_answer" in event and event["final_answer"]:
                    final_answer = event["final_answer"]
                    final_answer_container.markdown(final_answer)

        except Exception as e:
            st.error(f"An error occurred while running the agent: {e}")
            final_answer = "I'm sorry, I encountered a critical error. Please check the logs."
            final_answer_container.markdown(final_answer)

    # Append the final answer to the message history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    st.rerun()