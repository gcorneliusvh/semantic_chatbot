import streamlit as st
import pandas as pd
import os
import uuid
import json
import traceback
import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- Import and initialize the configuration first ---
# This ensures credentials are loaded before any other module that might need them.
from config import config

# --- Setup Logging ---
# This will create an 'app.log' file in your project root.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # Log to a file
        logging.StreamHandler()        # Log to the console/terminal
    ]
)

# Now that config is loaded, we can import the other modules
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
        # Add feedback buttons to assistant messages, but not the very first one
        if message["role"] == "assistant" and i > 0:
            feedback_key_base = f"feedback_{i}"
            col1, col2, _ = st.columns([1, 1, 10]) 
            with col1:
                if st.button("👍", key=f"{feedback_key_base}_pos"):
                    # The user's prompt is the message before the assistant's response
                    user_prompt = st.session_state.messages[i-1]['content']
                    save_feedback(st.session_state.session_id, user_prompt, message['content'], "positive")
                    st.toast("Thanks for your feedback!")
            with col2:
                if st.button("👎", key=f"{feedback_key_base}_neg"):
                    user_prompt = st.session_state.messages[i-1]['content']
                    save_feedback(st.session_state.session_id, user_prompt, message['content'], "negative")
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
        chain_input = {
            "original_question": prompt,
            "messages": [HumanMessage(content=prompt)]
        }

        final_answer_container = st.empty()
        final_answer = ""
        
        try:
            logging.info(f"Invoking agent for session: {st.session_state.session_id}")
            
            # Stream events from the graph to update the UI in real-time
            for event in agentic_chain.stream(chain_input, config=config, stream_mode="values"):
                if "plan" in event and event["plan"]:
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
                        except (json.JSONDecodeError, TypeError):
                            log_container.text(last_message.content)

                if "final_answer" in event and event["final_answer"]:
                    final_answer = event["final_answer"]
                    final_answer_container.markdown(final_answer)

        except Exception as e:
            # --- ENHANCED ERROR REPORTING ---
            # Log the full error and traceback to the file and console
            logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
            
            # Display a user-friendly message and the technical details in the UI
            st.error("I'm sorry, I encountered a critical error. The technical details are below, and have been saved to `app.log` for review.")
            tb_str = traceback.format_exc()
            st.code(tb_str, language="text")
            
            final_answer = "I was unable to complete the request due to an internal error."
            final_answer_container.markdown(final_answer)
            # --- END ENHANCEMENT ---

    # Append the final answer to the message history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    st.rerun()
