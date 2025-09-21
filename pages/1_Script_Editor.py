import streamlit as st
import os
import pandas as pd
import io
import contextlib

# --- Configuration ---
SCRIPT_DIR = "user_scripts"
CREATE_NEW_PROMPT = "--- Create New Script ---"
CACHE_FILE = "data.csv"

# Ensure the script directory exists
os.makedirs(SCRIPT_DIR, exist_ok=True)

st.set_page_config(page_title="Script Editor", layout="wide")
st.title("🐍 Python Script Editor")
st.warning(
    "**Warning:** Scripts are run locally with full permissions using `exec()`. "
    "This is a powerful feature for personal use but has security risks. "
    "Only run trusted code."
)

# --- Helper Functions ---

def get_scripts():
    """Returns a list of all .py files in the script directory."""
    return [f for f in os.listdir(SCRIPT_DIR) if f.endswith(".py")]

def load_data():
    """Loads the cached data.csv file if it exists."""
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return None

# --- Sidebar: Script Selection ---
st.sidebar.title("Script Manager")
scripts = get_scripts()
selected_script = st.sidebar.selectbox("Load Script", [CREATE_NEW_PROMPT] + scripts)

script_filename = ""
script_content = ""

if selected_script == CREATE_NEW_PROMPT:
    script_filename = st.sidebar.text_input("New Script Name (e.g., 'my_analysis.py')")
    script_content = (
        "# Enter your Python script here.\n"
        "# The cached data will be available in a pandas DataFrame named 'df'.\n"
        "# You can also use pandas (as pd) and streamlit (as st).\n\n"
        "if df is not None:\n"
        "    print(df.describe())\n"
        "else:\n"
        "    print('No data.csv found. Run a Looker query in the chatbot.')\n"
    )
else:
    script_filename = selected_script
    try:
        with open(os.path.join(SCRIPT_DIR, script_filename), "r") as f:
            script_content = f.read()
    except Exception as e:
        st.error(f"Error loading script: {e}")

# --- Main Page: Editor and Controls ---
st.subheader(f"Editing: `{script_filename if script_filename else 'New Script'}`")

# Use a key to force re-render when script changes
editor_key = f"editor_{selected_script}"
content_to_edit = st.text_area("Script Content", script_content, height=500, key=editor_key)

col1, col2, col3 = st.columns([1, 1, 1])

# --- Button Logic ---
with col1:
    if st.button("💾 Save Script", disabled=not script_filename):
        if not script_filename.endswith(".py"):
            script_filename += ".py"
        
        try:
            filepath = os.path.join(SCRIPT_DIR, script_filename)
            with open(filepath, "w") as f:
                f.write(content_to_edit)
            st.success(f"Saved script: {script_filename}")
            st.rerun() # Refresh the page to show new script in list
        except Exception as e:
            st.error(f"Error saving script: {e}")

with col2:
    if st.button("▶️ Run Script", disabled=(selected_script == CREATE_NEW_PROMPT and not script_filename)):
        st.subheader("Script Output")
        
        # Load the data
        df = load_data()
        
        if df is None:
            st.warning("No `data.csv` cache found. Running script without 'df'.")
        
        # Capture stdout to display print() statements
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                # Execute the script. Pass 'df', 'pd', and 'st' into its scope.
                exec(content_to_edit, {"df": df, "pd": pd, "st": st})
            except Exception as e:
                st.exception(e) # Print the full exception
        
        st.code(f.getvalue(), language="text")

with col3:
    if st.button("🗑️ Delete Script", disabled=(selected_script == CREATE_NEW_PROMPT)):
        try:
            filepath = os.path.join(SCRIPT_DIR, selected_script)
            os.remove(filepath)
            st.success(f"Deleted script: {selected_script}")
            st.rerun() # Refresh the page
        except Exception as e:
            st.error(f"Error deleting script: {e}")
