# In tools/cache_tool.py
import streamlit as st
import json
from langchain.tools import tool

@tool
def save_to_cache(dataset_name: str, data_json: str) -> str:
    """
    Saves a JSON dataset (passed as a string) to the in-memory cache
    under a given dataset_name.
    """
    if "data_cache" not in st.session_state:
        st.session_state.data_cache = {}
    try:
        # We store the raw JSON data, not the parsed object.
        # This gives the python agent more flexibility.
        st.session_state.data_cache[dataset_name] = data_json
        return f"Successfully saved dataset as '{dataset_name}' in the data_cache."
    except Exception as e:
        return f"Error saving to cache: {e}"