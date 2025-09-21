import pandas as pd
import streamlit as st
# --- THIS IS THE FIX ---
from langchain_core.tools import tool
# --- END FIX ---

@tool
def save_data_to_cache(dataf: pd.DataFrame, file_path: str = "data_cache.csv") -> str:
    """
    Saves a pandas DataFrame to a CSV file.
    """
    try:
        dataf.to_csv(file_path, index=False)
        return f"Data saved to {file_path}"
    except Exception as e:
        return f"Error saving data: {e}"

@tool
def load_df_from_cache(file_path: str = "data_cache.csv") -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.
    Returns None if the file is not found.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        # Return None so app.py can handle it gracefully
        return None
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None
