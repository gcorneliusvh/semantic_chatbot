import os
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict

# --- Configuration ---
CACHE_DIR = "multi_dataset_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Tool Input Schemas ---g
class SaveCacheInput(BaseModel):
    """Input schema for the save_data_to_cache tool."""
    # This tells Pydantic to allow complex objects like DataFrames.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame
    dataset_name: str = Field(description="A descriptive, snake_case name for the dataset, e.g., 'population_by_state' or 'sales_revenue'.")

class LoadCacheInput(BaseModel):
    """Input schema for the load_dataframes_from_cache tool."""
    dataset_names: List[str] = Field(description="A list of dataset names to load from the cache.")


@tool(args_schema=SaveCacheInput)
def save_data_to_cache(dataframe: pd.DataFrame, dataset_name: str) -> str:
    """Saves a pandas DataFrame to a named CSV file in the cache directory."""
    try:
        file_path = os.path.join(CACHE_DIR, f"{dataset_name}.csv")
        dataframe.to_csv(file_path, index=False)
        return f"Successfully saved dataset '{dataset_name}' with {len(dataframe)} rows."
    except Exception as e:
        return f"Error saving data: {e}"

@tool
def list_cached_datasets() -> List[str]:
    """Returns a list of all available dataset names in the cache."""
    return [f.replace('.csv', '') for f in os.listdir(CACHE_DIR) if f.endswith('.csv')]

@tool(args_schema=LoadCacheInput)
def load_dataframes_from_cache(dataset_names: List[str]) -> Dict[str, str]:
    """
    Loads one or more named datasets from the cache into a dictionary of
    DataFrames, returned as JSON strings.
    """
    loaded_data = {}
    for name in dataset_names:
        file_path = os.path.join(CACHE_DIR, f"{name}.csv")
        try:
            df = pd.read_csv(file_path)
            # Return as JSON string for the agent to handle
            loaded_data[name] = df.to_json(orient='records')
        except FileNotFoundError:
            loaded_data[name] = f"Error: Dataset '{name}' not found in cache."
    return loaded_data