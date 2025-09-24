# tools/cache_tool.py (Upgraded)
import os
import pandas as pd
from langchain_core.tools import tool

CACHE_DIR = "multi_dataset_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class CacheInput(BaseModel):
    dataframe: pd.DataFrame
    dataset_name: str = Field(description="A descriptive, snake_case name for the dataset, e.g., 'population_by_state' or 'sales_revenue'.")

@tool(args_schema=CacheInput)
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

class LoadCacheInput(BaseModel):
    dataset_names: List[str] = Field(description="A list of dataset names to load from the cache.")

@tool(args_schema=LoadCacheInput)
def load_dataframes_from_cache(dataset_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Loads one or more named datasets from the cache into a dictionary of pandas DataFrames."""
    loaded_data = {}
    for name in dataset_names:
        file_path = os.path.join(CACHE_DIR, f"{name}.csv")
        try:
            loaded_data[name] = pd.read_csv(file_path)
        except FileNotFoundError:
            # In a real agent, it would see this and know the data needs to be fetched first
            pass 
    return loaded_data