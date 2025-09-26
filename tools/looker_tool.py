import os
import json
import pandas as pd
import io
from urllib.parse import urlencode
import configparser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import looker_sdk
from looker_sdk import models40

# Assuming cache_tool is in the same directory or accessible
from .cache_tool import save_data_to_cache

# --- Pydantic Schema for LLM Input ---
class LookerQueryInput(BaseModel):
    fields: List[str] = Field(..., description="List of dimensions and measures, e.g., ['state.state_name', 'blockgroup.total_pop']")
    filters: Optional[Dict[str, str]] = Field(default={}, description="Dictionary of filters, e.g., {'state.state_name': 'California'}")
    sorts: Optional[List[str]] = Field(default=[], description="List of fields to sort by, e.g., ['blockgroup.total_pop desc']")
    limit: Optional[str] = Field(default="500", description="Row limit for the query")
    vis_config_string: str = Field(default='{"type": "table"}', description="A valid Looker vis_config JSON object, as a string.")
    dataset_name: str = Field(..., description="A descriptive, snake_case name to save the output dataset, e.g., 'population_by_state'.")

# --- Corrected Looker Query Function ---
def run_looker_query(
    model_name: str,
    explore_name: str,
    fields: List[str],
    dataset_name: str,
    filters: Optional[Dict[str, str]] = None,
    sorts: Optional[List[str]] = None,
    limit: Optional[str] = "500",
    vis_config_string: str = '{"type": "table"}',
) -> str:
    """
    Executes a dynamic query against a specified Looker model and explore using a looker.ini file.
    """
    # --- FIX: Input Validation ---
    if not fields:
        return json.dumps({"error": "Query failed: The 'fields' parameter cannot be empty. You must provide at least one dimension or measure."})

    filters = filters or {}
    sorts = sorts or []

    try:
        # --- FIX: Use an absolute path to the looker.ini file ---
        # This makes the path relative to this file, not the execution directory
        ini_file_path = os.path.join(os.path.dirname(__file__), '..', 'looker.ini')
        sdk = looker_sdk.init40(ini_file_path)
    except Exception as e:
        return json.dumps({"error": f"Looker SDK initialization failed. Ensure looker.ini is in the project root and is correct. Error: {e}"})
        
    try:
        query_payload = models40.WriteQuery(
            model=model_name, view=explore_name, fields=fields, filters=filters,
            sorts=sorts, limit=str(limit), vis_config=json.loads(vis_config_string)
        )
        data_result = sdk.run_inline_query(result_format="json", body=query_payload)
        
        config = configparser.ConfigParser()
        config.read(ini_file_path)
        base_url = config.get('Looker', 'browser_url', fallback="")

        if not base_url:
            print("Warning: 'browser_url' not set in looker.ini. Visualization URL will be incomplete.")
        
        query_string = urlencode({
            'fields': ",".join(fields), 'sorts': ",".join(sorts), 'limit': str(limit),
            'vis_config': vis_config_string, 'toggle': 'vis',
            **{f"f[{k}]": v for k, v in filters.items()}
        })
        viz_url = f"{base_url}/embed/explore/{model_name}/{explore_name}?{query_string}"
        
        df = pd.read_json(io.StringIO(data_result))
        cache_result = save_data_to_cache.func(dataframe=df, dataset_name=dataset_name)
        
        return json.dumps({
            "status": "Success", "summary": cache_result, "viz_url": viz_url,
            "data_preview": df.head().to_dict('records'), "data_stats": df.describe().to_dict()
        })
    except Exception as e:
        return json.dumps({"error": f"Error running Looker query: {str(e)}"})