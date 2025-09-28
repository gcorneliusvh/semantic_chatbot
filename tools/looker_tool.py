import os
import json
import pandas as pd
import io
from urllib.parse import urlencode
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import looker_sdk
from looker_sdk import models40

# --- Import the centralized config ---
# This ensures that credentials are loaded before the SDK is initialized.
from config import config

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
    Executes a dynamic query against a specified Looker model and explore using a centralized configuration.
    """
    # --- Input Validation ---
    if not fields:
        return json.dumps({"error": "Query failed: The 'fields' parameter cannot be empty. You must provide at least one dimension or measure."})

    filters = filters or {}
    sorts = sorts or []

    try:
        # The SDK will now automatically pick up the credentials from the environment variables
        # loaded by the config module at application startup.
        sdk = looker_sdk.init40()
    except Exception as e:
        return json.dumps({"error": f"Looker SDK initialization failed. Error: {e}"})
        
    try:
        query_payload = models40.WriteQuery(
            model=model_name, view=explore_name, fields=fields, filters=filters,
            sorts=sorts, limit=str(limit), vis_config=json.loads(vis_config_string)
        )
        data_result = sdk.run_inline_query(result_format="json", body=query_payload)
        
        # Get the browser_url from the environment for the embed URL
        base_url = os.environ.get("LOOKERSDK_BROWSER_URL", "")

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
