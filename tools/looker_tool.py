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
from .cache_tool import save_data_to_cache

# --- Pydantic Schema for LLM Input ---
class LookerQueryInput(BaseModel):
    fields: List[str] = Field(..., description="List of dimensions and measures, e.g., ['state.state_name', 'blockgroup.total_pop']")
    filters: Optional[Dict[str, str]] = Field(default={}, description="Dictionary of filters, e.g., {'state.state_name': 'California'}")
    sorts: Optional[List[str]] = Field(default=[], description="List of fields to sort by, e.g., ['blockgroup.total_pop desc']")
    limit: Optional[str] = Field(default="500", description="Row limit for the query")
    vis_config_string: str = Field(default='{"type": "table"}', description="A valid Looker vis_config JSON object, as a string.")
    dataset_name: str = Field(..., description="A descriptive, snake_case name to save the output dataset, e.g., 'population_by_state'.")

# --- Generic Looker Query Function (Internal) ---
def _run_looker_query_generic(
    model_name: str,
    explore_name: str,
    fields: List[str],
    filters: Optional[Dict[str, str]],
    sorts: Optional[List[str]],
    limit: Optional[str],
    vis_config_string: str,
    dataset_name: str
) -> str:
    """Internal function that executes the Looker query."""
    sdk = looker_sdk.init40()
    if not sdk:
        return json.dumps({"error": "Looker SDK not initialized."})
    try:
        query_payload = models40.WriteQuery(
            model=model_name, view=explore_name, fields=fields, filters=filters,
            sorts=sorts, limit=str(limit), vis_config=json.loads(vis_config_string)
        )
        data_result = sdk.run_inline_query(result_format="json", body=query_payload)
        base_url = os.environ.get("LOOKERSDK_BROWSER_URL", "").replace(":19999", "")
        query_string = urlencode({
            'fields': ",".join(fields), 'sorts': ",".join(sorts), 'limit': str(limit),
            'vis_config': vis_config_string, 'toggle': 'vis',
            **{f"f[{k}]": v for k, v in filters.items()}
        })
        viz_url = f"{base_url}/embed/explore/{model_name}/{explore_name}?{query_string}"
        df = pd.read_json(io.StringIO(data_result))
        
        # Call the tool directly using .func() to pass the DataFrame
        cache_result = save_data_to_cache.func(dataframe=df, dataset_name=dataset_name)
        
        return json.dumps({
            "status": "Success",
            "summary": cache_result,
            "viz_url": viz_url
        })
    except Exception as e:
        return json.dumps({"error": f"Error running Looker query: {str(e)}"})

# --- NEW: Robust Wrapper Function for the Tool ---
def run_looker_tool_from_model(model_name: str, explore_name: str, query_input: LookerQueryInput) -> str:
    """
    A robust wrapper that takes the Pydantic model directly and calls the
    underlying query function. This prevents TypeErrors from missing optional args.
    """
    return _run_looker_query_generic(
        model_name=model_name,
        explore_name=explore_name,
        fields=query_input.fields,
        filters=query_input.filters,
        sorts=query_input.sorts,
        limit=query_input.limit,
        vis_config_string=query_input.vis_config_string,
        dataset_name=query_input.dataset_name
    )