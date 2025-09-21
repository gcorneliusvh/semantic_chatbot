import streamlit as st
import os
import json
import pandas as pd
import io
from urllib.parse import urlencode

# --- Corrected Imports ---
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- SDK Imports ---
import looker_sdk
from looker_sdk import models40

# --- Local Tool Import ---
from tools.cache_tool import save_data_to_cache

# ==============================================================================
# Configuration
# ==============================================================================

# --- NEW: Helper to load metadata ---
def _get_explore_metadata():
    """Loads the explore metadata JSON file."""
    try:
        with open("acs_census_metadata.json") as f:
            return json.dumps(json.load(f))
    except FileNotFoundError:
        st.error("Fatal Error: acs_census_metadata.json not found.")
        return ""
    except Exception as e:
        st.error(f"Error loading acs_census_metadata.json: {e}")
        return ""

EXPLORE_METADATA = _get_explore_metadata()
MODEL_NAME = "data_block_acs_bigquery"
EXPLORE_NAME = "acs_census_data"

# ==============================================================================
# SDK Authentication (The stable version)
# ==============================================================================

def _get_looker_sdk() -> looker_sdk.sdk.api40.methods.Looker40SDK:
    """
    Initializes and returns a Looker SDK client using the stable os.environ method.
    """
    try:
        os.environ["LOOKERSDK_BASE_URL"] = st.secrets["looker"]["base_url"]
        os.environ["LOOKERSDK_CLIENT_ID"] = st.secrets["looker"]["client_id"]
        os.environ["LOOKERSDK_CLIENT_SECRET"] = st.secrets["looker"]["client_secret"]
        os.environ["LOOKERSDK_VERIFY_SSL"] = "true"
        os.environ["LOOKERSDK_TIMEOUT"] = "120"
        
        sdk = looker_sdk.init40()
        return sdk
    
    except KeyError as e:
        st.error(f"Missing Looker credential in st.secrets: {e}")
        return None
    except Exception as e:
        st.error(f"Error initializing Looker SDK: {e}")
        return None

# ==============================================================================
# Your Original Pydantic Schema
# ==============================================================================
class LookerQueryInput(BaseModel):
    """Input schema for running a Looker query."""
    fields: List[str] = Field(description="List of dimensions and measures, e.g., ['state.state_name', 'blockgroup.total_pop']")
    filters: Optional[Dict[str, str]] = Field(description="Dictionary of filters, e.g., {'state.state_name': 'California'}", default={})
    sorts: Optional[List[str]] = Field(description="List of fields to sort by, e.g., ['blockgroup.total_pop desc']", default=[])
    limit: Optional[str] = Field(description="Row limit for the query", default="500")
    vis_config_string: str = Field(
        description="A valid Looker vis_config JSON object, as a string. Use a simple 'table' viz if unsure.",
        default='{"type": "table"}'
    )

# ==============================================================================
# Your Original Inline Query Function (Auth and URL logic fixed)
# ==============================================================================

def _run_looker_query(
    fields: List[str], 
    filters: Optional[Dict[str, str]] = {}, 
    sorts: Optional[List[str]] = [], 
    limit: Optional[str] = "500",
    vis_config_string: str = '{"type": "table"}'
) -> str:
    """
    Runs a dynamic query to get data AND builds a full Expanded URL for embedding.
    This is the primary tool for all US Census data questions.
    """
    if not fields:
        return json.dumps({"error": "You must provide at least one field."})

    sdk = _get_looker_sdk()
    if not sdk:
        return json.dumps({"error": "Looker SDK not initialized. Check credentials."})

    try:
        # 1. Create the query payload
        query_payload = models40.WriteQuery(
            model=MODEL_NAME,
            view=EXPLORE_NAME,
            fields=fields,
            filters=filters,
            sorts=sorts,
            limit=str(limit),
            vis_config=json.loads(vis_config_string) # Parse string to dict for the SDK
        )
        
        # 2. Run the query inline to get the data
        print(f"--- Running Looker Inline Query --- \n{query_payload}")
        data_result = sdk.run_inline_query(
            result_format="json",
            body=query_payload
        )
        print("--- Data Query Successful ---")

        # 3. Build the Expanded URL parameters
        url_params = {}
        url_params['fields'] = ",".join(fields)
        url_params['sorts'] = ",".join(sorts)
        url_params['limit'] = str(limit)
        url_params['vis_config'] = vis_config_string
        url_params['toggle'] = 'vis' # Use the 'vis' tab
        
        filter_params = {}
        for key, value in filters.items():
            filter_params[f"f[{key}]"] = value
        
        url_params.update(filter_params)
        query_string = urlencode(url_params)
        
        # 4. Build the final Embed URL (FIXED: Read from st.secrets)
        base_url = st.secrets["looker"]["base_url"].replace(":19999", "") # Get browser URL
        viz_url = f"{base_url}/embed/explore/{MODEL_NAME}/{EXPLORE_NAME}?{query_string}"
        print(f"--- Expanded Embed URL Created: {viz_url} ---")
        
        # 5. Return both the data and the new URL
        final_output = {
            "data": data_result, # Send the parsed JSON, not a string of JSON
            "viz_url": viz_url
        }
        return json.dumps(final_output)

    except Exception as e:
        print(f"--- Query Failed --- \n{e}")
        return json.dumps({"error": f"Error running Looker query: {e}"})
    finally:
        if sdk:
            sdk.auth.logout()

# ==============================================================================
# Your Original Tool Definition (Imports fixed)
# ==============================================================================

looker_data_tool = StructuredTool.from_function(
    func=_run_looker_query,
    name="LookerDataQuery",
    description=(
        "Use this tool as the **primary source** for ANY questions about US population or census demographics. "
        "This includes nationwide totals (e.g., 'What is the total population of the US?') as well as specific breakdowns "
        "by geography (e.g., 'What is the population in California?', 'List median income by county in Texas').\n\n"
            
        "**CRITICAL QUERY-BUILDING INSTRUCTIONS:**\n"
        "1.  To get a total, aggregated value (like 'total population'), you **MUST** use the 'measure' field (e.g., `blockgroup.total_pop`).\n"
        "2.  For a general nationwide 'total population', you MUST use the measure `blockgroup.total_pop` and apply no filters.\n"
        "3.  To group data by a category (like 'by state'), you must add that 'dimension' (e.g., `state.state_name`) to the `fields` list along with the measure.\n"
        "4.  You MUST also generate a `vis_config_string`. This is a JSON object *as a string*.\n"
        "   - For single numbers (like total population), use: `'{\"type\": \"single_value\"}'`\n"
        "   - For tables (data grouped by state), use: `'{\"type\": \"table\"}'`\n"
        "   - For maps (data by state), use: `'{\"type\": \"looker_map\", \"map_field_name\": \"state.state_name\"}'`\n"
        "   - For bar charts (pop by state), use: `'{\"type\": \"looker_bar\", \"stacking\": \"normal\"}'`\n"
        "   - If unsure, default to: `'{\"type\": \"table\"}'`\n\n"
        
        f"Here is the complete schema of available fields: {EXPLORE_METADATA}"
    ),
    args_schema=LookerQueryInput
)

