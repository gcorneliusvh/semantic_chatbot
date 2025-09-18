# In tools/looker_tool.py

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import looker_sdk
import json
import os
from urllib.parse import urlencode

# --- Configuration (Unchanged) ---
os.environ["LOOKERSDK_CONFIG_FILE"] = "looker.ini"
SDK = looker_sdk.init40()

try:
    with open("acs_census_metadata.json") as f:
        EXPLORE_METADATA = json.dumps(json.load(f))
except FileNotFoundError:
    print("Error: acs_census_metadata.json not found.")
    EXPLORE_METADATA = ""

MODEL_NAME = "data_block_acs_bigquery"
EXPLORE_NAME = "acs_census_data"
# --- End Configuration ---


# --- Pydantic Schema (Unchanged) ---
class LookerQueryInput(BaseModel):
    """Input schema for running a Looker query."""
    fields: list[str] = Field(description="List of dimensions and measures, e.g., ['state.state_name', 'blockgroup.total_pop']")
    filters: dict[str, str] = Field(description="Dictionary of filters, e.g., {'state.state_name': 'California'}", default={})
    sorts: list[str] = Field(description="List of fields to sort by, e.g., ['blockgroup.total_pop desc']", default=[])
    limit: int = Field(description="Row limit for the query", default=10)
    vis_config_string: str = Field(
        description="A valid Looker vis_config JSON object, as a string. Use a simple 'table' viz if unsure.",
        default='{"type": "table"}'
    )
# --- End Schema ---


# --- UPDATED FUNCTION (with toggle=vis) ---
def _run_looker_query(
    fields: list[str], 
    filters: dict = {}, 
    sorts: list[str] = [], 
    limit: int = 10,
    vis_config_string: str = '{"type": "table"}'
) -> str:
    """Runs a query to get data AND builds a full Expanded URL for embedding."""
    if not fields:
        return json.dumps({"error": "You must provide at least one field."})

    try:
        # 1. Create the query payload
        query_payload = {
            "model": MODEL_NAME,
            "view": EXPLORE_NAME,
            "fields": fields,
            "filters": filters,
            "sorts": sorts,
            "limit": str(limit),
            "vis_config": vis_config_string
        }
        
        # 2. Run the query inline to get the data
        print(f"--- Running Looker Inline Query --- \n{json.dumps(query_payload, indent=2)}")
        data_result = SDK.run_inline_query(
            result_format="json",
            body=query_payload
        )
        data_json = json.dumps(data_result)
        print("--- Data Query Successful ---")

        # 3. Build the Expanded URL parameters
        url_params = {}
        url_params['fields'] = ",".join(fields)
        url_params['sorts'] = ",".join(sorts)
        url_params['limit'] = str(limit)
        url_params['vis_config'] = vis_config_string
        url_params['toggle'] = 'vis'  # <-- THIS IS THE FIX
        
        # Convert filter dict to f[key]=val format
        filter_params = {}
        for key, value in filters.items():
            filter_params[f"f[{key}]"] = value
        
        url_params.update(filter_params)
        
        # URL-encode all parameters
        query_string = urlencode(url_params)
        
        # 4. Build the final Embed URL
        base_url = SDK.auth.settings.base_url.rstrip('/') 
        viz_url = f"{base_url}/embed/explore/{MODEL_NAME}/{EXPLORE_NAME}?{query_string}"
        print(f"--- Expanded Embed URL Created: {viz_url} ---")
        
        # 5. Return both the data and the new URL
        final_output = {
            "data": data_json,
            "viz_url": viz_url
        }
        return json.dumps(final_output)

    except Exception as e:
        print(f"--- Query Failed --- \n{e}")
        return json.dumps({"error": f"Error running Looker query: {e}"})
# --- END UPDATED FUNCTION ---


# --- TOOL DEFINITION (Unchanged) ---
looker_data_tool = StructuredTool.from_function(
    func=_run_looker_query,
    name="LookerDataQuery",
    description=(
        "Use this tool as the **primary source** for ANY questions about US population or census demographics. "
        "This includes nationwide totals (e.g., 'What is the total population of the US?') as well as specific breakdowns "
        "by geography (e.g., 'What is the population in California?', 'List median income by county in Texas').\n\n"
            
        "**CRITICAL QUERY-BUILDING INSTRUCTIONS:**\n"
        "1.  To get a total, aggregated value (like 'total population'), you **MUST** use the 'measure' field (e.g., `blockgroup.total_pop`, `blockgroup.asian_pop`).\n"
        "2.  For a general nationwide 'total population', you MUST use the measure `blockgroup.total_pop` and apply no filters.\n"
        "3.  To group data by a category (like 'by state'), you must add that 'dimension' "
            "to the `fields` list.\n"
        "4.  **LEAST-SHOT VIZ PROMPT:** You MUST also generate a `vis_config_string`. This is a JSON object *as a string*. "
        "   - For single numbers (like total population), use: `'{\"type\": \"single_value\"}'`\n"
        "   - For tables (data grouped by state), use: `'{\"type\": \"table\"}'`\n"
        "   - For maps (data by state), use: `'{\"type\": \"looker_map\", \"map_field_name\": \"state.state_name\"}'`\n"
        "   - For bar charts (pop by state), use: `'{\"type\": \"looker_bar\", \"stacking\": \"normal\"}'`\n"
        "   - If unsure, default to: `'{\"type\": \"table\"}'`\n\n"
        
        f"Here is the complete schema of available fields: {EXPLORE_METADATA}"
    ),
    args_schema=LookerQueryInput
)