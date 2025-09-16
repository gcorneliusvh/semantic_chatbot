# In tools/looker_tool.py

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field  # Using Pydantic v2
import looker_sdk
import json
import os

# --- Configuration ---
# Point to our credentials file
os.environ["LOOKERSDK_CONFIG_FILE"] = "looker.ini"
SDK = looker_sdk.init40()

# Load the metadata we fetched in Phase 1
try:
    with open("acs_census_metadata.json") as f:
        EXPLORE_METADATA = json.dumps(json.load(f))
except FileNotFoundError:
    print("Error: acs_census_metadata.json not found.")
    print("Please run 01_fetch_metadata.py first.")
    EXPLORE_METADATA = ""

# The "ground truth" model/explore names we found
MODEL_NAME = "data_block_acs_bigquery"
EXPLORE_NAME = "acs_census_data"
# --- End Configuration ---


class LookerQueryInput(BaseModel):
    """Input schema for running a Looker query."""
    fields: list[str] = Field(description="List of dimensions and measures, e.g., ['state.state_name', 'blockgroup.total_pop']")
    filters: dict[str, str] = Field(description="Dictionary of filters, e.g., {'state.state_name': 'California'}", default={})
    sorts: list[str] = Field(description="List of fields to sort by, e.g., ['blockgroup.total_pop desc']", default=[])
    limit: int = Field(description="Row limit for the query", default=10)

# Add matching defaults to the function signature
def _run_looker_query(fields: list[str], filters: dict = {}, sorts: list[str] = [], limit: int = 10) -> str:
    """The actual function that runs the query."""
    if not fields:
        return "Error: You must provide at least one field."
        
    try:
        query_payload = {
            "model": MODEL_NAME,
            "view": EXPLORE_NAME,
            "fields": fields,
            "filters": filters,
            "sorts": sorts,
            "limit": str(limit)
        }
        
        print(f"--- Running Looker Query --- \n{json.dumps(query_payload, indent=2)}")
        result = SDK.run_inline_query(
            result_format="json",
            body=query_payload
        )
        print("--- Query Successful ---")
        return json.dumps(result)

    except Exception as e:
        print(f"--- Query Failed --- \n{e}")
        return f"Error running Looker query: {e}"

# This is the tool we will give to our main agent
looker_data_tool = StructuredTool.from_function(
    func=_run_looker_query,
    name="LookerDataQuery",
    description=(
        "Use this tool as the **primary source** for ANY questions about US population or census demographics. "
        "This includes nationwide totals (e.g., 'What is the total population of the US?') as well as specific breakdowns "
        "by geography (e.g., 'What is the population in California?', 'List median income by county in Texas').\n\n"
            
        "**CRITICAL QUERY-BUILDING INSTRUCTIONS:**\n"
        "1.  To get a total, aggregated value (like 'total population' or 'total asian population'), "
            "you **MUST** use the 'measure' field (e.g., `blockgroup.total_pop`, `blockgroup.asian_pop`).\n"
        "2.  For a general nationwide 'total population' question, you MUST use the measure `blockgroup.total_pop` and apply no filters.\n"
        "3.  Do NOT use fields ending in `_dim` for aggregation. Use the 'measure' version.\n"
        "4.  To group data by a category (like 'by state'), you must add that 'dimension' "
            "to the `fields` list (e.g., `fields=['state.state_name', 'blockgroup.total_pop']`).\n"
        "5.  To filter data, use a 'dimension' in the `filters` dictionary (e.g., `filters={{'state.state_name': 'California'}}`).\n\n"
            
        f"Here is the complete schema of available fields: {EXPLORE_METADATA}"
    ),
    args_schema=LookerQueryInput
)