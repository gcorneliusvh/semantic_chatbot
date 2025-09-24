# main.py
import os
import json
import pandas as pd
import io
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Tool Imports ---
# (Assuming your 'tools' directory and metadata file are in the same directory)
from tools.looker_tool import _run_looker_query, LookerQueryInput
from tools.knowledge_tool import get_census_data_definition, TermInput
from langchain_experimental.tools.python.tool import PythonREPLTool

# ==============================================================================
# 1. FastAPI App Initialization
# ==============================================================================
app = FastAPI(
    title="Looker Semantic Layer API",
    description="An API that exposes Looker and Python data analysis tools for use by a master AI agent.",
    version="1.0.0",
)

# ==============================================================================
# 2. In-Memory Cache (or Temporary File System)
# ==============================================================================
# For a server environment, we'll cache the last query's data in memory.
# For Cloud Run, this cache is temporary and scoped to the instance.
# A more robust solution could use Google Cloud Storage or a Redis instance.
TEMP_CACHE_PATH = "/tmp/data_cache.csv"

def save_df_to_temp_cache(df: pd.DataFrame):
    """Saves a DataFrame to a temporary location."""
    try:
        df.to_csv(TEMP_CACHE_PATH, index=False)
        return f"Data saved to {TEMP_CACHE_PATH}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving to cache: {e}")

def load_df_from_temp_cache() -> Optional[pd.DataFrame]:
    """Loads a DataFrame from the temporary cache if it exists."""
    if not os.path.exists(TEMP_CACHE_PATH):
        return None
    try:
        return pd.read_csv(TEMP_CACHE_PATH)
    except Exception as e:
        print(f"Error loading from cache: {e}")
        return None

# ==============================================================================
# 3. API Endpoints
# ==============================================================================

@app.post("/looker-query/", tags=["Looker Tool"])
async def run_looker_query_endpoint(query_input: LookerQueryInput):
    """
    Exposes the Looker Data Agent's primary capability.

    Receives query parameters, executes a query via the Looker SDK,
    caches the result, and returns a summary and visualization URL.
    """
    try:
        # The original _run_looker_query function saves the data and returns a JSON string
        result_str = _run_looker_query(
            fields=query_input.fields,
            filters=query_input.filters,
            sorts=query_input.sorts,
            limit=query_input.limit,
            vis_config_string=query_input.vis_config_string
        )

        result_json = json.loads(result_str)
        if "error" in result_json:
            raise HTTPException(status_code=500, detail=result_json["error"])

        # The original function already saves the data to a cache file via a helper,
        # which we will adapt to use the temp cache for the Python endpoint.
        # Here, we'll also return the data directly for immediate use.
        df = load_df_from_temp_cache()
        if df is not None:
             result_json["data_json"] = df.to_json(orient='records')
        else:
            result_json["data_json"] = []


        return result_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


class PythonAnalysisInput(BaseModel):
    question: str = Field(description="A natural language question about the data cached from the last Looker query.")

@app.post("/python-analysis/", tags=["Python Tool"])
async def analyze_cached_data_endpoint(analysis_input: PythonAnalysisInput):
    """
    Exposes the Python Data Agent's capability.

    It loads the cached data and uses a Python REPL tool to answer
    an analytical question about it.
    """
    df = load_df_from_temp_cache()
    if df is None:
        raise HTTPException(status_code=404, detail="No data available to analyze. Please run a /looker-query first.")

    try:
        # The PythonREPLTool needs to be initialized with the DataFrame in its local scope
        python_tool = PythonREPLTool(locals={"pd": pd, "df": df})

        # Construct a prompt for the tool
        # In a real scenario, this part could also be an LLM call, but for a tool,
        # we can often be more direct. Here we directly execute code based on the question.
        # For simplicity, we'll keep the direct execution part from your original agent.
        # This is a simplified example; a full implementation would use an LLM to generate the code.
        code_to_run = f"""
try:
    print(df.eval('{analysis_input.question}'))
except Exception:
    # A more robust solution would use an LLM to translate the question to Python code
    print("Could not directly evaluate the question. Here are the dataframe stats:")
    print(df.describe())
"""
        result = python_tool.run(code_to_run)

        return {"question": analysis_input.question, "answer": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during Python analysis: {str(e)}")


@app.post("/define-term/", tags=["Knowledge Tool"])
async def define_term_endpoint(term_input: TermInput):
    """
    Exposes the knowledge base capability to define census terms.
    """
    try:
        definition = get_census_data_definition.func(term=term_input.term)
        return {"term": term_input.term, "definition": definition}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "Looker Semantic Bot Connector is running."}