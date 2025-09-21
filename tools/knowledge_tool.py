import json
from langchain_core.tools import tool

# --- Pydantic Import ---
from pydantic import BaseModel, Field

# ==============================================================================
# Pydantic Input Schemas
# ==============================================================================

class TermInput(BaseModel):
    term: str = Field(description="The census term to define, e.g., 'poverty line' or 'median income'.")

# ==============================================================================
# Tool Definitions
# ==============================================================================

@tool(args_schema=TermInput)
def get_census_data_definition(term: str) -> str:
    """
    Looks up the definition for a specific US Census term from the metadata file.
    Use this to answer questions like 'What is the poverty line?' or 'Define median income'.
    
    Args:
        term: The census term to define, e.g., 'poverty line' or 'median income'.
    """
    try:
        with open('acs_census_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Simple case-insensitive search
        search_term = term.lower().strip()
        
        for key, value in metadata.items():
            if search_term in key.lower() or (value and search_term in value.lower()):
                return f"Definition for '{key}': {value}"
                
        return f"Sorry, I could not find a definition for the term '{term}'."
        
    except FileNotFoundError:
        return "Error: The census metadata file (acs_census_metadata.json) was not found."
    except Exception as e:
        return f"Error reading census metadata: {e}"

