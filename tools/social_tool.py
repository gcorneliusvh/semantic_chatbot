import streamlit as st
import os
import requests
from langchain_core.tools import tool

# --- Pydantic Import ---
from pydantic import BaseModel, Field

# ==============================================================================
# Pydantic Input Schemas
# ==============================================================================

class ProfileInput(BaseModel):
    profile_url: str = Field(description="The full URL of the LinkedIn profile to look up.")

# ==============================================================================
# Tool Definitions
# ==============================================================================

@tool(args_schema=ProfileInput)
def get_profile_data(profile_url: str) -> str:
    """
    Fetches data for a specific LinkedIn profile URL using the Proxycurl API.
    
    Args:
        profile_url: The full URL of the LinkedIn profile to look up.
    """
    try:
        api_key = st.secrets["PROXYCURL_API_KEY"]
        if not api_key:
            return "Error: PROXYCURL_API_KEY is not set in Streamlit secrets."
            
    except KeyError:
        return "Error: PROXYCURL_API_KEY not found in Streamlit secrets."
        
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    headers = {'Authorization': f'Bearer {api_key}'}
    params = {
        'url': profile_url,
    }

    try:
        response = requests.get(api_endpoint, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            return "Error: Invalid Proxycurl API key."
        elif response.status_code == 404:
            return f"Error: LinkedIn profile not found at {profile_url}"
        else:
            return f"Error: API request failed with status code {response.status_code}. Response: {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"Error making API request: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

