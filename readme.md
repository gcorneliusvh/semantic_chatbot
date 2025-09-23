# Looker + Gemini: The AI-Powered Semantic Layer Chatbot

This project is a multi-agent Streamlit application demonstrating how to build an AI chatbot on top of Looker's semantic layer to answer complex data and analytical questions.

The system uses a **Steering Agent** to route user queries to specialized agents, including the **Looker Data Agent** for census questions and the **Python Agent** for data analysis on cached results.

***

## 🚀 Deployment Guide

Follow these steps to deploy and run the application locally or on a cloud platform like Google Cloud Run.

### Step 1: Set Up Credentials

The application uses Streamlit Secrets to manage API keys. Create a folder named `.streamlit` in the project root and add a file named `secrets.toml` with the following structure:

```toml
# .streamlit/secrets.toml

# Google Gemini API Key
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"

# Looker SDK Credentials (used by looker_tool.py)
[looker]
# FIX: Removed markdown link formatting for compatibility
base_url = "[https://yourinstance.cloud.looker.com](https://yourinstance.cloud.looker.com)" # e.g. "[https://mycompany.cloud.looker.com:19999](https://mycompany.cloud.looker.com:19999)"
client_id = "YOUR_LOOKER_CLIENT_ID"
client_secret = "YOUR_LOOKER_CLIENT_SECRET"

# Proxycurl API Key (used by social_tool.py)
# NOTE: This tool is used by the Social Agent and can be removed if not needed.
PROXYCURL_API_KEY = "YOUR_PROXYCURL_API_KEY"
