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
base_url = "[https://yourinstance.cloud.looker.com:19999](https://yourinstance.cloud.looker.com:19999)" # Note: Ensure you include the :19999 for API calls
client_id = "YOUR_LOOKER_CLIENT_ID"
client_secret = "YOUR_LOOKER_CLIENT_SECRET"

# Proxycurl API Key (used by social_tool.py)
# NOTE: This tool is used by the Social Agent and can be removed if not needed.
PROXYCURL_API_KEY = "YOUR_PROXYCURL_API_KEY"

Step 2: Install Dependencies
The project uses several Python libraries for the core functionality, as specified in requirements.txt.

Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows
Install packages (using the provided requirements.txt):

Bash

pip install -r requirements.txt
Step 3: Generate Looker Metadata (Mandatory)
The tools/looker_tool.py and tools/knowledge_tool.py agents rely on a local file named acs_census_metadata.json (or your chosen name) to understand the available dimensions and measures.

The script 01_fetch_metadata.py is used to generate this file using the Looker SDK.

Ensure Looker SDK is configured (it uses the variables in secrets.toml or a local looker.ini file).

Run the script (This assumes you are using the same Model/Explore names as the original project: data_block_acs_bigquery::acs_census_data):

Bash

python 01_fetch_metadata.py
This step generates the acs_census_metadata.json file, which is then loaded by the Looker tool.

Step 4: Run the Application
Start the Streamlit application from the root directory:

Bash

streamlit run app.py
The application will launch in your default web browser, allowing you to interact with the multi-agent chatbot.

🛠️ Customization Guide: Using a Different Looker Explore
To adapt this chatbot to a different Looker Explore (e.g., your internal financial model instead of the US Census Data), you need to modify three core files.

1. Update Looker Model and Explore Names
Edit tools/looker_tool.py to target your new data source.

File: tools/looker_tool.py	Change From	Change To (Example)
MODEL_NAME	"data_block_acs_bigquery"	"your_custom_model"
EXPLORE_NAME	"acs_census_data"	"your_custom_explore"

Export to Sheets

2. Generate New Metadata JSON
You must generate a new metadata file containing the fields from your custom Explore.

Update 01_fetch_metadata.py:

Change MODEL_NAME and EXPLORE_NAME in this file to match the values from the step above.

Change output_filename to something descriptive, e.g., "your_explore_metadata.json".

Run the updated script:

Bash

python 01_fetch_metadata.py
This creates your new metadata file (e.g., your_explore_metadata.json).

3. Update Metadata Loading References
Update the Looker tool and the Knowledge tool to load your new metadata file.

In tools/looker_tool.py: Modify the _get_explore_metadata function to reference your new JSON file:

Python

# tools/looker_tool.py
def _get_explore_metadata():
    """Loads the explore metadata JSON file."""
    try:
        # CHANGE FILENAME HERE
        with open("your_explore_metadata.json") as f:
            return json.dumps(json.load(f))
        ...
In tools/knowledge_tool.py: Modify the get_census_data_definition function to load the correct file:

Python

# tools/knowledge_tool.py
@tool(args_schema=TermInput)
def get_census_data_definition(term: str) -> str:
    # CHANGE FILENAME HERE
    with open('your_explore_metadata.json', 'r') as f:
        metadata = json.load(f)
    ...
4. Adjust Agent Strategy Prompts (Highly Recommended)
For optimal AI performance, update the hardcoded prompts in app.py to guide the agent toward using relevant fields in your new domain.

app.py: Find the LOOKER_AGENT_PROMPT_TEMPLATE and specifically update the ANALYST STRATEGY section. Guide the agent to think like an analyst in your new domain (e.g., recommend fields like orders.count, products.inventory_level, or financials.revenue instead of census demographics).
