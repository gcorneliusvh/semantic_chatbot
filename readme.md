# Looker + Gemini: The AI-Powered Semantic Layer
This project is a workshop demonstrating how to build a multi-agent generative AI chatbot on top of Looker's semantic layer.

## Architecture
- **Frontend:** Streamlit
- **Backend:** Google Cloud Run
- **AI Orchestration:** LangChain with Gemini Pro
- **Semantic Layer:** Looker (via Python SDK)
- **Metadata/Cache:** Google BigQuery
- **Target Model:** ACS Census Data

## Project Goals
1.  Provide a natural language interface to a Looker Explore.
2.  Use a multi-agent architecture for specialized tasks (data, social, knowledge).
3.  Generate not just data, but also Looker-powered visualizations.