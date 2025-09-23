# Chatbot Architecture Manual

This document provides a comprehensive overview of the Looker-Gemini chatbot's architecture, including its multi-agent design, data flow, and technology stack.

## 1. Core Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend/Deployment** | Streamlit (Python) | User Interface and application hosting. |
| **AI Orchestration** | LangChain | Framework for connecting LLMs, tools, and agents. |
| **Foundation Models** | Gemini 2.5 Pro, Gemini 2.5 Flash | The reasoning and generation engines for agents and routing. |
| **Semantic Layer** | Looker SDK (API3) | Governed access to business data (e.g., ACS Census Explore). |
| **Web Search** | DuckDuckGoSearchRun | Tool for providing real-time general knowledge (fallback). |
| **Local Cache** | Pandas DataFrames (`data.csv`) | Temporary storage for query results for analysis by the Python Agent. |

## 2. Multi-Agent Design and Routing

The system employs a **Supervisor/Router** architecture, using a central LLM to classify user input and route it to one of four specialized agents via a `RunnableBranch`.

### The Router (Steering Agent)

* **Model:** Gemini 2.5 Flash (optimized for speed).
* **Mechanism:** Uses a Pydantic schema (`RouteQuery`) to classify the user's intent into one of four possible destinations: `looker`, `python_agent`, `social`, or `general`.

### Specialized Agents

| Agent Name | Primary Function | Core Tools | Dedicated LLM | Data Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| **Looker Data Agent** | Governed data retrieval and visualization from the Looker Explore. | `LookerDataQuery`, `get_census_data_definition` | Gemini 2.5 Flash | Looker Instance, `acs_census_metadata.json` |
| **Python Agent** | Performs complex data analysis, filtering, and calculation on retrieved data. | `PythonREPLTool` (with `pd` in scope) | Gemini 2.5 Pro | Reads from local cache (`data.csv`). |
| **General Agent** | Handles conversational small talk, knowledge definitions, and real-time web search. | `DuckDuckGoSearchRun`, `get_census_data_definition` | Gemini 2.5 Pro | `acs_census_metadata.json` (for definitions). |
| **Social Agent** | Handles specific external API calls (e.g., LinkedIn data). | `get_profile_data` (Proxycurl API) | Gemini 2.5 Flash | External API Key. |

## 3. Data Flow and Caching Pipeline

The application incorporates a crucial caching step to decouple data retrieval (expensive, Looker API call) from data analysis (fast, local Python operation).

1. **User Query In:** User enters a query (e.g., "Show population by state").
2. **Routing:** The **Router** determines the query is for `looker`.
3. **Data Retrieval (Looker Agent):**
   * The Looker Data Agent calls the wrapper function for `LookerDataQuery`.
   * This function uses the Looker SDK to execute an inline query on the target Explore.
   * It generates a full **Expanded Embed URL** for the resulting visualization.
   * It converts the query result into a Pandas DataFrame.
4. **Data Caching:**
   * The DataFrame is immediately saved to the local file system as **`data.csv`** via the `save_data_to_cache` tool.
5. **Visualization & Follow-up Generation:**
   * The Agent's final response (Markdown summary) and the generated **Embed URL** are sent back to the Streamlit UI.
   * The UI displays the visualization using an iframe and then calls `get_followup_questions` (Gemini 2.5 Flash) to suggest analytical questions based on the cached data's stats.
6. **Analysis Query:** User asks a follow-up (e.g., "What is the average median income?").
7. **Routing:** The **Router** determines the query is for `python_agent`.
8. **Data Analysis (Python Agent):**
   * The Python Agent's first step is always to load the data: `df = pd.read_csv('data.csv')`.
   * It uses the `PythonREPLTool` to execute the code required to answer the user's analytical question.

This pipeline ensures that once a large dataset is pulled from Looker, all subsequent analytical queries are performed instantaneously on the local cache.
