# Semantic Chatbot Technical Manual: Deep Dive and Training Guide

This manual serves as a comprehensive technical guide to the multi-agent Looker-Gemini chatbot. It details the architecture, the specific implementation of LangChain components, the interaction with the Looker SDK, and the core frontend logic built with Streamlit.

## 1. The Agentic Core: LangChain Orchestration

The application utilizes a sophisticated, modular agent architecture powered by LangChain, orchestrated by a central Router.

### 1.1 Model Selection Strategy (`llm_pro` vs. `llm_flash`)

The system employs a performance-based model selection strategy to optimize cost and speed:

| Model Instance | Gemini Model | Task Specialization | Rationale |
| :--- | :--- | :--- | :--- |
| `llm_flash` | Gemini 2.5 Flash | **Router, Looker Agent, Social Agent, Follow-up Questions** | Used for tasks requiring high speed, conversational fluency, tool-calling (Looker/Social), and structured output. |
| `llm_pro` | Gemini 2.5 Pro | **Python Agent, General Agent (Web Search)** | Used for tasks requiring deep reasoning, complex code generation (Python), and synthesizing information from multiple external search results. |

### 1.2 The Supervisor: Routing and Structured Output

The system starts with a **Router Agent** to classify user intent. This is implemented using **Structured Output** for reliable, predictable routing.

1.  **Pydantic Schema (`RouteQuery`)**: Defined in `app.py`, this schema explicitly limits the LLM's response to one of four `Literal` string values (`"looker"`, `"python_agent"`, etc.).

2.  **Output Parser (`PydanticOutputParser`)**: This component forces the LLM to structure its output as a JSON object matching the `RouteQuery` schema. This highly reliable approach ensures the code always knows which agent to call next.

3.  **Routing Chain (`RunnableBranch`)**: The `router` output (the destination string) is fed into a `RunnableBranch`. This is the core logic that executes the corresponding agent chain based on the classification result.

### 1.3 Agent Execution and Transparency

All specialized agents use the **Structured Chat Agent** framework, which is built on the **ReAct (Reasoning and Acting)** pattern:

* **Thought:** The agent reasons about the input and the available tools.

* **Action:** It generates a JSON object specifying the `action` (tool name) and `action_input` (parameters).

* **Observation:** The tool executes the request and returns its output, which the agent uses for its next turn.

**Key Components:**

* **`AgentExecutor`**: Manages this iterative process. Critical configurations include `handle_parsing_errors=True` and `max_iterations`, preventing the agent from getting stuck in infinite loops.

* **`StreamlitCallbackHandler`**: This custom class intercepts the `on_agent_action`, `on_tool_end`, and `on_llm_end` events from the AgentExecutor. It collects the entire `Thought/Action/Observation` history, which is then displayed in the Streamlit UI's expander for full transparency.

## 2. Looker SDK, Semantic Layer, and Visualization

The Looker integration is implemented in `tools/looker_tool.py`, leveraging Looker's semantic layer to ensure data governance.

### 2.1 Looker SDK Elements and Governed Queries

The `LookerDataQuery` tool is the main gateway to the Looker instance:

1.  **Authentication (`_get_looker_sdk`)**: It uses `looker_sdk.init40()` to initialize the SDK, which reads API credentials (`client_id`, `client_secret`, `base_url`) directly from Streamlit's `secrets.toml`. This is a stable, recommended way to connect a programmatic client.

2.  **Query Construction**: The LLM's structured request (validated by the `LookerQueryInput` Pydantic schema) is mapped to the Looker SDK's query model:

    * `models40.WriteQuery`: This object specifies the target `model`, `view` (Explore), `fields`, `filters`, and `sorts`.

3.  **Data Retrieval**: `sdk.run_inline_query(result_format="json", body=query_payload)` is called. This executes the query defined by the payload, bypassing the need to create a persistent query object. The result is returned as a JSON string, which is immediately converted to a Pandas DataFrame for caching.

### 2.2 Metadata Grounding (The LLM's Semantic Context)

The `acs_census_metadata.json` file is fundamental to the **Looker Data Agent's** success:

* **Purpose**: It provides the LLM with a list of *every single available* Looker dimension and measure, along with its label and a detailed business description.

* **Injection**: This entire JSON structure is included directly in the `looker_data_tool`'s description string (via the `EXPLORE_METADATA` variable). This is known as **prompt grounding**. The agent uses this schema to map natural language requests (e.g., "how many households?") to the correct LookML field name (e.g., `blockgroup.households`).

### 2.3 Visualization and iFrame Rendering

The chatbot does not render charts; it generates the necessary inputs for Looker to render its own charts within the app.

1.  **`vis_config_string`**: This is a crucial input parameter to the Looker tool. The LLM is explicitly prompted to generate this JSON string (e.g., `{"type": "looker_column"}`) to define the chart type, which is passed to the `WriteQuery` model.

2.  **Embed URL Construction**: After executing the data query, the function reconstructs the original query parameters (`fields`, `filters`, `sorts`, and `vis_config`) and uses the Python `urllib.parse.urlencode` utility to securely package them into a single query string.

3.  **Display**: The finalized URL (`{BASE_URL}/embed/explore/...`) is passed back to `app.py`, where `st.components.v1.iframe(viz_url, height=600)` is used to embed the live, interactive Looker visualization, ensuring a seamless data experience.

## 3. Data Caching and Python Analysis Pipeline

The system's two-stage data flow dramatically improves the user experience for iterative analysis.

### 3.1 The Caching Layer (`tools/cache_tool.py`)

The `cache_tool.py` defines low-level functions for file I/O:

* **Decoupling**: The primary purpose of `save_data_to_cache` is to write the Looker result to a local `data.csv` file. This separates the network-intensive Looker API call from the CPU-intensive data analysis.

* **Persistence**: `load_df_from_cache` is used by the Streamlit sidebar (`setup_sidebar`) and, critically, by the **Python Agent** to load data for in-memory analysis without repeating the Looker query.

### 3.2 The Python Agent (`app.py`)

The Python Agent is a powerful analyst tool, specifically designed to manipulate the cached data:

* **Initialization**: The agent is initialized with the `PythonREPLTool`, which executes code in a sandbox.

* **Hardcoded Instruction**: The agent's prompt includes a **critical, non-negotiable instruction**: "Your primary task is to answer the user's question by analyzing a pandas DataFrame named `df`. Your **FIRST ACTION** must *always* be to load this file into a pandas DataFrame named `df`."

* **Process**: This forces the agent to use its tool to run the code: `import pandas as pd; df = pd.read_csv('data.csv')`. All subsequent analytical steps (e.g., calculating means, standard deviations, custom filtering) are performed on this local `df`.

## 4. Streamlit UI, State Management, and UX

Streamlit ties the complex backend logic to a fluid, conversational user interface.

### 4.1 State and History Management

* **`st.session_state.messages`**: The core state container. Unlike simple chat history, each assistant message is stored as a dictionary that also contains metadata:

    * `content`: The main text response (the Agent's `Final Answer`).

    * `viz_url`: The Looker Embed URL, if generated.

    * `agent_thoughts`: The complete trace of the agent's decision-making process.

    * `followup`: The list of suggested questions, if generated.

### 4.2 Follow-up Question Generation

This is a key UX feature that drives continuous exploration:

1.  **Input**: The `get_followup_questions` function is called *after* a successful Looker query. It receives the original user query, the chat history, and the raw **`data_stats`** (the JSON output of `df.describe()`).

2.  **LLM Chain**: A fast LLM (Gemini 2.5 Flash) is tasked with analyzing these statistical metrics (min, max, mean, count) and generating three insightful questions that are grounded in the actual data retrieved (e.g., "The mean population is X; what states are above that mean?").

3.  **Interaction**: These questions are rendered as Streamlit buttons. Clicking a button updates `st.session_state.clicked_prompt` and forces an `st.rerun()`, immediately submitting the follow-up as a new user query, often routing to the **Python Agent**.

### 4.3 Script Editor (The Advanced User View)

The `pages/1_Script_Editor.py` file provides a powerful utility for advanced users or debugging:

* It manually loads the persistent `data.csv` file into a DataFrame named `df`.

* It allows users to write and execute arbitrary Python code against this `df` using the built-in Python `exec()` function within a controlled sandbox environment.

* This demonstrates the core capability of the cache: providing raw, accessible data for external analysis.
