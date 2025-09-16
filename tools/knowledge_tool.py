from langchain_google_vertexai import ChatVertexAI
from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool
from langchain.tools import Tool

# 1. Create the base LLM for THIS TOOL ONLY
base_llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)

# 2. Create and bind the Google Search tool to THIS LLM
google_search_tool = VertexTool(google_search={})
llm_with_search = base_llm.bind_tools([google_search_tool])

# 3. This tool's function just calls the grounded LLM
def _run_general_knowledge(query: str) -> str:
    """Runs the query against the grounded Gemini model."""
    print(f"--- Calling Grounded Knowledge Tool for: {query} ---")
    try:
        # We simply invoke the LLM that has search built-in
        response = llm_with_search.invoke(query)
        return response.content
    except Exception as e:
        return f"Error in knowledge search: {e}"

# 4. Define the LangChain Tool for our main agent to use
# In tools/knowledge_tool.py
# (This is the new, simple description)
general_knowledge_tool = Tool.from_function(
    func=_run_general_knowledge,
    name="GeneralKnowledgeSearch",
    description=(
        "Use this for any general knowledge questions or facts about current events, companies, people, or "
        "any data NOT related to the US Census (e.g., 'What is the capital of France?' or 'population of Canada')."
    )
)