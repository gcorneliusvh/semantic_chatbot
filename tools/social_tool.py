# In tools/social_tool.py
from langchain.tools import Tool
from langchain_google_vertexai import ChatVertexAI

# UPDATED to gemini-2.5-pro
llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0.5)

def _social_chat(query: str) -> str:
    """Responds to social queries."""
    
    # We can give it a simple persona prompt
    prompt = f"""
    You are a friendly and helpful assistant. 
    A user just said this to you: "{query}"
    Respond in a brief, conversational way.
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Sorry, I had trouble with that: {e}"


social_tool = Tool.from_function(
    func=_social_chat,
    name="SocialChat",
    description="Use this tool for greetings, farewells, and casual, non-data, non-knowledge questions (e.g., 'how are you?', 'hello', 'thanks')."
)