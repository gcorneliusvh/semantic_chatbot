import os
import uuid
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Any, Dict

# Import the main agentic chain from your core logic file
from core_logic_langgraph import agentic_chain
from langchain_core.messages import HumanMessage, AIMessage

# ==============================================================================
# 1. FastAPI App Initialization
# ==============================================================================
app = FastAPI(
    title="Semantic Chatbot Connector API",
    description="Provides API access to a multi-agent, LangGraph-powered system for integration with platforms like Google Agentspace.",
    version="1.1.0"
)

# ==============================================================================
# 2. Pydantic Models for API Input/Output
# ==============================================================================

class Message(BaseModel):
    """Represents a single message in the chat history."""
    role: str = Field(..., description="The role of the message sender, either 'user' or 'assistant'.")
    content: str = Field(..., description="The text content of the message.")

class ChatInput(BaseModel):
    """The schema for the main /invoke_agent endpoint request body."""
    input: str = Field(..., description="The user's current query or prompt.")
    # Use a unique session_id to maintain memory across calls
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="A unique identifier for the conversation thread.")
    chat_history: List[Message] = Field(default=[], description="A list of previous messages in the conversation.")

class ChatOutput(BaseModel):
    """The schema for the main /invoke_agent endpoint response body."""
    output: Any = Field(..., description="The final answer or output from the agent.")
    session_id: str = Field(..., description="The session ID for the conversation, to be passed in subsequent requests.")

# ==============================================================================
# 3. API Endpoints
# ==============================================================================

@app.post("/invoke_agent", response_model=ChatOutput, tags=["Agent"])
async def invoke_agent_endpoint(payload: ChatInput):
    """
    Main endpoint to interact with the multi-agent LangGraph system.

    This endpoint takes a user's query and the conversation history,
    invokes the agentic chain, and returns the final, synthesized response.
    """
    try:
        # 1. Convert the chat history from Pydantic models to LangChain message objects
        history_messages = []
        for msg in payload.chat_history:
            if msg.role == "user":
                history_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                history_messages.append(AIMessage(content=msg.content))

        # 2. Set up the input for the LangGraph chain
        # The `configurable` dictionary is how LangGraph manages memory for different sessions
        config = {"configurable": {"thread_id": payload.session_id}}
        
        chain_input = {
            "original_question": payload.input,
            "chat_history": history_messages
        }

        # 3. Invoke the main agentic chain from your core logic
        # We'll stream the output to get the final state
        final_state = None
        async for event in agentic_chain.astream_events(chain_input, config, version="v1"):
             if event["event"] == "on_chain_end":
                 final_state = event["data"]["output"]


        if not final_state or not final_state.get("final_answer"):
             raise HTTPException(status_code=500, detail="Agent did not produce a final answer.")

        return {
            "output": final_state["final_answer"],
            "session_id": payload.session_id
        }

    except Exception as e:
        # Provide a more detailed error message for debugging
        raise HTTPException(status_code=500, detail=f"An error occurred while invoking the agent: {str(e)}")


@app.get("/", tags=["Health Check"])
async def root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Semantic Chatbot Connector is running."}