# In 02_test_looker_agent.py

import os
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_google_vertexai import ChatVertexAI
from langchain import hub

# Import our new tool
from tools.looker_tool import looker_data_tool

# Set up the LLM
llm = ChatVertexAI(model="gemini-2.5-pro", temperature=0)

# Set up the Agent
# We only give it one tool: the one we just made.
tools = [looker_data_tool]

# This is a pre-built prompt that tells the LLM how to use tools
agent_prompt = hub.pull("hwchase17/structured-chat-agent")

# Create the agent
agent = create_structured_chat_agent(llm, tools, agent_prompt)

# The AgentExecutor is what actually runs the agent and its tools
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # This is critical! It shows the agent's "thoughts"
    handle_parsing_errors=True # Handles errors gracefully
)

print("--- Looker Analyst Agent Initialized ---")
print("Ask questions about US census data. Type 'exit' to quit.")

# Start a simple chat loop
while True:
    try:
        query = input("\n[USER]: ")
        if query.lower() == 'exit':
            break
        
        # Call the agent with the user's query
        response = agent_executor.invoke({"input": query})
        
        print(f"\n[AGENT]: {response['output']}")
        
    except Exception as e:
        print(f"An error occurred: {e}")