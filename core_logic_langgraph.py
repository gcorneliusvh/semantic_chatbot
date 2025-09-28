import os
import json
from functools import partial
from typing import TypedDict, Annotated, List, Sequence

# --- Langchain & LangGraph Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain import hub

# --- Tool Imports ---
from tools.looker_tool import run_looker_query, LookerQueryInput
from tools.cache_tool import list_cached_datasets, load_dataframes_from_cache

# ==============================================================================
# 1. LLM & Tool Initialization
# ==============================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

ALL_TOOLS = []
TOOL_DESCRIPTIONS = ""

def load_all_tools():
    """Loads all specialist tools for the supervisor agent."""
    global ALL_TOOLS, TOOL_DESCRIPTIONS
    
    all_tools_list = []
    tool_desc_list = []
    tool_dir = "generated_tools"

    if os.path.exists(tool_dir):
        for filename in os.listdir(tool_dir):
            if filename.endswith(".json"):
                with open(os.path.join(tool_dir, filename), 'r') as f:
                    config = json.load(f)
                    tool_name = config["tool_name"]
                    
                    looker_tool = StructuredTool.from_function(
                        func=partial(run_looker_query, model_name=config["model_name"], explore_name=config["explore_name"]),
                        name=tool_name,
                        description=config["description_for_router"],
                        args_schema=LookerQueryInput
                    )
                    all_tools_list.append(looker_tool)
                    tool_desc_list.append(f"- {tool_name}: {config['description_for_router']}")

    python_tools = [list_cached_datasets, load_dataframes_from_cache, PythonREPLTool()]
    all_tools_list.extend(python_tools)
    tool_desc_list.append("- PythonREPLTool: A Python shell for complex data analysis, calculations, and transformations on cached data. Use `list_cached_datasets` and `load_dataframes_from_cache` to access data.")
    
    ALL_TOOLS = all_tools_list
    TOOL_DESCRIPTIONS = "\n".join(tool_desc_list)

load_all_tools()

# ==============================================================================
# 2. Graph State & Supervisor Agent
# ==============================================================================
class AgentState(TypedDict):
    original_question: str
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    plan: str
    final_answer: str
    

SUPERVISOR_PROMPT_TEMPLATE = """You are an expert data analyst and the supervisor of a team of tools. Your job is to answer a user's question by creating a plan and then executing that plan by calling your available tools in sequence.

**Available Tools:**
{tool_descriptions}

You have access to the following tools:
{tools}

When responding, you MUST use the following format:

Thought: Do I need to use a tool? Yes
Action: The action to take. Should be one of [{tool_names}]
Action Input: the input to the action, as a valid JSON object
Observation: The result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# ==============================================================================
# 3. Assemble the Graph
# ==============================================================================
def create_agentic_graph():
    """Creates the LangGraph agentic chain."""
    
    prompt = PromptTemplate.from_template(SUPERVISOR_PROMPT_TEMPLATE).partial(
        tool_descriptions=TOOL_DESCRIPTIONS,
        tools=hub.pull("hwchase17/react-json").prompt.partial(tools=ALL_TOOLS)
    )
    
    agent_runnable = create_react_agent(llm, ALL_TOOLS, prompt)

    def run_agent(state):
        """Runs the agent and formats the input correctly."""
        print("--- SUPERVISOR AGENT ---")
        agent_outcome = agent_runnable.invoke(state)
        messages = []

        if "output" in agent_outcome.return_values:
            final_answer = agent_outcome.return_values["output"]
            messages.append(AIMessage(content=final_answer, name="Supervisor"))
            return {"final_answer": final_answer, "messages": messages}
        else:
            tool_calls = []
            tool_call_id_counter = 0
            for action in agent_outcome:
                tool_call_id_counter += 1
                tool_calls.append({"name": action.tool, "args": action.tool_input, "id": str(tool_call_id_counter)})
            messages.append(AIMessage(content="", tool_calls=tool_calls))
            return {"messages": messages}

    def execute_tools(state):
        """Executes the tools called by the agent."""
        print("--- TOOL EXECUTOR ---")
        last_message = state["messages"][-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return
            
        tool_outputs = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_to_call = {t.name: t for t in ALL_TOOLS}[tool_name]
            output = tool_to_call.invoke(tool_call["args"])
            tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
            
        return {"messages": tool_outputs}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"action": "action", END: END}
    )
    workflow.add_edge("action", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

agentic_chain = create_agentic_graph()
