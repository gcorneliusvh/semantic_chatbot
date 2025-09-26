import os
import json
from functools import partial
from typing import TypedDict, Annotated, List, Union

# --- Langchain & LangGraph Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# --- Tool Imports ---
from tools.looker_tool import run_looker_tool_from_model, LookerQueryInput
from tools.cache_tool import list_cached_datasets, load_dataframes_from_cache
from pydantic import BaseModel, Field

# ==============================================================================
# 1. LLM & Tool Initialization
# ==============================================================================
llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def load_dynamic_tools():
    """Loads all tool configurations and creates a list of LangChain tools."""
    all_tools = []
    tool_descriptions_for_planner = []

    if not os.path.exists("generated_tools"):
        return [], {}

    for filename in os.listdir("generated_tools"):
        if filename.endswith(".json"):
            filepath = os.path.join("generated_tools", filename)
            with open(filepath, 'r') as f:
                config = json.load(f)
                tool_name = config["tool_name"]

                # Use partial to pre-fill the model and explore names into our new wrapper
                specific_looker_func = partial(
                    run_looker_tool_from_model,
                    model_name=config["model_name"],
                    explore_name=config["explore_name"]
                )
                
                # The agent now only needs to provide the 'query_input' object
                dynamic_tool = StructuredTool.from_function(
                    func=specific_looker_func,
                    name=tool_name,
                    description=config["description_for_router"],
                    args_schema=LookerQueryInput 
                )
                all_tools.append(dynamic_tool)
                tool_descriptions_for_planner.append(f"- Tool: `{tool_name}`\n  Description: {config['description_for_router']}")
    
    python_tool_desc = "Executes Python code in a REPL environment to perform complex data analysis, calculations, or transformations on one or more datasets that have already been loaded from the cache. Use `list_cached_datasets` first to see available data, then `load_dataframes_from_cache` to load them."
    all_tools.extend([list_cached_datasets, load_dataframes_from_cache, PythonREPLTool()])
    tool_descriptions_for_planner.append(f"- Tool: `PythonREPL`\n  Description: {python_tool_desc}")
    
    return all_tools, "\n".join(tool_descriptions_for_planner)

ALL_TOOLS, TOOLS_DESCRIPTION = load_dynamic_tools()
tool_executor_node = ToolNode(ALL_TOOLS) # Define the executor node here

# ==============================================================================
# 2. Graph State & Nodes
# ==============================================================================
class GraphState(TypedDict):
    original_question: str
    plan: str
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    final_answer: str

# --- Planner Node ---
PLANNER_PROMPT = """You are an expert AI data analyst and planner... (Your full planner prompt remains the same)"""
planner_prompt = PromptTemplate.from_template(PLANNER_PROMPT)
planner_agent = planner_prompt | llm_pro

def planner_node(state: GraphState):
    print("--- 🧠 PLANNER ---")
    plan = planner_agent.invoke({
        "question": state["original_question"],
        "tool_description": TOOLS_DESCRIPTION
    })
    return {"messages": [HumanMessage(content=plan.content, name="Planner")]}


# --- Tool Agent & Node ---
tool_agent = llm_flash.bind_tools(ALL_TOOLS)

def tool_agent_node(state: GraphState):
    """Responsible for converting the plan into a specific tool call."""
    print("--- 🛠️ TOOL AGENT ---")
    # The plan is in the last message from the planner
    plan = state["messages"][-1].content
    
    tool_prompt = f"""You are a tool-calling expert. Based on the provided plan, select and call the single, most appropriate tool to execute the *next* step of the plan.

**Plan:**
{plan}
"""
    tool_call_message = tool_agent.invoke(tool_prompt)
    return {"messages": [tool_call_message]}


# --- Synthesizer Node ---
SYNTHESIZER_PROMPT = """You are an expert AI data analyst... (Your full synthesizer prompt remains the same)"""
synthesizer_prompt = PromptTemplate.from_template(SYNTHESIZER_PROMPT)
synthesizer_agent = synthesizer_prompt | llm_pro

def response_synthesizer_node(state: GraphState):
    print("--- ✍️ SYNTHESIZER ---")
    final_answer = synthesizer_agent.invoke({
        "question": state["original_question"],
        "tool_outputs": "\n".join([str(msg.content) for msg in state["messages"] if isinstance(msg, ToolMessage)])
    })
    return {"final_answer": final_answer.content}


# ==============================================================================
# 3. Assemble the Graph
# ==============================================================================
def create_agentic_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_agent", tool_agent_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("synthesizer", response_synthesizer_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "tool_agent")
    workflow.add_edge("tool_agent", "tool_executor")
    workflow.add_edge("tool_executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

agentic_chain = create_agentic_graph()