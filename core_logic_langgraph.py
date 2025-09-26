import os
import json
from functools import partial
from typing import TypedDict, Annotated, List

# --- Langchain & LangGraph Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# --- Tool Imports ---
from tools.looker_tool import run_looker_query, LookerQueryInput # <-- FIXED THIS LINE
from tools.cache_tool import list_cached_datasets, load_dataframes_from_cache

# ==============================================================================
# 1. LLM & Tool Initialization
# ==============================================================================
llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def load_dynamic_tools():
    """Loads all tool configurations from the 'generated_tools' directory."""
    all_tools = []
    tool_descriptions_for_planner = []
    tool_dir = "generated_tools"

    if not os.path.exists(tool_dir):
        print(f"Warning: '{tool_dir}' directory not found. No dynamic tools will be loaded.")
        return [], {}

    for filename in os.listdir(tool_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(tool_dir, filename)
            with open(filepath, 'r') as f:
                config = json.load(f)
                tool_name = config["tool_name"]

                # Use partial to pre-fill the model and explore names into the corrected function
                specific_looker_func = partial(
                    run_looker_query, # <-- FIXED THIS LINE
                    model_name=config["model_name"],
                    explore_name=config["explore_name"]
                )
                
                dynamic_tool = StructuredTool.from_function(
                    func=specific_looker_func,
                    name=tool_name,
                    description=config["description_for_router"],
                    args_schema=LookerQueryInput
                )
                all_tools.append(dynamic_tool)
                tool_descriptions_for_planner.append(f"- Tool: `{tool_name}`\n  Description: {config['description_for_router']}")
    
    python_tool_desc = "Executes Python code in a REPL environment. Use this for complex data analysis, calculations, or transformations on one or more datasets that have already been loaded from the cache. Always use `list_cached_datasets` first to see available data, then `load_dataframes_from_cache` to load them into pandas DataFrames."
    all_tools.extend([list_cached_datasets, load_dataframes_from_cache, PythonREPLTool()])
    tool_descriptions_for_planner.append(f"- Tool: `PythonREPLTool`\n  Description: {python_tool_desc}")
    
    return all_tools, "\n".join(tool_descriptions_for_planner)

ALL_TOOLS, TOOLS_DESCRIPTION = load_dynamic_tools()
tool_executor_node = ToolNode(ALL_TOOLS)

# ==============================================================================
# 2. Graph State & Nodes
# ==============================================================================
class GraphState(TypedDict):
    original_question: str
    plan: str
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    final_answer: str

# --- Planner Node ---
PLANNER_PROMPT = """You are an expert AI data analyst and planner. Your job is to create a detailed, step-by-step plan to answer a user's complex data question.

You have access to a suite of tools for data retrieval and analysis. Based on the user's question, you must identify the necessary steps, the right tools to use for each step, and the sequence in which to execute them.

**Available Tools:**
{tool_description}

**Instructions:**
1.  **Deconstruct the Request:** Break down the user's question into smaller, logical sub-questions.
2.  **Identify Necessary Tools:** For each sub-question, determine which of the available tools is best suited to provide the answer.
3.  **Formulate a Plan:** Create a clear, numbered, step-by-step plan. Each step must explicitly state which tool to use and what information to extract or calculate.
4.  **Consider Dependencies:** If a step requires data from a previous step (e.g., analyzing data that must first be retrieved), ensure the plan reflects this dependency. Any analysis using `PythonREPLTool` must be preceded by a step to retrieve the necessary data using a Looker tool.
5.  **Output:** Your final output must only be the numbered plan.

**User's Question:**
{question}

**Your Plan:**"""
planner_prompt = PromptTemplate.from_template(PLANNER_PROMPT)
planner_agent = planner_prompt | llm_pro

def planner_node(state: GraphState):
    print("--- 🧠 PLANNER ---")
    plan = planner_agent.invoke({
        "question": state["original_question"],
        "tool_description": TOOLS_DESCRIPTION
    })
    return {"plan": plan.content, "messages": [AIMessage(content=f"**Plan Created:**\n\n{plan.content}", name="Planner")]}


# --- Tool Agent Node ---
tool_agent_llm = llm_flash.bind_tools(ALL_TOOLS)

def tool_agent_node(state: GraphState):
    """Responsible for converting the plan into a specific tool call."""
    print("--- 🛠️ TOOL AGENT ---")
    
    tool_prompt = f"""You are a tool-calling expert. Your job is to select and execute the single most appropriate tool to address the *next* step in the provided plan.
The user's original question was: {state['original_question']}

**Full Plan:**
{state['plan']}

**Conversation History (Tool Outputs):**
{[msg.content for msg in state['messages'] if isinstance(msg, ToolMessage)]}

Based on the plan and history, which tool should be called NEXT?
"""
    tool_call_message = tool_agent_llm.invoke(tool_prompt)
    return {"messages": [tool_call_message]}


# --- Synthesizer Node ---
SYNTHESIZER_PROMPT = """You are an expert AI data analyst. Your task is to provide a final, comprehensive answer to the user's original question based on the plan and all the collected tool outputs.

**User's Original Question:**
{question}

**Execution Plan:**
{plan}

**Collected Data (Tool Outputs):**
{tool_outputs}

**Instructions:**
1.  **Synthesize:** Review all the tool outputs and combine the information to form a coherent, complete answer.
2.  **Address the Core Question:** Ensure your answer directly addresses all parts of the user's original question.
3.  **Format the Output:** Your final response must be in Markdown. Use bullet points or bold text to highlight key metrics and findings. Do not mention the tools by name.
"""
synthesizer_prompt = PromptTemplate.from_template(SYNTHESIZER_PROMPT)
synthesizer_agent = synthesizer_prompt | llm_pro

def response_synthesizer_node(state: GraphState):
    print("--- ✍️ SYNTHESIZER ---")
    final_answer = synthesizer_agent.invoke({
        "question": state["original_question"],
        "plan": state["plan"],
        "tool_outputs": "\n---\n".join([f"Output from {msg.name}:\n{msg.content}" for msg in state["messages"] if isinstance(msg, ToolMessage)])
    })
    return {"final_answer": final_answer.content}

# --- Router Node ---
def router_node(state: GraphState):
    """Decides whether to continue executing the plan or synthesize a final answer."""
    print("--- 🤔 ROUTER ---")
    
    # Simple heuristic: if there are more than 5 tool calls, synthesize.
    tool_calls = len([msg for msg in state["messages"] if isinstance(msg, ToolMessage)])
    if tool_calls > 5:
        print("--- Decision: Max tools reached, Synthesize ---")
        return "synthesize"

    router_prompt_template = f"""You are a routing expert. Your job is to decide if the AI agent has gathered enough information to answer the user's question, or if it needs to continue executing its plan.

**User's Original Question:**
{state['original_question']}

**Execution Plan:**
{state['plan']}

**Information Gathered So Far (Tool Outputs):**
{[msg.content for msg in state['messages'] if isinstance(msg, ToolMessage)]}

Based on the information gathered, is the plan fully complete?
- If all steps of the plan have been addressed and you have enough information, respond with the single word: **synthesize**.
- If there are still steps in the plan that need to be executed, respond with the single word: **continue**.
"""
    decision = llm_flash.invoke(router_prompt_template)
    
    if "synthesize" in decision.content.lower():
        print("--- Decision: Synthesize ---")
        return "synthesize"
    else:
        print("--- Decision: Continue ---")
        return "continue"

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
    
    workflow.add_conditional_edges(
        "tool_executor",
        router_node,
        {
            "continue": "tool_agent",
            "synthesize": "synthesizer"
        }
    )
    workflow.add_edge("synthesizer", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

agentic_chain = create_agentic_graph()