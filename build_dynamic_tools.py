import os
import json
import looker_sdk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# 1. Configuration
# =================================d=============================================
CONFIG_FILE = "explores.json"
OUTPUT_DIR = "generated_tools"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. LLM Initialization for Prompt Generation
# ==============================================================================
# NOTE: This script assumes you have set your GOOGLE_API_KEY as an
# environment variable before running.
try:
    # Use a powerful model for this creative task. Temperature is slightly > 0
    # to allow for more creative and nuanced prompt writing.
    llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    print("✅ Gemini Pro LLM initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing LLM. Make sure GOOGLE_API_KEY is set in your environment. Error: {e}")
    exit()

# ==============================================================================
# 3. Prompt Template for AI-Powered Prompt Engineering
# ==============================================================================
PROMPT_GENERATION_TEMPLATE = """
You are an expert AI prompt engineer specializing in creating instructions for autonomous data analysis agents that use Looker.
Your task is to write a detailed and effective prompt template for a LangChain ReAct agent that will query a specific Looker Explore.

Based on the provided metadata (a list of all available dimensions and measures), you must generate a prompt that guides another AI to act as a specialized data analyst for this specific dataset.

**METADATA FOR THE EXPLORE:**
{metadata}

The generated prompt MUST include the following sections, formatted in Markdown:

1.  **Identity**: Start with a clear identity statement, for example: "You are an expert data analyst for US Census Data." or "You are a senior e-commerce sales analyst."

2.  **Tool Overview**: Briefly state that the agent has a primary tool, `LookerDataQuery`, for retrieving data and visualizations from the pre-defined Looker Explore.

3.  **ANALYST STRATEGY**: This is the most critical section. You must analyze the provided fields and write 3-5 specific, actionable bullet points that guide the agent on how to answer strategic questions.
    * Infer the primary purpose of the data. Is it for sales analysis? Demographic research? Financial reporting?
    * Identify the most important measures (e.g., `orders.total_revenue`, `blockgroup.total_pop`, `financials.net_profit`). These are the key metrics.
    * Identify the most important dimensions for grouping and filtering (e.g., `products.category`, `state.state_name`, `transactions.date`).
    * Provide at least two concrete examples of how to combine these fields to answer a typical business question. For instance: "To analyze sales performance over time, your primary action should be to query the `orders.total_revenue` measure grouped by the `orders.created_date` dimension."

4.  **FINAL ANSWER STRATEGY**: Instruct the agent on how to format its final response. It MUST produce a Markdown response with two sections: '### Summary' and '### Insights'. The 'Insights' section should be a brief analysis based on the data retrieved.

Do not include LangChain placeholders like `{{tools}}` or `{{agent_scratchpad}}`. You are writing the *content* of the prompt template that will be used by the agent later. Return only the generated text for the prompt.
"""

# ==============================================================================
# 4. Helper Functions
# ==============================================================================
def fetch_explore_metadata(sdk, model_name, explore_name):
    """Fetches and processes metadata for a single Looker Explore."""
    print(f"  - Fetching metadata for {model_name}::{explore_name}...")
    try:
        explore = sdk.lookml_model_explore(
            lookml_model_name=model_name,
            explore_name=explore_name,
            fields="fields"
        )
        fields_data = []
        if explore.fields:
            if explore.fields.dimensions:
                for dim in explore.fields.dimensions:
                    fields_data.append({"name": dim.name, "label": dim.label, "description": dim.description, "type": "dimension"})
            if explore.fields.measures:
                 for mea in explore.fields.measures:
                    fields_data.append({"name": mea.name, "label": mea.label, "description": mea.description, "type": "measure"})
        print(f"  - Successfully fetched {len(fields_data)} fields.")
        return fields_data
    except Exception as e:
        print(f"  - ❌ ERROR fetching metadata: {e}")
        return None

def generate_prompt_from_metadata(metadata):
    """Uses an LLM to generate a custom agent prompt from explore metadata."""
    print("  - Generating custom agent prompt with Gemini...")
    prompt_template = PromptTemplate.from_template(PROMPT_GENERATION_TEMPLATE)
    chain = prompt_template | llm_pro | StrOutputParser()

    metadata_str = json.dumps(metadata, indent=2)

    try:
        generated_prompt = chain.invoke({"metadata": metadata_str})
        print("  - ✅ Successfully generated custom prompt.")
        return generated_prompt
    except Exception as e:
        print(f"  - ❌ ERROR during prompt generation: {e}")
        return None

# ==============================================================================
# 5. Main Build Process
# ==============================================================================
def build_tools():
    """Main function to build all dynamic tools from the config file."""
    print("🚀 Starting dynamic tool build process...")

    try:
        with open(CONFIG_FILE, 'r') as f:
            explores_config = json.load(f)
    except FileNotFoundError:
        print(f"❌ FATAL: Configuration file '{CONFIG_FILE}' not found. Please create it.")
        return

    # Initialize Looker SDK (reads from environment variables)
    try:
        sdk = looker_sdk.init40()
        print("✅ Looker SDK initialized successfully.")
    except Exception as e:
        print(f"❌ FATAL: Could not initialize Looker SDK. Check your credentials (LOOKERSDK_...). Error: {e}")
        return

    for explore in explores_config:
        tool_name = explore.get("tool_name")
        model = explore.get("model_name")
        explore_name = explore.get("explore_name")

        if not all([tool_name, model, explore_name]):
            print(f"⚠️ WARNING: Skipping invalid entry in {CONFIG_FILE}: {explore}")
            continue

        print(f"\n--- Processing Tool: {tool_name} ---")

        # Step 1: Fetch Metadata from Looker
        metadata = fetch_explore_metadata(sdk, model, explore_name)
        if not metadata:
            continue

        # Step 2: Generate the custom prompt for this explore's agent
        custom_prompt = generate_prompt_from_metadata(metadata)
        if not custom_prompt:
            continue

        # Step 3: Assemble the final tool configuration object
        tool_config = {
            "tool_name": tool_name,
            "model_name": model,
            "explore_name": explore_name,
            "description_for_router": explore.get("description_for_router", f"Tool for querying the {explore_name} dataset."),
            "metadata": metadata,
            "agent_prompt_template": custom_prompt
        }

        # Step 4: Save the complete configuration to a dedicated file
        output_path = os.path.join(OUTPUT_DIR, f"{tool_name}.json")
        with open(output_path, 'w') as f:
            json.dump(tool_config, f, indent=2)

        print(f"  - ✅ Successfully created tool configuration at '{output_path}'")

    print("\n🎉 Dynamic tool build process complete.")

if __name__ == "__main__":
    # Ensure you have set your environment variables for GOOGLE_API_KEY and LOOKERSDK
    build_tools()