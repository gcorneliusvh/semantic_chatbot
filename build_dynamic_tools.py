import os
import json
import looker_sdk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# NOTE: No need for dotenv anymore

# ==============================================================================
# 1. Configuration
# ==============================================================================
CONFIG_FILE = "explores.json"
OUTPUT_DIR = "generated_tools"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ... (LLM Initialization remains the same) ...

# ==============================================================================
# 5. Main Build Process
# ==============================================================================
def build_tools():
    """Main function to build all dynamic tools from the config file."""
    print("🚀 Starting dynamic tool build process...")

    # ... (rest of the function is the same until SDK initialization) ...

    # Initialize Looker SDK (reads from looker.ini)
    try:
        sdk = looker_sdk.init40("looker.ini")
        print("✅ Looker SDK initialized successfully.")
    except Exception as e:
        print(f"❌ FATAL: Could not initialize Looker SDK. Check your looker.ini file. Error: {e}")
        return

    # ... (rest of the function remains the same) ...

if __name__ == "__main__":
    build_tools()