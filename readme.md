# **Looker \+ Gemini: The AI-Powered Semantic Layer Chatbot**

This project is a multi-agent Streamlit application demonstrating how to build an AI chatbot on top of Looker's semantic layer to answer complex data and analytical questions.

The system uses a **Steering Agent** to route user queries to specialized agents, including the **Looker Data Agent** for census questions and the **Python Agent** for data analysis on cached results.

## **🚀 Deployment Guide**

Follow these steps to deploy and run the application locally or on a cloud platform like Google Cloud Run.

### **Step 1: Set Up Credentials**

The application uses Streamlit Secrets to manage API keys. Create a folder named .streamlit in the project root and add a file named secrets.toml with the following structure:

\# .streamlit/secrets.toml

\# Google Gemini API Key  
GOOGLE\_API\_KEY \= "YOUR\_GEMINI\_API\_KEY"

\# Looker SDK Credentials (used by looker\_tool.py)  
\[looker\]  
base\_url \= "\[https://yourinstance.cloud.looker.com:19999\](https://yourinstance.cloud.looker.com:19999)" \# Note: Ensure you include the :19999 for API calls  
client\_id \= "YOUR\_LOOKER\_CLIENT\_ID"  
client\_secret \= "YOUR\_LOOKER\_CLIENT\_SECRET"

\# Proxycurl API Key (used by social\_tool.py)  
\# NOTE: This tool is used by the Social Agent and can be removed if not needed.  
PROXYCURL\_API\_KEY \= "YOUR\_PROXYCURL\_API\_KEY"

### **Step 2: Install Dependencies**

The project uses several Python libraries for the core functionality, as specified in requirements.txt.

1. **Create a virtual environment** (recommended):  
   python \-m venv venv  
   source venv/bin/activate  \# On Linux/macOS  
   .\\venv\\Scripts\\activate   \# On Windows

2. **Install packages** (using the provided requirements.txt):  
   pip install \-r requirements.txt

### **Step 3: Generate Looker Metadata (Mandatory)**

The tools/looker\_tool.py and tools/knowledge\_tool.py agents rely on a local file named acs\_census\_metadata.json (or your chosen name) to understand the available dimensions and measures.

The script 01\_fetch\_metadata.py is used to generate this file using the Looker SDK.

1. **Ensure Looker SDK is configured** (it uses the variables in secrets.toml or a local looker.ini file).  
2. **Run the script** (This assumes you are using the same Model/Explore names as the original project: data\_block\_acs\_bigquery::acs\_census\_data):  
   python 01\_fetch\_metadata.py

   This step generates the acs\_census\_metadata.json file, which is then loaded by the Looker tool.

### **Step 4: Run the Application**

Start the Streamlit application from the root directory:

streamlit run app.py

The application will launch in your default web browser, allowing you to interact with the multi-agent chatbot.

## **🛠️ Customization Guide: Using a Different Looker Explore**

To adapt this chatbot to a different Looker Explore (e.g., your internal financial model instead of the US Census Data), you need to modify three core files.

### **1\. Update Looker Model and Explore Names**

Edit tools/looker\_tool.py to target your new data source.

| File: tools/looker\_tool.py | Change From | Change To (Example) |
| :---- | :---- | :---- |
| MODEL\_NAME | "data\_block\_acs\_bigquery" | "your\_custom\_model" |
| EXPLORE\_NAME | "acs\_census\_data" | "your\_custom\_explore" |

### **2\. Generate New Metadata JSON**

You must generate a new metadata file containing the fields from your custom Explore.

1. **Update 01\_fetch\_metadata.py**:  
   * Change MODEL\_NAME and EXPLORE\_NAME in this file to match the values from the step above.  
   * Change output\_filename to something descriptive, e.g., "your\_explore\_metadata.json".  
2. **Run the updated script**:  
   python 01\_fetch\_metadata.py

3. This creates your new metadata file (e.g., your\_explore\_metadata.json).

### **3\. Update Metadata Loading References**

Update the Looker tool and the Knowledge tool to load your new metadata file.

* **In tools/looker\_tool.py**: Modify the \_get\_explore\_metadata function to reference your new JSON file:  
  \# tools/looker\_tool.py  
  def \_get\_explore\_metadata():  
      """Loads the explore metadata JSON file."""  
      try:  
          \# CHANGE FILENAME HERE  
          with open("your\_explore\_metadata.json") as f:  
              return json.dumps(json.load(f))  
          ...

* **In tools/knowledge\_tool.py**: Modify the get\_census\_data\_definition function to load the correct file:  
  \# tools/knowledge\_tool.py  
  @tool(args\_schema=TermInput)  
  def get\_census\_data\_definition(term: str) \-\> str:  
      \# CHANGE FILENAME HERE  
      with open('your\_explore\_metadata.json', 'r') as f:  
          metadata \= json.load(f)  
      ...

### **4\. Adjust Agent Strategy Prompts (Highly Recommended)**

For optimal AI performance, update the hardcoded prompts in app.py to guide the agent toward using relevant fields in your new domain.

* **app.py**: Find the LOOKER\_AGENT\_PROMPT\_TEMPLATE and specifically update the **ANALYST STRATEGY** section. Guide the agent to think like an analyst in your new domain (e.g., recommend fields like orders.count, products.inventory\_level, or financials.revenue instead of census demographics).
