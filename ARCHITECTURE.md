# Architecture Manual
This document details the design and data flow of the Looker-Gemini chatbot.

## Agentic Design
The system uses a multi-agent design, orchestrated by a "Steering Agent."

1.  **Steering Agent:** The main router. It analyzes the user's intent and routes the query to the correct specialized agent (or "tool").
2.  **Social Agent:** Handles non-data-related small talk (e.g., "Hello," "How are you?").
3.  **General Knowledge Agent:** Uses Google Search to answer general questions (e.g., "Are there any Telus offices in the US?").
4.  **Looker Data Agent:** The core of the system. This is a specialized agent that answers questions about the **ACS Census** data.
5.  **Consolidating Agent:** This is the "meta-agent" (or `AgentExecutor` in LangChain) that takes the outputs from all tools and synthesizes a final, coherent answer.

*...more to come...*