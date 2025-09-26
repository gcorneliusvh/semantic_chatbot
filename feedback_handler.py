import json
import os
from datetime import datetime

FEEDBACK_FILE = "feedback_log.json"

def _load_feedback():
    """Loads the feedback log from the JSON file."""
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_feedback(session_id: str, question: str, answer: str, feedback: str):
    """
    Saves a user's feedback to the log file.

    Args:
        session_id: The unique identifier for the conversation.
        question: The user's prompt.
        answer: The agent's final answer.
        feedback: The feedback given ('positive' or 'negative').
    """
    feedback_log = _load_feedback()
    
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "feedback": feedback
    }
    
    feedback_log.append(feedback_entry)
    
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_log, f, indent=2)
        
    print(f"Feedback saved: {feedback}")