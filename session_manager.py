from langchain_core.chat_history import InMemoryChatMessageHistory

# This dictionary is the SINGLE SOURCE OF TRUTH for all session data.
# It will be shared across all modules that import it.
session_histories = {}

def get_session_history(session_id: str) -> dict:
    """
    Retrieves or creates a session history for the given session_id.
    Each session contains the chat history and the last response object.
    """
    if session_id not in session_histories:
        session_histories[session_id] = {
            "chat_history": InMemoryChatMessageHistory(),
            "last_response": None  # To store the last Response object
        }
    return session_histories[session_id]

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    A helper function specifically for LangChain runnables that need
    a function returning a BaseChatMessageHistory object.
    """
    return get_session_history(session_id)["chat_history"]