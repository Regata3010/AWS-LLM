"""
Conversation Memory Management for BiasGuard Agent
(Optional - LangGraph has built-in memory, but adds persistence)
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Optional
import os

class BiasGuardMemory:
    """
    Enhanced memory with SQLite persistence
    
    Default MemorySaver stores in RAM (lost on restart)
    SqliteSaver persists conversations to database
    """
    
    def __init__(self, db_path: str = "biasguard_conversations.db"):
        self.db_path = db_path
        
        # Use SQLite for persistence
        if os.path.exists(db_path):
            self.checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        else:
            # First time - create DB
            self.checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    
    def get_checkpointer(self):
        """Get the checkpointer for LangGraph agent"""
        return self.checkpointer
    
    def get_conversation_history(self, thread_id: str, limit: int = 20) -> list:
        """
        Retrieve conversation history for a user
        
        Args:
            thread_id: Usually user.id
            limit: Number of recent messages
        
        Returns:
            List of messages
        """
        # LangGraph checkpointer API
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.checkpointer.get(config)
            if state and "messages" in state.values:
                return state.values["messages"][-limit:]
        except:
            pass
        
        return []
    
    def clear_conversation(self, thread_id: str):
        """Clear conversation history for a user"""
        config = {"configurable": {"thread_id": thread_id}}
        # Note: LangGraph doesn't have built-in clear, so we'd need to handle this
        # For now, just create new thread_id to start fresh
        pass

# Singleton
_memory_instance = None

def get_memory() -> BiasGuardMemory:
    """Get or create memory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = BiasGuardMemory()
    return _memory_instance