
"""
BiasGuard AI Compliance Agent API
Endpoints for conversational compliance assistance
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from api.models.database import get_db
from api.models.user import User
from core.auth.dependencies import get_current_active_user
from core.src.logger import logging
from api.models.requests import ChatRequest
from api.models.responses import ChatResponse
from core.ai.agent import ComplianceAgent
from core.cache.redis_client import get_redis
import os
from core.ai.rag import get_rag_instance




router = APIRouter()



@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Chat with BiasGuard AI compliance assistant
    
    The agent uses:
    - CAG: Real-time data from your models (get_user_models, get_model_status, identify_violations)
    - RAG: CFPB/EEOC regulation knowledge (search_regulations, get_regulation_details)
    - LangGraph: Multi-step autonomous reasoning
    
    Example queries:
    - "What models do I have?"
    - "Is my loan model compliant?"
    - "What's the four-fifths rule?"
    - "Explain ECOA Section 1002.6(a)"
    
    RBAC: Users see only their organization's models (unless superuser)
    """
    try:
        agent = ComplianceAgent()
        
        # Add model context if provided
        message = request.message
        if request.model_id:
            message = f"[Context: Regarding model_id={request.model_id}] {message}"
        
        # Get response from agent
        result = await agent.chat(
            message=message,
            user=current_user,
            db=db,
            thread_id=request.thread_id
        )
        
        logging.info(f"Agent chat: {current_user.username} | Tools: {result['tools_used']}")
        
        return {
            "response": result["response"],
            "tools_used": result["tools_used"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logging.error(f"Agent chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )

@router.get("/chat/history")
async def get_chat_history(
    thread_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get conversation history for current user
    
    Returns last 20 messages (coming soon - requires checkpointer persistence)
    """
    return {
        "thread_id": thread_id or current_user.id,
        "messages": [],
        "note": "Conversation history persistence coming soon"
    }

@router.delete("/chat/history")
async def clear_chat_history(
    thread_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Clear conversation history"""
    return {
        "status": "success",
        "message": "History cleared (feature in development)"
    }

@router.post("/rag/ingest")
async def ingest_regulations(
    current_user: User = Depends(get_current_active_user)
):
    """
    Manually trigger regulation ingestion into Pinecone
    (Superuser only)
    
    Use this to reload regulations after updating source documents
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Only superusers can trigger regulation ingestion"
        )
    
    try:
        # Import here to avoid circular dependency
        from core.ai.rag import get_rag_instance
        
        rag = get_rag_instance()
        # Note: ingest_regulations method removed from RegulationRAG
        # Use the ingest_regulations.py script instead
        
        return {
            "status": "success",
            "message": "Use scripts/ingest_regulations.py to reload regulations",
            "note": "API-based ingestion coming soon"
        }
    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/health")
# async def agent_health_check():
#     """Check if AI agent is properly configured"""
#     try:
#         import os
        
#         checks = {
#             "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
#             "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
#             "agent_initialized": False,
#             "rag_initialized": False
#         }
        
#         # Try to initialize
#         try:
#             from core.ai.agent import get_agent
#             agent = get_agent()
#             checks["agent_initialized"] = True
#         except:
#             pass
        
#         try:
#             from core.ai.rag import get_rag_instance
#             rag = get_rag_instance()
#             checks["rag_initialized"] = True
#         except:
#             pass
        
#         all_good = all(checks.values())
        
#         return {
#             "status": "healthy" if all_good else "degraded",
#             "checks": checks,
#             "message": "AI agent ready" if all_good else "Some components not initialized"
#         }
    
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }

@router.get("/health")
async def agent_health_check():
    """
    Check if AI agent is properly configured
    
    CACHED for 30 seconds (reduces service check overhead)
    """
    try:
        
        redis = get_redis()
        cache_key = "agent:health"
        
        # Try cache first
        cached = redis.get(cache_key)
        if cached:
            logging.info("Cache HIT: agent health")
            return cached
        
        # Cache MISS - check services
        logging.info("Cache MISS: agent health - Checking services")
        
        import os
        
        checks = {
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
            "agent_initialized": False,
            "rag_initialized": False,
            "redis_connected": redis.enabled
        }
        
        # Try to initialize agent
        try:
            
            agent = ComplianceAgent()
            checks["agent_initialized"] = True
        except Exception as e:
            logging.warning(f"Agent init check failed: {e}")
            pass
        
        # Try to initialize RAG
        try:
            
            rag = get_rag_instance()
            checks["rag_initialized"] = True
        except Exception as e:
            logging.warning(f"RAG init check failed: {e}")
            pass
        
        all_good = all(checks.values())
        
        response = {
            "status": "healthy" if all_good else "degraded",
            "checks": checks,
            "message": "AI agent ready" if all_good else "Some components not initialized",
            "cached": False
        }
        
        
        redis.set(cache_key, response, ttl=30)
        
        return response
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "cached": False
        }