from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from api.models.database import get_db
from api.models import schemas, crud
from core.src.logger import logging
from api.models.user import User
from core.auth.dependencies import get_current_active_user
from core.cache.redis_client import get_redis

router = APIRouter()


@router.get("/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    """
    Get aggregated statistics for dashboard
    
    Returns:
        {
            "total_models": 24,
            "compliant_models": 18,
            "models_at_risk": 4,
            "critical_models": 2,
            "compliance_rate": 75.0,
            "total_analyses": 45
        }
    """
    try:
        redis = get_redis()
        org_id = None if current_user.is_superuser else current_user.organization_id
        cache_key = f"dashboard:summary:{org_id or 'all'}"
        cached = redis.get(cache_key)
        if cached:
            logging.info(f"Cache HIT: {cache_key}")
            return cached
        
        # Cache MISS - query database
        logging.info(f"Cache MISS: {cache_key} - Querying database")
        stats = crud.get_dashboard_stats(db,org_id)
        redis.set(cache_key, stats, ttl=60)  # Cache for 60 seconds 
        return stats
    except Exception as e:
        logging.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/recent")
async def get_recent_activity(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get recent model activities
    
    Query params:
        limit: Number of recent models (default: 10)
    """
    try:
        recent_models = crud.get_recent_models(db, limit=limit, organization_id=current_user.organization_id)
        
        return {
            "recent_models": [schemas.ModelResponse.model_validate(m) for m in recent_models]
        }
    except Exception as e:
        logging.error(f"Failed to get recent activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))