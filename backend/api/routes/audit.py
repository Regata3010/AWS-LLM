from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import func, desc
from api.models.database import get_db
from api.models.audit_log import AuditLog
from api.models.user import User
from core.auth.dependencies import get_current_active_user, require_admin, require_role

router = APIRouter()

# Response Models
class AuditLogResponse(BaseModel):
    id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: dict
    ip_address: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True

@router.get("/logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=100),
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """
    Get audit logs (admin and compliance officers only)
    
    - Filter by action or resource_type
    - Pagination with skip/limit
    - Organization-scoped (only see your org's logs)
    """
    query = db.query(AuditLog).filter(
        AuditLog.organization_id == current_user.organization_id
    )
    
    if action:
        query = query.filter(AuditLog.action == action)
    
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    
    logs = query.order_by(desc(AuditLog.timestamp)).offset(skip).limit(limit).all()
    
    return logs

@router.get("/logs/stats")
async def get_audit_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Get audit statistics for the organization"""
    total_logs = db.query(AuditLog).filter(
        AuditLog.organization_id == current_user.organization_id
    ).count()
    
    # Count by action type
    actions = db.query(
        AuditLog.action,
        func.count(AuditLog.id).label('count')
    ).filter(
        AuditLog.organization_id == current_user.organization_id
    ).group_by(AuditLog.action).all()
    
    return {
        "total_logs": total_logs,
        "actions": {action: count for action, count in actions}
    }