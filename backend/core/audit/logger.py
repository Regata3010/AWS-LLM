from sqlalchemy.orm import Session
from api.models.audit_log import AuditLog
from api.models.user import User
from fastapi import Request
from typing import Optional, Dict
import uuid

class AuditLogger:
    """Centralized audit logging utility"""
    
    def __init__(self, db: Session, user: User, request: Request):
        self.db = db
        self.user = user
        self.request = request
    
    def log(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """
        Log an audit event
        
        Args:
            action: Action performed (e.g., 'model_trained', 'bias_detected')
            resource_type: Type of resource (e.g., 'model', 'analysis')
            resource_id: ID of the resource
            details: Additional context as JSON
        """
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            user_id=self.user.id,
            organization_id=self.user.organization_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=self.request.client.host if self.request.client else None,
            user_agent=self.request.headers.get("user-agent")
        )
        
        self.db.add(audit_log)
        self.db.commit()
        
        return audit_log

def get_audit_logger(
    db: Session,
    user: User,
    request: Request
) -> AuditLogger:
    """Factory function to create audit logger"""
    return AuditLogger(db, user, request)