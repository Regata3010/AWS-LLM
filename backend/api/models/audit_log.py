from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base
import uuid

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    action = Column(String, nullable=False)  # model_trained, bias_detected, mitigation_applied
    resource_type = Column(String, nullable=False)  # model, analysis, mitigation
    resource_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    # user = relationship("User", back_populates="audit_logs")
    
    # Indexes for fast queries
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_org_timestamp', 'organization_id', 'timestamp'),
        Index('idx_action', 'action'),
    )