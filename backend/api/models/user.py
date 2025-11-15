from sqlalchemy import Column, String, Boolean, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base
import uuid

class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    plan_tier = Column(String, default="free")  # free, pro, enterprise
    max_users = Column(Integer, default=5)
    max_models = Column(Integer, default=5)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    users = relationship("User", back_populates="organization")

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    role = Column(String, default="member")
    is_superuser = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    organization = relationship("Organization", back_populates="users")

class Invitation(Base):
    __tablename__ = "invitations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    token = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, nullable=False)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    role = Column(String, default="member")
    invited_by = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)