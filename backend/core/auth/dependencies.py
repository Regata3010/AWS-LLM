from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from api.models.database import get_db
from api.models.user import User
from core.auth.jwt import decode_access_token
from typing import List
from core.src.logger import logging

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    
    # Decode token
    payload = decode_access_token(token)
    user_id: str = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (convenience function)"""
    return current_user


# RBAC: Role-Based Access Control
class RoleChecker:
    """
    Dependency to check user roles
    
    Usage:
        @router.delete("/models/{id}")
        async def delete_model(
            user: User = Depends(require_role(["admin"]))
        ):
            ...
    """
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: User = Depends(get_current_user)) -> User:
        # Superuser bypasses all role checks
        if user.is_superuser:
            return user
        
        # Check if user has required role
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(self.allowed_roles)}. Your role: {user.role}"
            )
        return user

def require_role(roles: List[str]):
    """Factory function to create role checker"""
    return RoleChecker(allowed_roles=roles)

# Convenience role checkers
require_admin = require_role(["admin"])
require_member_or_admin = require_role(["admin", "member"])

# RBAC: Permission Helpers
def check_organization_access(user: User, resource_org_id: str) -> bool:
    """
    Check if user can access resource from organization
    
    Returns True if:
    - User is superuser (platform admin)
    - User belongs to same organization
    """
    if user.is_superuser:
        return True
    
    return user.organization_id == resource_org_id

def check_can_delete(user: User, resource_org_id: str) -> bool:
    """
    Check if user can delete resource
    
    Only admins and superusers can delete
    """
    if user.is_superuser:
        return True
    
    if user.organization_id != resource_org_id:
        return False
    
    return user.role == "admin"

def check_can_invite(user: User) -> bool:
    """
    Check if user can invite others to their organization
    
    Only admins and superusers can invite
    """
    if user.is_superuser:
        return True
    
    return user.role == "admin"

def require_same_org(user: User, resource_org_id: str):
    """Raise exception if user cannot access resource"""
    if not check_organization_access(user, resource_org_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this organization's resources"
        )
        
async def verify_websocket_token(token: str, db: Session) -> User:
    """
    Verify JWT token for WebSocket connections
    
    Usage:
        user = await verify_websocket_token(token, db)
    
    Raises ValueError if auth fails
    """
    try:
        # Decode token
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise ValueError("Invalid token payload")
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise ValueError("User not found")
        
        if not user.is_active:
            raise ValueError("User is inactive")
        
        return user
        
    except Exception as e:
        logging.error(f"WebSocket auth failed: {e}")
        raise ValueError(f"Authentication failed: {str(e)}")


def check_model_access(user: User, model_org_id: str) -> bool:
    """
    Check if user can access a model
    
    Returns True if:
    - User is superuser (platform admin)
    - User belongs to same organization as model
    """
    if user.is_superuser:
        return True
    
    if not model_org_id:
        # Model has no org (legacy data)
        return True
    
    return user.organization_id == model_org_id