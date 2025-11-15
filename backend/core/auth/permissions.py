# backend/core/auth/permissions.py
"""
RBAC Permission System for BiasGuard

Two-level permissions:
1. is_superuser: Platform admin (me) - sees everything
2. role: Organization-level permissions (admin/member)

Permission Matrix:
┌─────────────────────┬───────────┬──────────┬──────────────┐
│ Action              │ Superuser │ Admin    │ Member       │
├─────────────────────┼───────────┼──────────┼──────────────┤
│ View models         │ All orgs  │ Own org  │ Own org      │
│ Create model        │ ✓         │ ✓        │ ✓            │
│ Delete model        │ ✓         │ ✓        │ ✗            │
│ Upload predictions  │ ✓         │ ✓        │ ✓            │
│ Analyze bias        │ ✓         │ ✓        │ ✓            │
│ Generate reports    │ ✓         │ ✓        │ ✓            │
│ Invite users        │ ✓         │ ✓        │ ✗            │
│ Manage settings     │ ✓         │ ✓        │ ✗            │
│ View audit logs     │ ✓         │ ✓        │ ✗            │
└─────────────────────┴───────────┴──────────┴──────────────┘
"""

from api.models.user import User
from fastapi import HTTPException, status

class Permission:
    """Permission definitions"""
    
    # Model permissions
    CREATE_MODEL = "create_model"
    VIEW_MODEL = "view_model"
    DELETE_MODEL = "delete_model"
    
    # Prediction permissions
    UPLOAD_PREDICTIONS = "upload_predictions"
    ANALYZE_BIAS = "analyze_bias"
    
    # Report permissions
    GENERATE_REPORT = "generate_report"
    VIEW_REPORT = "view_report"
    
    # User management
    INVITE_USER = "invite_user"
    MANAGE_USERS = "manage_users"
    
    # Organization
    MANAGE_ORG_SETTINGS = "manage_org_settings"
    VIEW_AUDIT_LOGS = "view_audit_logs"

# Permission mappings
ROLE_PERMISSIONS = {
    "admin": [
        Permission.CREATE_MODEL,
        Permission.VIEW_MODEL,
        Permission.DELETE_MODEL,
        Permission.UPLOAD_PREDICTIONS,
        Permission.ANALYZE_BIAS,
        Permission.GENERATE_REPORT,
        Permission.VIEW_REPORT,
        Permission.INVITE_USER,
        Permission.MANAGE_USERS,
        Permission.MANAGE_ORG_SETTINGS,
        Permission.VIEW_AUDIT_LOGS,
    ],
    "member": [
        Permission.CREATE_MODEL,
        Permission.VIEW_MODEL,
        # No delete
        Permission.UPLOAD_PREDICTIONS,
        Permission.ANALYZE_BIAS,
        Permission.GENERATE_REPORT,
        Permission.VIEW_REPORT,
        # No invite/manage
    ],
}

def has_permission(user: User, permission: str) -> bool:
    """
    Check if user has specific permission
    
    Superusers have all permissions
    """
    # Superuser has all permissions
    if user.is_superuser:
        return True
    
    # Check role permissions
    user_permissions = ROLE_PERMISSIONS.get(user.role, [])
    return permission in user_permissions

def require_permission(user: User, permission: str):
    """
    Raise exception if user doesn't have permission
    
    Usage:
        require_permission(current_user, Permission.DELETE_MODEL)
    """
    if not has_permission(user, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have permission: {permission}. Required role: admin"
        )

def can_access_organization(user: User, org_id: str) -> bool:
    """Check if user can access organization's data"""
    #superuser check everything
    if user.is_superuser:
        return True
    
    # User can only access their own organization
    return user.organization_id == org_id

def filter_by_organization(query, model_class, user: User):
    """
    Filter SQLAlchemy query by organization
    
    Superusers see all data, others see only their org
    
    Usage:
        query = db.query(ExternalModel)
        query = filter_by_organization(query, ExternalModel, current_user)
    """
    if user.is_superuser:
        return query  # No filter for superuser
    
    return query.filter(model_class.organization_id == user.organization_id)