from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import timedelta
import uuid
from api.models.database import get_db
from api.models.user import User, Organization
from core.auth.password import verify_password, get_password_hash
from core.auth.jwt import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from core.auth.dependencies import get_current_active_user
from api.models.responses import UserResponse, TokenResponse
from api.models.requests import UserRegister


router = APIRouter()


# ENDPOINTS
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user and create their organization
    
    - First user becomes superuser (admin)
    - Creates organization with provided name
    - Returns JWT token and user data
    """
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already registered"
        )
    
    # Check if this is the first user (becomes admin)
    user_count = db.query(User).count()
    is_first_user = user_count == 0
    
    # Create organization
    if is_first_user:
        org_name = "BiasGuard Admin Owner"  # First user gets admin org
    else:
        org_name = user_data.organization_name
    
    org = Organization(
        id=str(uuid.uuid4()),
        name=org_name,
        plan_tier="free",
        max_users=50,
        max_models=1000
    )
    db.add(org)
    db.flush()
    
    # Create user
    user = User(
        id=str(uuid.uuid4()),
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.username,  # Use username as full_name
        organization_id=org.id,
        role="admin" if is_first_user else "member",
        is_superuser=is_first_user,  # First user is superuser
        is_active=True,
        is_verified=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "is_superuser": user.is_superuser,
            "organization_id": user.organization_id
        }
    )
    
    # Return response matching frontend expectations
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "organization_id": user.organization_id,
            "organization_name": org.name,
            "role": user.role,
            "is_superuser": user.is_superuser,
            "is_active": user.is_active
        }
    }

@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login with username and password
    
    Returns JWT access token and user data
    """
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) | (User.email == form_data.username)
    ).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    # Get organization name
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    org_name = org.name if org else None
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "is_superuser": user.is_superuser,
            "organization_id": user.organization_id
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # Return response matching frontend expectations
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "organization_id": user.organization_id,
            "organization_name": org_name,
            "role": user.role,
            "is_superuser": user.is_superuser,
            "is_active": user.is_active
        }
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    # Get organization name
    db = next(get_db())
    org = db.query(Organization).filter(Organization.id == current_user.organization_id).first()
    org_name = org.name if org else None
    
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "organization_id": current_user.organization_id,
        "organization_name": org_name,
        "role": current_user.role,
        "is_superuser": current_user.is_superuser,
        "is_active": current_user.is_active
    }

@router.post("/logout")
async def logout():
    """
    Logout (client should delete token)
    
    JWT tokens are stateless, so we just return success.
    Client must delete the token from storage.
    """
    return {"message": "Successfully logged out"}