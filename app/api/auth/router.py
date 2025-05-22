# app/api/auth/router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
import secrets
import os
from werkzeug.security import check_password_hash

from app.auth.jwt_handler import generate_token
from app.auth.email import send_reset_email
from app.logger import logger
from app.config import config

# Create router
router = APIRouter()

# Models
class UserRegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    role: Optional[str] = "user"

class UserLoginRequest(BaseModel):
    username: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    password: str

# Auth routes
@router.post("/auth/register", status_code=201)
async def register(request: UserRegisterRequest):
    try:
        if not request.username or not request.password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        
        # Import here to avoid circular imports
        from app.db.repository import save_user
        
        if request.role == 'admin' and not config.ALLOW_ADMIN_REGISTRATION:
            raise HTTPException(status_code=403, detail="Admin registration is not allowed")
        
        user_id = save_user(request.username, request.password, request.email, request.role)
        
        if not user_id:
            raise HTTPException(status_code=409, detail="Username already exists")
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/login")
async def login(request: UserLoginRequest):
    try:
        if not request.username or not request.password:
            raise HTTPException(status_code=400, detail="Username and password are required")
        
        # Import here to avoid circular imports
        from app.db.repository import get_user_by_username, update_last_login
        
        user = get_user_by_username(request.username)
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        if not check_password_hash(user["password_hash"], request.password):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        update_last_login(user["id"])
        
        token = generate_token(user["id"], user["username"], user["role"])
        
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    try:
        if not request.email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Import here to avoid circular imports
        from app.db.repository import get_user_by_email, save_reset_token
        
        user = get_user_by_email(request.email)
        
        # Always return success even if email not found (security best practice)
        if not user:
            return {
                "success": True,
                "message": "If your email is registered, you will receive a password reset link"
            }
        
        reset_token = secrets.token_urlsafe(32)
        save_reset_token(user["id"], reset_token)
        
        # Get frontend URL and sanitize it
        frontend_url = config.FRONTEND_URL.strip()
        # Remove any non-breaking spaces or other invisible characters
        frontend_url = ''.join(c for c in frontend_url if c.isprintable() and c != '\xa0')
        
        # Create sanitized reset link
        reset_link = f"{frontend_url}/reset-password?token={reset_token}"
        
        # Log the link for debugging
        logger.info(f"RESET LINK FOR {request.email}: {reset_link}")
        
        # Return success without actually sending email
        return {
            "success": True,
            "message": "If your email is registered, you will receive a password reset link",
            "debug_token": reset_token if config.DEBUG else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forgot password: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    try:
        if not request.token or not request.password:
            raise HTTPException(status_code=400, detail="Token and new password are required")
        
        # Import here to avoid circular imports
        from app.db.repository import verify_reset_token, update_user_password, invalidate_reset_token
        
        user_id = verify_reset_token(request.token)
        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        update_user_password(user_id, request.password)
        
        invalidate_reset_token(request.token)
        
        return {
            "success": True,
            "message": "Password has been reset successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in reset password: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
