import jwt
import datetime
import logging
from app.config import config
from app.logger import logger

def generate_token(user_id, username, role):
    """Generate a JWT token for the user"""
    try:
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }
        
        token = jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")
        return token
    except Exception as e:
        logger.error(f"Error generating token: {str(e)}")
        raise Exception(f"Failed to generate token: {str(e)}")

def verify_token(token):
    """Verify a JWT token and return the payload"""
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        return None
        
def generate_reset_token(user_id: str) -> str:
    """Generate a password reset token"""
    try:
        payload = {
            "user_id": user_id,
            "purpose": "password_reset",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        
        token = jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")
        return token
    except Exception as e:
        logger.error(f"Error generating reset token: {str(e)}")
        raise Exception(f"Failed to generate reset token: {str(e)}")
