# app/db/repository.py
import logging
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from contextlib import contextmanager
from werkzeug.security import generate_password_hash

from app.config import config
from app.logger import logger

class DatabaseError(Exception):
    pass

# Database connection management
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            config.DATABASE_URL,
            cursor_factory=RealDictCursor
        )
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise DatabaseError(f"Failed to connect to database: {str(e)}")
    finally:
        if conn is not None:
            conn.close()

@contextmanager
def get_db_cursor(commit=False):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Database query error: {str(e)}")
            raise DatabaseError(f"Database query failed: {str(e)}")
        finally:
            cursor.close()

# Database initialization
def init_db():
    try:
        with get_db_cursor(commit=True) as cur:
            # Create conversations table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_input TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create sessions table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id UUID PRIMARY KEY,
                started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                client_ip TEXT,
                metadata JSONB
            )
            """)
            
            # Create indexes
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations ((metadata->>'session_id'));
            """)
            
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at);
            """)
            
            # Create users table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email VARCHAR(100),
                role VARCHAR(20) DEFAULT 'user',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP WITH TIME ZONE
            )
            """)
            
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
            """)
            
            # Create password reset tokens table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id),
                token TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                is_valid BOOLEAN DEFAULT TRUE
            )
            """)
            
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_token ON password_reset_tokens (token);
            """)
            
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise DatabaseError(f"Failed to initialize database: {str(e)}")

# Conversation functions
def save_conversation_to_db(user_input: str, ai_response: str, conversation_id: str = None, metadata: Dict[str, Any] = None) -> str:
    try:
        with get_db_cursor(commit=True) as cur:
            if metadata and 'session_id' in metadata:
                session_id = metadata['session_id']
                _update_or_create_session(cur, session_id, metadata)
            
            if conversation_id:
                cur.execute(
                    """
                    INSERT INTO conversations (id, user_input, ai_response, metadata)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (conversation_id, user_input, ai_response, Json(metadata) if metadata else None)
                )
            else:
                cur.execute(
                    """
                    INSERT INTO conversations (user_input, ai_response, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (user_input, ai_response, Json(metadata) if metadata else None)
                )
            
            result = cur.fetchone()
            return str(result['id'])
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        raise DatabaseError(f"Failed to save conversation: {str(e)}")

def _update_or_create_session(cur, session_id: str, metadata: Dict[str, Any]):
    try:
        cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
        session_exists = cur.fetchone() is not None
        
        if session_exists:
            cur.execute(
                """
                UPDATE sessions 
                SET last_activity = CURRENT_TIMESTAMP,
                    metadata = metadata || %s::jsonb
                WHERE id = %s
                """,
                (Json({k: v for k, v in metadata.items() if k not in ['session_id', 'client_ip', 'user_agent']}), session_id)
            )
        else:
            cur.execute(
                """
                INSERT INTO sessions (id, user_agent, client_ip, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    session_id,
                    metadata.get('user_agent'),
                    metadata.get('client_ip'),
                    Json({k: v for k, v in metadata.items() if k not in ['session_id', 'client_ip', 'user_agent']})
                )
            )
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise DatabaseError(f"Failed to update session: {str(e)}")


# Conversation retrieval functions
def get_conversation_by_id(conversation_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT id, user_input, ai_response, metadata, created_at
                FROM conversations
                WHERE id = %s
                """,
                (conversation_id,)
            )
            
            result = cur.fetchone()
            if not result:
                return None
                
            return _format_conversation_result(result)
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise DatabaseError(f"Failed to retrieve conversation: {str(e)}")

def get_all_conversations(session_id: str = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    try:
        with get_db_cursor() as cur:
            if session_id:
                cur.execute(
                    """
                    SELECT id, user_input, ai_response, metadata, created_at
                    FROM conversations
                    WHERE metadata->>'session_id' = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (session_id, limit, offset)
                )
            else:
                cur.execute(
                    """
                    SELECT id, user_input, ai_response, metadata, created_at
                    FROM conversations
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset)
                )
            
            results = cur.fetchall()
            return [_format_conversation_result(row) for row in results]
    except Exception as e:
        logger.error(f"Error retrieving conversations: {str(e)}")
        raise DatabaseError(f"Failed to retrieve conversations: {str(e)}")

def _format_conversation_result(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "user_input": row["user_input"],
        "ai_response": row["ai_response"],
        "metadata": row["metadata"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None
    }

# Session management functions
def get_session_by_id(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT id, started_at, last_activity, user_agent, client_ip, metadata
                FROM sessions
                WHERE id = %s
                """,
                (session_id,)
            )
            
            result = cur.fetchone()
            if not result:
                return None
                
            return {
                "id": str(result["id"]),
                "started_at": result["started_at"].isoformat() if result["started_at"] else None,
                "last_activity": result["last_activity"].isoformat() if result["last_activity"] else None,
                "user_agent": result["user_agent"],
                "client_ip": result["client_ip"],
                "metadata": result["metadata"]
            }
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        raise DatabaseError(f"Failed to retrieve session: {str(e)}")

def get_all_sessions(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    try:
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT id, started_at, last_activity, user_agent, client_ip, metadata
                FROM sessions
                ORDER BY last_activity DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            
            results = cur.fetchall()
            return [{
                "id": str(row["id"]),
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "last_activity": row["last_activity"].isoformat() if row["last_activity"] else None,
                "user_agent": row["user_agent"],
                "client_ip": row["client_ip"],
                "metadata": row["metadata"]
            } for row in results]
    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        raise DatabaseError(f"Failed to retrieve sessions: {str(e)}")

# User management functions
def save_user(username: str, password: str, email: str = None, role: str = "user") -> Optional[str]:
    try:
        password_hash = generate_password_hash(password)
        
        with get_db_cursor(commit=True) as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return None
            
            cur.execute(
                """
                INSERT INTO users (username, password_hash, email, role)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (username, password_hash, email, role)
            )
            
            result = cur.fetchone()
            return str(result['id'])
    except Exception as e:
        logger.error(f"Error saving user: {str(e)}")
        raise DatabaseError(f"Failed to save user: {str(e)}")

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    try:
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT id, username, password_hash, email, role, created_at, last_login
                FROM users
                WHERE username = %s
                """,
                (username,)
            )
            
            result = cur.fetchone()
            if not result:
                return None
                
            return {
                "id": str(result["id"]),
                "username": result["username"],
                "password_hash": result["password_hash"],
                "email": result["email"],
                "role": result["role"],
                "created_at": result["created_at"].isoformat() if result["created_at"] else None,
                "last_login": result["last_login"].isoformat() if result["last_login"] else None
            }
    except Exception as e:
        logger.error(f"Error retrieving user: {str(e)}")
        raise DatabaseError(f"Failed to retrieve user: {str(e)}")

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    try:
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT id, username, password_hash, email, role
                FROM users
                WHERE email = %s
                """,
                (email,)
            )
            
            result = cur.fetchone()
            if not result:
                return None
                
            return {
                "id": str(result["id"]),
                "username": result["username"],
                "password_hash": result["password_hash"],
                "email": result["email"],
                "role": result["role"]
            }
    except Exception as e:
        logger.error(f"Error retrieving user by email: {str(e)}")
        raise DatabaseError(f"Failed to retrieve user: {str(e)}")

def update_last_login(user_id: str) -> bool:
    try:
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id
                """,
                (user_id,)
            )
            
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Error updating last login: {str(e)}")
        raise DatabaseError(f"Failed to update last login: {str(e)}")

# Password reset functions
def save_reset_token(user_id: str, token: str, expires_at: datetime.datetime = None) -> bool:
    try:
        if not expires_at:
            expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE password_reset_tokens
                SET is_valid = FALSE
                WHERE user_id = %s AND is_valid = TRUE
                """,
                (user_id,)
            )
            
            cur.execute(
                """
                INSERT INTO password_reset_tokens (user_id, token, expires_at, is_valid)
                VALUES (%s, %s, %s, TRUE)
                RETURNING id
                """,
                (user_id, token, expires_at)
            )
            
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Error saving reset token: {str(e)}")
        raise DatabaseError(f"Failed to save reset token: {str(e)}")

def verify_reset_token(token: str) -> Optional[str]:
    try:
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT user_id
                FROM password_reset_tokens
                WHERE token = %s AND is_valid = TRUE AND expires_at > NOW()
                """,
                (token,)
            )
            
            result = cur.fetchone()
            if not result:
                return None
                
            return str(result["user_id"])
    except Exception as e:
        logger.error(f"Error verifying reset token: {str(e)}")
        raise DatabaseError(f"Failed to verify reset token: {str(e)}")

def update_user_password(user_id: str, new_password: str) -> bool:
    try:
        password_hash = generate_password_hash(new_password)
        
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE users
                SET password_hash = %s
                WHERE id = %s
                RETURNING id
                """,
                (password_hash, user_id)
            )
            
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Error updating user password: {str(e)}")
        raise DatabaseError(f"Failed to update user password: {str(e)}")

def invalidate_reset_token(token: str) -> bool:
    try:
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE password_reset_tokens
                SET is_valid = FALSE
                WHERE token = %s
                RETURNING id
                """,
                (token,)
            )
            
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Error invalidating reset token: {str(e)}")
        raise DatabaseError(f"Failed to invalidate reset token: {str(e)}")

# Statistics functions
def get_conversation_stats() -> Dict[str, Any]:
    try:
        with get_db_cursor() as cur:
            stats = {}
            
            cur.execute("SELECT COUNT(*) FROM conversations")
            stats["total_conversations"] = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '24 hours'")
            stats["conversations_24h"] = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '7 days'")
            stats["conversations_7d"] = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) FROM sessions")
            stats["total_sessions"] = cur.fetchone()["count"]
            
            cur.execute("SELECT COUNT(*) FROM sessions WHERE last_activity > NOW() - INTERVAL '24 hours'")
            stats["active_sessions_24h"] = cur.fetchone()["count"]
            
            cur.execute("""
                SELECT AVG(conversation_count) 
                FROM (
                    SELECT COUNT(*) as conversation_count 
                    FROM conversations 
                    GROUP BY metadata->>'session_id'
                ) as session_counts
            """)
            result = cur.fetchone()
            stats["avg_conversations_per_session"] = float(result["avg"]) if result["avg"] is not None else 0
            
            return stats
    except Exception as e:
        logger.error(f"Error retrieving conversation stats: {str(e)}")
        raise DatabaseError(f"Failed to retrieve conversation stats: {str(e)}")

def delete_conversation(conversation_id: str) -> bool:
    try:
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM conversations WHERE id = %s RETURNING id",
                (conversation_id,)
            )
            
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise DatabaseError(f"Failed to delete conversation: {str(e)}")

def delete_session_data(session_id: str) -> bool:
    try:
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM conversations WHERE metadata->>'session_id' = %s",
                (session_id,)
            )
            
            cur.execute(
                "DELETE FROM sessions WHERE id = %s RETURNING id",
                (session_id,)
            )
            
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Error deleting session data: {str(e)}")
        raise DatabaseError(f"Failed to delete session data: {str(e)}")
