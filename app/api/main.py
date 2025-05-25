from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import asyncio
import uuid
import time
import os
import json

from app.agent.manus import Manus
from app.flow.flow_factory import FlowFactory, FlowType
from app.config import config
from app.logger import logger
from app.api.auth.router import router as auth_router
from app.db.repository import init_db

app = FastAPI(
    title="OpenAgentFramework API",
    docs_url="/",  
    redoc_url="/redoc" 
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth_router)

# Session manager for agents
class AgentSessionManager:
    def __init__(self, max_idle_time=1800):  # 30 minutes default
        self.sessions = {}
        self.last_activity = {}
        self.max_idle_time = max_idle_time
        
    async def get_agent(self, session_id=None):
        """Get or create an agent for the given session"""
        if not session_id or session_id not in self.sessions:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = await Manus.create()
            
        self.last_activity[session_id] = time.time()
        return self.sessions[session_id], session_id
        
    async def cleanup_idle_sessions(self):
        """Clean up idle sessions"""
        current_time = time.time()
        for session_id, last_active in list(self.last_activity.items()):
            if current_time - last_active > self.max_idle_time:
                await self.cleanup_session(session_id)
                
    async def cleanup_session(self, session_id):
        """Clean up a specific session"""
        if session_id in self.sessions:
            await self.sessions[session_id].cleanup()
            del self.sessions[session_id]
            del self.last_activity[session_id]
            
    async def cleanup_all(self):
        """Clean up all sessions"""
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)

# Create session manager
session_manager = AgentSessionManager()

# Request/Response models
class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    stream: bool = False
    
class ChatResponse(BaseModel):
    session_id: str
    response: str
    
class FlowRequest(BaseModel):
    prompt: str
    flow_type: str = "PLANNING"
    
class ToolsResponse(BaseModel):
    tools: List[Dict[str, Any]]

# API key dependency
async def verify_api_key(x_api_key: str = Header(None)):
    if os.environ.get("API_KEY_ENABLED", "false").lower() == "true":
        api_key = os.environ.get("API_KEY")
        if not api_key or x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

# Get current user dependency
from app.api.auth.router import get_current_user

# Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks, 
    current_user: dict = Depends(get_current_user)
):
    """Process a chat message with Manus agent"""
    try:
        # Get or create agent
        agent, session_id = await session_manager.get_agent(request.session_id)
        
        # Handle streaming if requested
        if request.stream:
            async def stream_generator():
                # Initialize streaming response
                response_parts = []
                
                # Set up streaming callback
                async def stream_callback(chunk):
                    response_parts.append(chunk)
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Process request with streaming
                await agent.run(request.prompt, stream_callback=stream_callback)
                
                # Send final message
                yield f"data: {json.dumps({'done': True, 'full_response': ''.join(response_parts)})}\n\n"
                yield "data: [DONE]\n\n"
            
            # Return streaming response
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        
        # Process request normally
        response = await agent.run(request.prompt)
        
        # Schedule cleanup for inactive sessions
        background_tasks.add_task(session_manager.cleanup_idle_sessions)
        
        return {
            "session_id": session_id,
            "response": response
        }
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flow")
async def execute_flow(
    request: FlowRequest, 
    current_user: dict = Depends(get_current_user)
):
    """Execute a flow-based task"""
    try:
        # Create agents dictionary
        agents = {
            "manus": await Manus.create(),
        }
        
        # Create flow
        flow_type = getattr(FlowType, request.flow_type, FlowType.PLANNING)
        flow = FlowFactory.create_flow(
            flow_type=flow_type,
            agents=agents,
        )
        
        # Execute flow with timeout
        result = await asyncio.wait_for(
            flow.execute(request.prompt),
            timeout=3600,  # 60 minute timeout
        )
        
        # Clean up agents
        for agent in agents.values():
            await agent.cleanup()
            
        return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timed out")
    except Exception as e:
        logger.error(f"Error executing flow: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tools", response_model=ToolsResponse)
async def list_tools(current_user: dict = Depends(get_current_user)):
    """List available tools"""
    try:
        # Create temporary agent to get tools
        agent = await Manus.create()
        
        # Get tools
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.to_param()["function"]["parameters"]
            }
            for tool in agent.available_tools.tools
        ]
        
        # Clean up
        await agent.cleanup()
        
        return {"tools": tools}
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(
    request: Request, 
    current_user: dict = Depends(get_current_user)
):
    """Upload a file to the workspace"""
    try:
        form = await request.form()
        file = form.get("file")
        
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # Create workspace directory if it doesn't exist
        os.makedirs(config.workspace_root, exist_ok=True)
        
        # Save file to workspace
        file_path = os.path.join(config.workspace_root, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        return {
            "filename": file.filename,
            "path": file_path,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files(current_user: dict = Depends(get_current_user)):
    """List files in the workspace"""
    try:
        # Create workspace directory if it doesn't exist
        os.makedirs(config.workspace_root, exist_ok=True)
        
        # Get list of files
        files = []
        for filename in os.listdir(config.workspace_root):
            file_path = os.path.join(config.workspace_root, filename)
            if os.path.isfile(file_path):
                files.append({
                    "name": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                })
                
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "config": {
            "llm_type": config.llm['default'].api_type if 'default' in config.llm else "unknown",
            "workspace": str(config.workspace_root),
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "config": {
            "llm_type": config.llm['default'].api_type if 'default' in config.llm else "unknown",
            "workspace": config.workspace_root,
            "tools_count": len(config.tool_config.enabled_tools)
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting OpenAgentFramework API")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
    
    # Log configuration with safe access
    try:
        logger.info(f"LLM Provider: {config.llm['default'].api_type if 'default' in config.llm else 'unknown'}")
        logger.info(f"Workspace: {config.workspace_root}")
    except Exception as e:
        logger.error(f"Error logging configuration: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down OpenAgentFramework API")
    await session_manager.cleanup_all()
