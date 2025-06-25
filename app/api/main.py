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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

class AgentSessionManager:
    def __init__(self, max_idle_time=1800):
        self.sessions = {}
        self.last_activity = {}
        self.max_idle_time = max_idle_time

    async def get_agent(self, session_id=None):
        if not session_id or session_id not in self.sessions:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = await Manus.create()

        self.last_activity[session_id] = time.time()
        return self.sessions[session_id], session_id

    async def cleanup_idle_sessions(self):
        current_time = time.time()
        for session_id, last_active in list(self.last_activity.items()):
            if current_time - last_active > self.max_idle_time:
                await self.cleanup_session(session_id)

    async def cleanup_session(self, session_id):
        if session_id in self.sessions:
            await self.sessions[session_id].cleanup()
            del self.sessions[session_id]
            del self.last_activity[session_id]

    async def cleanup_all(self):
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)

session_manager = AgentSessionManager()

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

async def verify_api_key(x_api_key: str = Header(None)):
    if os.environ.get("API_KEY_ENABLED", "false").lower() == "true":
        api_key = os.environ.get("API_KEY")
        if not api_key or x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

from app.api.auth.router import get_current_user

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    try:
        agent, session_id = await session_manager.get_agent(request.session_id)

        if request.stream:
            async def stream_generator():
                stream_queue = asyncio.Queue(maxsize=100)
                response_parts = []
                agent_task = None
                
                try:
                    def stream_callback(chunk):
                        if chunk and len(chunk.strip()) > 0:
                            response_parts.append(chunk)
                            try:
                                stream_queue.put_nowait(chunk)
                            except asyncio.QueueFull:
                                logger.warning("Stream queue full, dropping chunk")
                    
                    agent_task = asyncio.create_task(
                        asyncio.wait_for(
                            agent.run(request.prompt, stream_callback=stream_callback),
                            timeout=300
                        )
                    )
                    
                    last_activity = time.time()
                    
                    while not agent_task.done():
                        try:
                            chunk = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                            last_activity = time.time()
                            yield f"data: {json.dumps({'chunk': chunk, 'timestamp': time.time()})}\n\n"
                            
                        except asyncio.TimeoutError:
                            if time.time() - last_activity > 30:
                                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                                last_activity = time.time()
                            continue
                        except Exception as e:
                            logger.error(f"Streaming error: {e}")
                            yield f"data: {json.dumps({'error': 'Streaming interrupted'})}\n\n"
                            break
                    
                    try:
                        await agent_task
                        yield f"data: {json.dumps({
                            'done': True, 
                            'full_response': ''.join(response_parts),
                            'chunks_count': len(response_parts),
                            'session_id': session_id
                        })}\n\n"
                        
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'error': 'Request timeout', 'partial_response': ''.join(response_parts)})}\n\n"
                    except Exception as e:
                        logger.error(f"Agent execution error: {e}")
                        yield f"data: {json.dumps({'error': 'Processing failed', 'details': str(e)})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Stream generator error: {e}")
                    yield f"data: {json.dumps({'error': 'Stream failed', 'details': str(e)})}\n\n"
                    
                finally:
                    if agent_task and not agent_task.done():
                        agent_task.cancel()
                        try:
                            await agent_task
                        except asyncio.CancelledError:
                            pass
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        response = await agent.run(request.prompt)
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
    try:
        agents = {
            "manus": await Manus.create(),
        }

        flow_type = getattr(FlowType, request.flow_type, FlowType.PLANNING)
        flow = FlowFactory.create_flow(
            flow_type=flow_type,
            agents=agents,
        )

        result = await asyncio.wait_for(
            flow.execute(request.prompt),
            timeout=3600,
        )

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
    try:
        agent = await Manus.create()

        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.to_param()["function"]["parameters"]
            }
            for tool in agent.available_tools.tools
        ]

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
    try:
        form = await request.form()
        file = form.get("file")

        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        os.makedirs(config.workspace_root, exist_ok=True)

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
    try:
        os.makedirs(config.workspace_root, exist_ok=True)

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
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "llm_type": config.llm['default'].api_type if 'default' in config.llm else "unknown",
            "workspace": config.workspace_root,
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Starting OpenAgentFramework API")

    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

    try:
        logger.info(f"LLM Provider: {config.llm['default'].api_type if 'default' in config.llm else 'unknown'}")
        logger.info(f"Workspace: {config.workspace_root}")
    except Exception as e:
        logger.error(f"Error logging configuration: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down OpenAgentFramework API")
    await session_manager.cleanup_all()
