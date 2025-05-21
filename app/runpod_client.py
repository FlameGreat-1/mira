"""
RunPod API Client for OpenAgentFramework
Provides compatibility layer between OpenAI-style API and RunPod API
"""

import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.exceptions import LLMError
from app.logger import logger


class AsyncRunPodClient:
    """
    Client for interacting with RunPod-hosted models
    
    Provides an interface compatible with OpenAI's AsyncClient
    """
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str,
        timeout: int = 120,
        retry_count: int = 3
    ):
        """
        Initialize RunPod client
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for RunPod API
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_count = retry_count
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Create namespaces to match OpenAI client structure
        self.chat = ChatCompletions(self)
        
    async def _make_request(self, endpoint: str, payload: Dict) -> Dict:
        """
        Make HTTP request to RunPod API
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            
        Returns:
            API response as dictionary
            
        Raises:
            LLMError: If request fails
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"RunPod API error: {response.status} - {error_text}")
                    
                    return await response.json()
        except aiohttp.ClientError as e:
            raise LLMError(f"RunPod API request failed: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMError(f"RunPod API request timed out after {self.timeout.total} seconds")
    
    async def _make_streaming_request(self, endpoint: str, payload: Dict) -> AsyncIterator[bytes]:
        """
        Make streaming HTTP request to RunPod API
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            
        Returns:
            Async iterator of response chunks
            
        Raises:
            LLMError: If request fails
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Add streaming flag to payload
                payload["stream"] = True
                
                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"RunPod API streaming error: {response.status} - {error_text}")
                    
                    # Process the stream
                    async for chunk in response.content:
                        if chunk:
                            yield chunk
        except aiohttp.ClientError as e:
            raise LLMError(f"RunPod API streaming request failed: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMError(f"RunPod API streaming request timed out after {self.timeout.total} seconds")


class ChatCompletions:
    """
    Chat completions API namespace
    
    Provides methods compatible with OpenAI's chat completions API
    """
    
    def __init__(self, client: AsyncRunPodClient):
        """
        Initialize chat completions namespace
        
        Args:
            client: RunPod client instance
        """
        self.client = client
    
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(LLMError)
    )
    async def create(self, **kwargs) -> Union[Dict, AsyncIterator[Dict]]:
        """
        Create a chat completion
        
        Args:
            **kwargs: Parameters for chat completion
                - messages: List of message dictionaries
                - model: Model name (ignored, uses RunPod model)
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - stream: Whether to stream the response
                
        Returns:
            Chat completion response or stream
            
        Raises:
            LLMError: If request fails
        """
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        
        # Extract system prompt and user prompt
        system_prompt = None
        user_prompt = None
        conversation_history = []
        
        # Process all messages to build conversation history
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                # Keep track of all user messages for context
                conversation_history.append({"role": "user", "content": content})
                user_prompt = content  # Last user message becomes the primary prompt
            elif role == "assistant":
                # Keep track of assistant responses for context
                conversation_history.append({"role": "assistant", "content": content})
        
        # Prepare request payload
        payload = {
            "prompt": user_prompt or "",
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096)
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system_prompt"] = system_prompt
        
        # Add conversation history if there's more than just the current prompt
        if len(conversation_history) > 1:
            # Remove the last user message as it's already in the prompt
            history = conversation_history[:-1]
            payload["conversation_history"] = history
            
        # Add image handling if present in messages
        for message in messages:
            if message.get("role") == "user" and isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            # Handle base64 images
                            payload["image_base64"] = image_url.split(",", 1)[1]
                        else:
                            # Handle URL images
                            payload["image_url"] = image_url
        
        if stream:
            # Handle streaming response
            return self._handle_streaming_response(payload)
        
        # Make request to generate endpoint for non-streaming response
        response = await self.client._make_request("api/generate", payload)
        
        # Convert to OpenAI-compatible format
        return self._convert_to_openai_response(response)
    
    async def _handle_streaming_response(self, payload: Dict) -> AsyncIterator[Dict]:
        """
        Handle streaming response from RunPod API
        
        Args:
            payload: Request payload
            
        Returns:
            Async iterator of OpenAI-compatible chunks
            
        Raises:
            LLMError: If streaming fails
        """
        # Set up streaming endpoint
        streaming_endpoint = "api/generate/stream"
        
        # Create a unique ID for this streaming session
        stream_id = f"runpod-stream-{int(time.time())}"
        
        # Initialize accumulated text for token counting
        accumulated_text = ""
        
        try:
            async for chunk in self.client._make_streaming_request(streaming_endpoint, payload):
                # Parse the chunk as JSON
                try:
                    chunk_data = json.loads(chunk)
                    
                    # Extract the text from the chunk
                    chunk_text = chunk_data.get("text", "")
                    accumulated_text += chunk_text
                    
                    # Create OpenAI-compatible chunk
                    openai_chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": chunk_data.get("model", "runpod-model"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk_text
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    # Check if this is the final chunk
                    if chunk_data.get("finish_reason"):
                        openai_chunk["choices"][0]["finish_reason"] = chunk_data["finish_reason"]
                    
                    yield openai_chunk
                    
                except json.JSONDecodeError:
                    # If the chunk is not valid JSON, try to extract text
                    chunk_text = chunk.decode("utf-8", errors="replace")
                    accumulated_text += chunk_text
                    
                    # Create a simple chunk with the text
                    yield {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "runpod-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": chunk_text
                                },
                                "finish_reason": None
                            }
                        ]
                    }
            
            # Send a final chunk to indicate completion
            yield {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "runpod-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            
            # Log token usage estimate
            try:
                # Estimate token count using tiktoken if available
                import tiktoken
                tokenizer = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = len(tokenizer.encode(payload.get("prompt", "")))
                completion_tokens = len(tokenizer.encode(accumulated_text))
                
                logger.info(f"Estimated token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")
            except ImportError:
                # If tiktoken is not available, log a message
                logger.info("Token counting not available (tiktoken not installed)")
                
        except Exception as e:
            # Log the error and raise it
            logger.error(f"Error in streaming response: {str(e)}")
            raise LLMError(f"Streaming error: {str(e)}")
    
    def _convert_to_openai_response(self, runpod_response: Dict) -> Dict:
        """
        Convert RunPod response to OpenAI-compatible format
        
        Args:
            runpod_response: Response from RunPod API
            
        Returns:
            OpenAI-compatible response
        """
        # Extract response text
        response_text = runpod_response.get("text", "")
        
        # Extract tool calls if present
        tool_calls = []
        if "tool_calls" in runpod_response:
            for tool_call in runpod_response["tool_calls"]:
                formatted_tool_call = {
                    "id": tool_call.get("id", f"call_{int(time.time())}"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": tool_call.get("arguments", "{}")
                    }
                }
                tool_calls.append(formatted_tool_call)
        
        # Create message with appropriate content
        message = {
            "role": "assistant",
            "content": response_text
        }
        
        # Add tool calls if present
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        # Create OpenAI-compatible response
        response = {
            "id": runpod_response.get("job_id", f"runpod-{int(time.time())}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": runpod_response.get("model", "runpod-model"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": runpod_response.get("finish_reason", "stop")
                }
            ],
            "usage": {
                "prompt_tokens": runpod_response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": runpod_response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": runpod_response.get("usage", {}).get("total_tokens", 0)
            }
        }
        
        # If token counts are not provided, estimate them
        if response["usage"]["prompt_tokens"] == 0 and response["usage"]["completion_tokens"] == 0:
            try:
                # Estimate token count using tiktoken if available
                import tiktoken
                tokenizer = tiktoken.get_encoding("cl100k_base")
                prompt = runpod_response.get("prompt", "")
                prompt_tokens = len(tokenizer.encode(prompt)) if prompt else 0
                completion_tokens = len(tokenizer.encode(response_text))
                total_tokens = prompt_tokens + completion_tokens
                
                response["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            except ImportError:
                # If tiktoken is not available, use character-based estimation
                prompt = runpod_response.get("prompt", "")
                prompt_tokens = len(prompt) // 4 if prompt else 0
                completion_tokens = len(response_text) // 4
                total_tokens = prompt_tokens + completion_tokens
                
                response["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
        
        return response
