import math
import asyncio
import os
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from huggingface_hub import InferenceClient

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "deepseek-ai/DeepSeek-R1-0528",
]


class TokenCounter:
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        detail = image_item.get("detail", "medium")

        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        if detail == "high" or detail == "medium":
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        total_tokens = self.FORMAT_TOKENS

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS

            tokens += self.count_text(message.get("role", ""))

            if "content" in message:
                tokens += self.count_content(message["content"])

            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()
            elif self.api_type == "huggingface_deepseek":
                hf_token = os.environ.get("HF_TOKEN", self.api_key)
                self.hf_client = InferenceClient(token=hf_token)
                self.model_id = llm_config.model_id if hasattr(llm_config, "model_id") else "deepseek-ai/DeepSeek-R1-0528"
                logger.info(f"Initialized Hugging Face DeepSeek client with model: {self.model_id}")
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        formatted_messages = []

        for message in messages:
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                if supports_images and message.get("base64_image"):
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    del message["base64_image"]
                elif not supports_images and message.get("base64_image"):
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        try:
            supports_images = self.model in MULTIMODAL_MODELS

            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            input_tokens = self.count_message_tokens(messages)

            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                raise TokenLimitExceeded(error_message)

            if self.api_type == "huggingface_deepseek":
                formatted_messages = []
                
                if system_msgs:
                    for msg in system_msgs:
                        if isinstance(msg, Message):
                            formatted_messages.append({"role": "system", "content": msg.content})
                        elif isinstance(msg, dict) and "content" in msg:
                            formatted_messages.append({"role": "system", "content": msg["content"]})
                
                for msg in messages:
                    if isinstance(msg, Message):
                        formatted_messages.append({"role": msg.role, "content": msg.content})
                    elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
                
                params = {
                    "model": self.model_id,
                    "messages": formatted_messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature if temperature is not None else self.temperature
                }
                
                self.update_token_count(input_tokens)
                
                if not stream:
                    response = await asyncio.to_thread(self.hf_client.chat_completion, **params)
                    
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from Hugging Face")
                    
                    content = response.choices[0].message.content
                    
                    if hasattr(response, "usage"):
                        self.update_token_count(
                            response.usage.prompt_tokens, 
                            response.usage.completion_tokens
                        )
                    else:
                        completion_tokens = self.count_tokens(content)
                        self.update_token_count(0, completion_tokens)
                    
                    return content
                
                response_iterator = await asyncio.to_thread(
                    self.hf_client.chat_completion,
                    stream=True,
                    **params
                )
                
                collected_messages = []
                completion_text = ""
                
                for chunk in response_iterator:
                    chunk_message = ""
                    try:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            chunk_message = chunk.choices[0].delta.content or ""
                        elif hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'message'):
                            chunk_message = chunk.choices[0].message.content or ""
                    except (AttributeError, IndexError) as e:
                        logger.warning(f"Error parsing chunk: {e}")
                        continue
                    
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    logger.debug(f"Streaming chunk: {chunk_message}")
                
                full_response = "".join(collected_messages).strip()
                
                if not full_response:
                    raise ValueError("Empty response from streaming Hugging Face")
                
                completion_tokens = self.count_tokens(completion_text)
                logger.info(f"Estimated completion tokens for streaming response: {completion_tokens}")
                self.update_token_count(0, completion_tokens)
                
                return full_response
            else:
                params = {
                    "model": self.model,
                    "messages": messages,
                }

                if self.model in REASONING_MODELS:
                    params["max_completion_tokens"] = self.max_tokens
                else:
                    params["max_tokens"] = self.max_tokens
                    params["temperature"] = (
                        temperature if temperature is not None else self.temperature
                    )

                if not stream:
                    response = await self.client.chat.completions.create(
                        **params, stream=False
                    )

                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from LLM")

                    self.update_token_count(
                        response.usage.prompt_tokens, response.usage.completion_tokens
                    )

                    return response.choices[0].message.content

                self.update_token_count(input_tokens)

                response = await self.client.chat.completions.create(**params, stream=True)

                collected_messages = []
                completion_text = ""
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    logger.debug(f"Streaming chunk: {chunk_message}")

                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")

                completion_tokens = self.count_tokens(completion_text)
                logger.info(
                    f"Estimated completion tokens for streaming response: {completion_tokens}"
                )
                self.update_token_count(0, completion_tokens)

                return full_response

        except TokenLimitExceeded:
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        tools: List[dict],
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        temperature: Optional[float] = None,
    ) -> ChatCompletionMessage:
        try:
            supports_images = self.model in MULTIMODAL_MODELS

            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            input_tokens = self.count_message_tokens(messages)

            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                raise TokenLimitExceeded(error_message)

            if self.api_type == "huggingface_deepseek":
                formatted_messages = []
                
                if system_msgs:
                    for msg in system_msgs:
                        if isinstance(msg, Message):
                            formatted_messages.append({"role": "system", "content": msg.content})
                        elif isinstance(msg, dict) and "content" in msg:
                            formatted_messages.append({"role": "system", "content": msg["content"]})
                
                for msg in messages:
                    if isinstance(msg, Message):
                        formatted_messages.append({"role": msg.role, "content": msg.content})
                    elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
                
                self.update_token_count(input_tokens)
                
                response = await asyncio.to_thread(
                    self.hf_client.chat_completion,
                    model=self.model_id,
                    messages=formatted_messages,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature
                )
                
                if response and response.choices and response.choices[0].message:
                    from openai.types.chat import ChatCompletionMessage
                    
                    content = response.choices[0].message.content
                    tool_calls = response.choices[0].message.tool_calls
                    
                    completion_tokens = self.count_tokens(content or "")
                    self.update_token_count(0, completion_tokens)
                    
                    return ChatCompletionMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls
                    )
                else:
                    raise ValueError("Empty or invalid response from Hugging Face")
            else:
                params = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                }

                if tool_choice is not None and tool_choice != ToolChoice.AUTO:
                    params["tool_choice"] = (
                        tool_choice.value if hasattr(tool_choice, "value") else tool_choice
                    )

                if self.model in REASONING_MODELS:
                    params["max_completion_tokens"] = self.max_tokens
                else:
                    params["max_tokens"] = self.max_tokens
                    params["temperature"] = (
                        temperature if temperature is not None else self.temperature
                    )

                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message

        except TokenLimitExceeded:
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask_tool")
            raise
            
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        try:
            if not images:
                raise ValueError("No images provided")

            if self.api_type == "huggingface_deepseek":
                formatted_messages = []
                
                if system_msgs:
                    for msg in system_msgs:
                        if isinstance(msg, Message):
                            formatted_messages.append({"role": "system", "content": msg.content})
                        elif isinstance(msg, dict) and "content" in msg:
                            formatted_messages.append({"role": "system", "content": msg["content"]})
                
                for msg in messages:
                    if isinstance(msg, Message):
                        formatted_messages.append({"role": msg.role, "content": msg.content})
                    elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
                
                if images and formatted_messages:
                    last_user_msg_idx = -1
                    for i in range(len(formatted_messages) - 1, -1, -1):
                        if formatted_messages[i]["role"] == "user":
                            last_user_msg_idx = i
                            break
                    
                    if last_user_msg_idx >= 0:
                        if isinstance(formatted_messages[last_user_msg_idx]["content"], str):
                            text_content = formatted_messages[last_user_msg_idx]["content"]
                            formatted_messages[last_user_msg_idx]["content"] = [
                                {"type": "text", "text": text_content}
                            ]
                        elif not formatted_messages[last_user_msg_idx]["content"]:
                            formatted_messages[last_user_msg_idx]["content"] = []
                        
                        for img in images:
                            if isinstance(img, str):
                                formatted_messages[last_user_msg_idx]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": img}
                                })
                            elif isinstance(img, dict) and "url" in img:
                                formatted_messages[last_user_msg_idx]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": img["url"]}
                                })
                            elif isinstance(img, dict) and "base64" in img:
                                formatted_messages[last_user_msg_idx]["content"].append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}
                                })
                
                input_tokens = self.count_message_tokens(formatted_messages)
                
                if not self.check_token_limit(input_tokens):
                    error_message = self.get_limit_error_message(input_tokens)
                    raise TokenLimitExceeded(error_message)
                
                self.update_token_count(input_tokens)
                
                params = {
                    "model": self.model_id,
                    "messages": formatted_messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature if temperature is not None else self.temperature
                }
                
                if not stream:
                    response = await asyncio.to_thread(self.hf_client.chat_completion, **params)
                    
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from Hugging Face")
                    
                    content = response.choices[0].message.content
                    
                    if hasattr(response, "usage"):
                        self.update_token_count(
                            response.usage.prompt_tokens, 
                            response.usage.completion_tokens
                        )
                    else:
                        completion_tokens = self.count_tokens(content)
                        self.update_token_count(0, completion_tokens)
                    
                    return content
                
                response_iterator = await asyncio.to_thread(
                    self.hf_client.chat_completion,
                    stream=True,
                    **params
                )
                
                collected_messages = []
                completion_text = ""
                
                for chunk in response_iterator:
                    chunk_message = ""
                    try:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            chunk_message = chunk.choices[0].delta.content or ""
                        elif hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'message'):
                            chunk_message = chunk.choices[0].message.content or ""
                    except (AttributeError, IndexError) as e:
                        logger.warning(f"Error parsing chunk: {e}")
                        continue
                    
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    logger.debug(f"Streaming chunk: {chunk_message}")
                
                full_response = "".join(collected_messages).strip()
                
                if not full_response:
                    raise ValueError("Empty response from streaming Hugging Face")
                
                completion_tokens = self.count_tokens(completion_text)
                logger.info(f"Estimated completion tokens for streaming response: {completion_tokens}")
                self.update_token_count(0, completion_tokens)
                
                return full_response
            else:
                if self.model not in MULTIMODAL_MODELS:
                    raise ValueError(f"Model {self.model} does not support image inputs")

                if system_msgs:
                    system_msgs = self.format_messages(system_msgs, True)
                    messages = system_msgs + self.format_messages(messages, True)
                else:
                    messages = self.format_messages(messages, True)

                last_user_msg_idx = -1
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        last_user_msg_idx = i
                        break

                if last_user_msg_idx < 0:
                    raise ValueError("No user message found to attach images to")

                if isinstance(messages[last_user_msg_idx]["content"], str):
                    text_content = messages[last_user_msg_idx]["content"]
                    messages[last_user_msg_idx]["content"] = [
                        {"type": "text", "text": text_content}
                    ]
                elif not messages[last_user_msg_idx]["content"]:
                    messages[last_user_msg_idx]["content"] = []

                for img in images:
                    if isinstance(img, str):
                        messages[last_user_msg_idx]["content"].append(
                            {"type": "image_url", "image_url": {"url": img}}
                        )
                    elif isinstance(img, dict) and "url" in img:
                        messages[last_user_msg_idx]["content"].append(
                            {"type": "image_url", "image_url": {"url": img["url"]}}
                        )
                    elif isinstance(img, dict) and "base64" in img:
                        messages[last_user_msg_idx]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img['base64']}"
                                },
                            }
                        )

                input_tokens = self.count_message_tokens(messages)

                if not self.check_token_limit(input_tokens):
                    error_message = self.get_limit_error_message(input_tokens)
                    raise TokenLimitExceeded(error_message)

                params = {
                    "model": self.model,
                    "messages": messages,
                }

                if self.model in REASONING_MODELS:
                    params["max_completion_tokens"] = self.max_tokens
                else:
                    params["max_tokens"] = self.max_tokens
                    params["temperature"] = (
                        temperature if temperature is not None else self.temperature
                    )

                if not stream:
                    response = await self.client.chat.completions.create(
                        **params, stream=False
                    )

                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from LLM")

                    self.update_token_count(
                        response.usage.prompt_tokens, response.usage.completion_tokens
                    )

                    return response.choices[0].message.content

                self.update_token_count(input_tokens)

                response = await self.client.chat.completions.create(**params, stream=True)

                collected_messages = []
                completion_text = ""
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    logger.debug(f"Streaming chunk: {chunk_message}")

                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")

                completion_tokens = self.count_tokens(completion_text)
                logger.info(
                    f"Estimated completion tokens for streaming response: {completion_tokens}"
                )
                self.update_token_count(0, completion_tokens)

                return full_response

        except TokenLimitExceeded:
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask_with_images")
            raise
