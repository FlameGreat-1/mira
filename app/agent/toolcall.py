import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection
from app.sandbox.client import SANDBOX_CLIENT


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None
    _stream_callback = None
    consecutive_no_tool_steps: int = Field(default=0)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None
    max_consecutive_no_tool_steps: int = Field(default=2)

    async def think(self) -> bool:
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )

        if self._stream_callback is not None and content:
            try:
                await self._stream_callback(content)
            except Exception as e:
                logger.warning(f"Stream callback error: {e}")

        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        if not tool_calls:
            self.consecutive_no_tool_steps += 1
            if self.consecutive_no_tool_steps >= self.max_consecutive_no_tool_steps:
                logger.info("🏁 Auto-terminating: Maximum consecutive steps without tools reached")
                self.state = AgentState.FINISHED
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                return False
        else:
            self.consecutive_no_tool_steps = 0

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True

            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            self._current_base64_image = None
            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            args = json.loads(command.function.arguments or "{}")
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            await self._handle_special_tool(name=name, result=result)

            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        return True

    def _is_special_tool(self, name: str) -> bool:
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None, stream_callback=None, **kwargs) -> str:
        try:
            self._stream_callback = stream_callback
            
            if self.state != AgentState.IDLE:
                raise RuntimeError(f"Cannot run agent from state: {self.state}")

            if request:
                self.update_memory("user", request)

            results: List[str] = []
            async with self.state_context(AgentState.RUNNING):
                while (
                    self.current_step < self.max_steps and self.state != AgentState.FINISHED
                ):
                    self.current_step += 1
                    logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                    
                    if self._stream_callback is not None:
                        try:
                            await self._stream_callback(f"Step {self.current_step}: ")
                        except Exception as e:
                            logger.warning(f"Stream callback error: {e}")
                    
                    step_result = await self.step()

                    if self.is_stuck():
                        self.handle_stuck_state()

                    results.append(f"Step {self.current_step}: {step_result}")
                    
                    if self._stream_callback is not None:
                        try:
                            await self._stream_callback(step_result)
                        except Exception as e:
                            logger.warning(f"Stream callback error: {e}")

                    if self.state == AgentState.FINISHED:
                        logger.info("🏁 Agent marked as FINISHED, terminating execution")
                        if self._stream_callback is not None:
                            try:
                                await self._stream_callback("\n🏁 Task completed successfully!")
                            except Exception as e:
                                logger.warning(f"Stream callback error: {e}")
                        break

                if self.current_step >= self.max_steps and self.state != AgentState.FINISHED:
                    self.current_step = 0
                    self.state = AgentState.IDLE
                    termination_msg = f"Terminated: Reached max steps ({self.max_steps})"
                    results.append(termination_msg)
                    if self._stream_callback is not None:
                        try:
                            await self._stream_callback(f"\n⏰ {termination_msg}")
                        except Exception as e:
                            logger.warning(f"Stream callback error: {e}")
                elif self.state == AgentState.FINISHED:
                    self.current_step = 0
                    self.state = AgentState.IDLE
                    
            await SANDBOX_CLIENT.cleanup()
            return "\n".join(results) if results else "No steps executed"
            
        finally:
            callback_to_clear = self._stream_callback
            self._stream_callback = None
            await self.cleanup()
