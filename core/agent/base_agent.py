import re
import json
import asyncio
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from openai import OpenAI

from core.agent.prompt_manager import PromptManager #
from core.agent.prompt_manager import PromptContext
from utils.logger_system import LoggerSystem, log_msg, log_json 
from utils.json_utils import render_history_json, parse_json_output




class LLMResponse:
    def __init__(self, raw_output, normalized_output, is_final, final_answer, task, action, tool_input):
        self.raw_output = raw_output
        self.normalized_output = normalized_output
        self.is_final = is_final
        self.final_answer = final_answer
        self.task = task
        self.action = action
        self.tool_input = tool_input

class AgentSessionResult:
    def __init__(self, final_answer, history, success):
        self.final_answer = final_answer
        self.history = history
        self.success = success

class BaseReActAgent:

    def __init__(
        self,
        name: str,
        model: str,
        tools: List[BaseTool],
        prompt_manager: PromptManager,
        max_steps: int,
        llm_client: Optional[OpenAI],
        # logger: Optional[LoggerSystem], # Removed
        user_prompt_template: str = "explore_user_prompt.j2",
        accepted_return_types: List[str] = ["final", "selection", "review","evaluation"] 
    ):
        """
        Args:
            accepted_return_types: List of "type" field values that are considered valid final outputs.
                                   Default is ["final"]. 
                                   For Select, pass ["selection"]. 
                                   For Evaluate, pass ["evaluation"].
                                   "action" is always handled internally as a tool step.
        """
        self.MAX_LLM_RETRY = 12
        self.RETRY_BASE_DELAY = 2.0
        self.MAX_TOOL_RETRY = 5
        
        self.name = name
        self.model = model
        self.max_steps = max_steps
        self.llm_client = llm_client
        # self.logger = logger # Removed
        self.user_prompt_template = user_prompt_template
        self.prompt_manager = prompt_manager
        self.accepted_return_types = accepted_return_types

        # 工具字典
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

        # 系统 + 用户提示词模板
        # Note: prompt_manager will handle the actual content, but we define the strict structure here
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_instruction}"),
            ("user", "{user_instruction}")
        ])
    
    async def llm(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        调用 LLM 并解析 JSON 输出：
        - { "type": "action", ... }
        - { "type": "final", ... }
        - { "type": "selection", ... }
        - { "type": "evaluation", ... }
        """
        for attempt in range(1, self.MAX_LLM_RETRY + 1):
            try:
                log_msg("INFO", f"Agent '{self.name}' 尝试第 {attempt} 次调用 LLM, 使用模型 {self.model}")

                # 获取 LLM 响应
                resp = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                choices = getattr(resp, "choices", None)
                if not choices:
                    log_msg("ERROR", f"Agent '{self.name}' 调用 LLM 失败，LLM 响应格式错误")
                    log_msg("ERROR", "LLM response missing choices")
                
                raw_text = getattr(choices[0].message, "content", None)
                if raw_text is None:
                     # Fallback for some clients returning dict
                    raw_text = choices[0].message.get("content")

                if raw_text is None:
                    log_msg("ERROR", f"Agent '{self.name}' 调用 LLM 失败，LLM 响应格式错误")
                    log_msg("ERROR", "LLM response missing content")
                if not isinstance(raw_text, str):
                    log_msg("ERROR", f"Agent '{self.name}' 调用 LLM 失败，LLM 响应格式错误")
                    log_msg("ERROR", "LLM response content type error")

                # JSON 解析
                parsed_json = parse_json_output(raw_text)
                
                # 检查 type 字段
                msg_type = parsed_json.get("type")
                if not msg_type:
                    log_msg("ERROR", "JSON output missing 'type' field")

                # 1) ReAct Action
                if msg_type == "action":
                    task = parsed_json.get("task")
                    tool_name = parsed_json.get("tool")
                    tool_input = parsed_json.get("input")
                    
                    if not tool_name or tool_input is None:
                         log_msg("ERROR", "ActionStep missing tool or input")
                    
                    return LLMResponse(
                        raw_output=raw_text,
                        normalized_output=json.dumps(parsed_json),
                        is_final=False,
                        final_answer=None,
                        task=task,
                        action=tool_name,
                        tool_input=tool_input
                    )
                
                # 2) Final Result (Standard 'final' or custom types like 'selection', 'evaluation')
                # If the type is known to be a "Return Action", treat it as final answer
                if msg_type == "final" or msg_type in self.accepted_return_types:
                    return LLMResponse(
                        raw_output=raw_text,
                        normalized_output=json.dumps(parsed_json),
                        is_final=True,
                        final_answer=parsed_json, # For JSON mode, final_answer IS the dict object
                        task=None,
                        action=None,
                        tool_input=None
                    )
                
                # Unknown type
                log_msg("ERROR", f"Unknown message type: {msg_type}")

            except Exception as exc:
                if attempt < self.MAX_LLM_RETRY:
                    delay = min(self.RETRY_BASE_DELAY * attempt, 10.0)
                    log_msg("WARNING", f"Agent '{self.name}' 调用 LLM 失败/解析错误: {exc} | 第 {attempt} 次重试")
                    await asyncio.sleep(delay)
                    continue
                else:
                    log_msg("ERROR", f"Agent '{self.name}' 调用 LLM 失败，重试次数已达上限")
                    log_msg("ERROR", f"LLM consecutive requests failed: {exc}")
        
    async def _react_loop(self, instruction_text: str, prompt_context: PromptContext) -> AgentSessionResult:
        
        history_records: List[Dict[str, Any]] = []
        consecutive_tool_errors: int = 0
        step: int = 0

        while step < self.max_steps:
            log_msg("INFO", f"step: {step}")

            # 平铺历史（JSON String）
            history = render_history_json(history_records)
            
            user_prompt = self.prompt_manager.build_user_prompt(prompt_context, history)
            system_instruction = self.prompt_manager.build_system_prompt(prompt_context)

            # 创建 OpenAI 兼容消息
            formatted_messages = self.prompt.format_messages(
                system_instruction=system_instruction,
                user_instruction=user_prompt,
            )
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            chat_messages = [
                {"role": role_map.get(message.type, message.type), "content": message.content}
                for message in formatted_messages
            ]

            # 第二阶段：调用 LLM
            response = await self.llm(chat_messages)

            # 第三阶段：Final Answer / Custom Return
            if response.is_final:
                if response.final_answer is None:
                    # Should not happen logic wise if is_final is set correctly
                    log_msg("ERROR", "FinalAnswer missing content")

                # Log completion
                log_json({
                    "step": step,
                    "task": "Task Complete",
                    "final_answer": response.final_answer,
                    "action": None,
                    "tool_input": None,
                    "observation": None,
                })

                return AgentSessionResult(
                    final_answer=response.final_answer,
                    history=history_records,
                    success=True
                )

            # ---------------------------
            # ActionStep 的工具执行
            # ---------------------------
            if response.action in self.tools:
                tool = self.tools[response.action]

                try:
                    if response.tool_input is None:
                         log_msg("ERROR", "Tool input missing")
                    
                    # JSON mode: tool_input is already a dict (or primitive), direct pass
                    observation = tool.run(response.tool_input)

                except Exception as e:
                    observation = f"Tool execution failed: {e}"
                    log_msg("ERROR", observation)

            else:
                observation = f"Unknown tool: {response.action}"
                log_msg("ERROR", observation)

            # 工具错误处理
            if isinstance(observation, str) and (
                observation.startswith("Unknown tool") or
                observation.startswith("Tool execution failed")
            ):
                consecutive_tool_errors += 1
                if consecutive_tool_errors > self.MAX_TOOL_RETRY:
                    log_msg("ERROR", "Consecutive tool failures, terminating")
                    log_msg("ERROR", f"Consecutive tool failures, terminating: {observation}")
                
                 # Log error
                log_json({
                    "step": step,
                    "task": response.task,
                    "final_answer": None,
                    "action": response.action,
                    "tool_input": response.tool_input,
                    "observation": f"Error: {observation}",
                })
                
                # Append to history
                history_records.append({
                    "step": step,
                    "task": response.task,
                    "tool": response.action, # Standardize naming in history
                    "input": response.tool_input,
                    "observation": f"Error: {observation}",
                    "type": "action"
                })
                step += 1
                continue

            # 正常观察
            consecutive_tool_errors = 0

            log_json({
                 "step": step,
                 "task": response.task,
                 "final_answer": None,
                 "action": response.action,
                 "tool_input": response.tool_input,
                 "observation": str(observation),
            })

            history_records.append({
                "step": step,
                "task": response.task,
                "tool": response.action,
                "input": response.tool_input,
                "observation": str(observation), # Observation might be object or string
                "type": "action"
            })
            step += 1

        # 超过最大步数
        log_msg("WARNING", "达到最大 ReAct 步数，任务未完成")
        return AgentSessionResult(
            final_answer={"error": "Max steps reached", "success": False},
            history=history_records,
            success=False
        )
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph 节点入口
        """

        instruction_text = state["task_description"]
        prompt_context = state["prompt_context"]

        # 调用内部 ReAct 推理循环
        session = await self._react_loop(instruction_text, prompt_context)

        return {
            "agent_output": session.final_answer,
            "agent_history": session.history,
            "agent_success": session.success,
            "agent_name": self.name,
            "raw_session": session,
        }




