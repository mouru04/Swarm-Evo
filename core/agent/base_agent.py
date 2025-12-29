"""
基于 LangGraph 的 Agent 模块。

本模块使用 LangGraph 的 StateGraph 构建 Agent，取代原有的 while 循环实现。
核心优势：
- 使用 LLM 的 Native Tool Calling，消除 JSON 解析错误
- 图结构清晰可视化，易于调试和扩展
- 状态可持久化，支持中断和恢复
"""

from typing import Dict, List, Any, Optional, Annotated, Sequence, TypedDict, Literal, Union
import operator

from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from core.agent.prompt_manager import PromptManager, PromptContext
from core.agent.tools import get_tools
from utils.logger_system import log_msg, log_json
from utils.json_utils import parse_json_output


# ============================================================================
# 状态定义
# ============================================================================
class AgentState(TypedDict):
    """
    Agent 状态定义。
    
    使用 TypedDict 定义强类型状态，LangGraph 会自动处理状态更新。
    """
    # 消息历史列表，使用 add operator 实现追加
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 当前步数
    step_count: int
    # 最大步数限制
    max_steps: int
    # Agent 名称
    agent_name: str
    # 任务是否成功完成
    success: bool
    # 最终答案 (如果有)
    final_answer: Optional[Dict[str, Any]]


# ============================================================================
# 结果类定义 (兼容性)
# ============================================================================
class AgentSessionResult:
    """Agent 执行结果，保持与旧代码兼容。"""
    
    def __init__(
        self, 
        final_answer: Optional[Dict[str, Any]], 
        history: List[Dict[str, Any]], 
        success: bool
    ):
        self.final_answer = final_answer
        self.history = history
        self.success = success


# ============================================================================
# Agent 图构建器
# ============================================================================
class AgentGraphBuilder:
    """
    Agent 图构建器。
    
    负责创建和编译 LangGraph StateGraph。
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: List[BaseTool],
        max_steps: int = 16,
    ):
        """
        初始化 Agent 图构建器。

        参数:
            name: Agent 名称。
            llm: LangChain Chat Model (已绑定工具)。
            tools: 工具列表。
            max_steps: 最大执行步数。
        """
        self.name = name
        self.tools = tools
        self.max_steps = max_steps
        
        # 绑定工具到 LLM
        self.llm_with_tools = llm.bind_tools(tools)
        
        # 工具节点
        self.tool_node = ToolNode(tools)

    def _should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """
        决定是否继续执行。

        逻辑:
        1. 如果达到最大步数，结束
        2. 如果 LLM 返回了 tool_calls，继续执行工具
        3. 否则，结束
        """
        messages = state["messages"]
        step_count = state["step_count"]
        max_steps = state["max_steps"]

        # 检查步数限制
        if step_count >= max_steps:
            log_msg("WARNING", f"Agent '{state['agent_name']}' 达到最大步数 {max_steps}")
            return "end"

        # 获取最后一条消息
        last_message = messages[-1] if messages else None
        
        if last_message is None:
            return "end"

        # 检查是否有 tool_calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "end"

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """
        调用 LLM 节点。

        参数:
            state: 当前状态。

        返回:
            状态更新字典。
        """
        messages = state["messages"]
        step_count = state["step_count"]
        agent_name = state["agent_name"]

        try:
            response = self.llm_with_tools.invoke(messages)
            
            # 记录响应信息
            if isinstance(response, AIMessage):
                # 准备 JSON 日志数据
                json_data = {
                    "agent_name": agent_name,
                    "step_count": step_count,
                    "response": {
                        "content": response.content,
                        "tool_calls": response.tool_calls,
                        "response_metadata": response.response_metadata
                    }
                }
                log_json(json_data)

                if response.tool_calls:
                    # 获取工具名称和参数用于日志
                    tool_calls_info = [f"{tc['name']}" for tc in response.tool_calls]
                    log_msg("INFO", f"Agent '{agent_name}' Step {step_count}: 调用 LLM，请求工具 {tool_calls_info} 成功")
                else:
                    log_msg("INFO", f"Agent '{agent_name}' Step {step_count}: 调用 LLM，返回最终回复 成功")
            
            return {
                "messages": [response],
                "step_count": step_count + 1,
            }

        except Exception as e:
            msg = f"Agent '{agent_name}' Step {step_count}: 调用 LLM 失败: {e}"
            log_msg("ERROR", msg)
            log_json({
                "agent_name": agent_name,
                "step_count": step_count,
                "error": str(e)
            })
            # 返回错误消息
            error_msg = AIMessage(content=f"LLM 调用失败: {e}")
            return {
                "messages": [error_msg],
                "step_count": step_count + 1,
            }

    def _call_tools(self, state: AgentState) -> Dict[str, Any]:
        """
        调用工具节点。

        直接从 AIMessage.tool_calls 中提取工具调用并执行。
        """
        agent_name = state["agent_name"]
        messages = state["messages"]
        
        # 获取最后一条消息（应该是 AIMessage with tool_calls）
        last_message = messages[-1] if messages else None
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            log_msg("WARNING", f"Agent '{agent_name}' 无工具调用")
            return {"messages": []}
        
        # 构建工具字典
        tool_map = {tool.name: tool for tool in self.tools}
        
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            
            # log_msg("INFO", f"Agent '{agent_name}' 执行工具: {tool_name}")
            
            if tool_name not in tool_map:
                error_content = f"未知工具: {tool_name}"
                log_msg("ERROR", error_content)
                tool_messages.append(ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call_id
                ))
                continue
            
            try:
                tool = tool_map[tool_name]
                # 直接调用工具
                result = tool.invoke(tool_args)
                # log_msg("INFO", f"Agent '{agent_name}' 工具 {tool_name} 执行成功")
                tool_messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                ))
            except Exception as e:
                error_content = f"工具执行失败: {e}"
                log_msg("ERROR", f"Agent '{agent_name}' {error_content}")
                tool_messages.append(ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call_id
                ))
        
        return {"messages": tool_messages}

    def build(self) -> StateGraph:
        """
        构建并返回编译后的图。

        返回:
            编译后的 StateGraph。
        """
        # 创建图
        graph = StateGraph(AgentState)

        # 添加节点
        graph.add_node("agent", self._call_model)
        graph.add_node("tools", self._call_tools)

        # 设置入口点
        graph.set_entry_point("agent")

        # 添加条件边
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )

        # 工具执行后返回 agent
        graph.add_edge("tools", "agent")

        # 编译
        return graph.compile()


# ============================================================================
# Agent 包装器 (兼容旧接口)
# ============================================================================
class BaseReActAgent:
    """
    基于 LangGraph 的 ReAct Agent。

    保持与旧代码兼容的接口，同时使用新的图执行引擎。
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: List[BaseTool],
        prompt_manager: PromptManager,
        max_steps: int = 16,
        accepted_return_types: Optional[List[str]] = None,
    ):
        """
        初始化 Agent。

        参数:
            name: Agent 名称。
            llm: LangChain Chat Model。
            tools: 工具列表。
            prompt_manager: 提示词管理器。
            max_steps: 最大执行步数。
            accepted_return_types: 接受的返回类型 (兼容性参数，新架构中不再需要)。
        """
        self.name = name
        self.llm = llm
        self.tools = tools
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps
        self.accepted_return_types = accepted_return_types or ["final"]

        # 构建图
        builder = AgentGraphBuilder(
            name=name,
            llm=llm,
            tools=tools,
            max_steps=max_steps,
        )
        self.graph = builder.build()

    async def run(
        self, 
        task_instruction: str, 
        prompt_context: PromptContext
    ) -> AgentSessionResult:
        """
        执行 Agent 任务。

        参数:
            task_instruction: 任务指令。
            prompt_context: 提示词上下文。

        返回:
            AgentSessionResult 执行结果。
        """
        log_msg("INFO", f"Agent '{self.name}' 开始执行任务")

        # 构建初始消息
        initial_messages = self.prompt_manager.build_initial_messages(
            context=prompt_context,
            task_instruction=task_instruction
        )

        # 初始状态
        initial_state: AgentState = {
            "messages": initial_messages,
            "step_count": 0,
            "max_steps": self.max_steps,
            "agent_name": self.name,
            "success": False,
            "final_answer": None,
        }

        try:
            # 执行图
            # 设置 recursion_limit 为 max_steps * 3，确保能覆盖 Agent->Tools->Agent 循环
            final_state = await self.graph.ainvoke(
                initial_state, 
                config={"recursion_limit": self.max_steps * 3 + 2}
            )

            # 提取结果
            messages = final_state.get("messages", [])
            step_count = final_state.get("step_count", 0)

            # 从最后一条 AI 消息中提取答案
            final_answer = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    content = msg.content
                    # 尝试解析 JSON (兼容 Select/Review 等需要结构化输出的任务)
                    try:
                        # 对于非 JSON 任务 (Merge/Explore)，解析失败是正常的，因此抑制错误日志
                        parsed = parse_json_output(content, suppress_error_log=True)
                        if parsed and isinstance(parsed, dict):
                            final_answer = parsed
                        else:
                            final_answer = {"content": content}
                    except Exception:
                        final_answer = {"content": content}
                    break

            # 构建历史记录 (兼容旧格式)
            history = self._extract_history(messages)

            # 如果没有找到 final_answer (例如达到最大步数)，构造默认错误返回
            if final_answer is None:
                final_answer = {
                    "content": "Error: Agent failed to produce a final answer (likely max steps reached).",
                    "error": "no_final_answer",
                    "step_count": step_count
                }

            success = final_answer is not None and step_count < self.max_steps

            log_msg("INFO", f"Agent '{self.name}' 任务完成, 步数: {step_count}, 成功: {success}")

            return AgentSessionResult(
                final_answer=final_answer,
                history=history,
                success=success
            )

        except Exception as e:
            log_msg("ERROR", f"Agent '{self.name}' 执行失败: {e}")
            return AgentSessionResult(
                final_answer={"error": str(e)},
                history=[],
                success=False
            )

    def _extract_history(self, messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        """
        从消息列表中提取历史记录。

        转换为旧格式以保持兼容性。
        """
        history = []
        step = 0

        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        history.append({
                            "step": step,
                            "type": "action",
                            "tool": tc["name"],
                            "input": tc["args"],
                            "task": "Tool Call",
                        })
                        step += 1
            elif isinstance(msg, ToolMessage):
                # 找到对应的历史记录并添加 observation
                if history and history[-1].get("type") == "action":
                    history[-1]["observation"] = msg.content

        return history

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph 节点入口 (兼容旧接口)。

        参数:
            state: 输入状态字典。

        返回:
            输出状态字典。
        """
        task_instruction = state.get("task_description", "")
        prompt_context = state.get("prompt_context")

        if prompt_context is None:
            log_msg("ERROR", "prompt_context not provided in state")
            return {
                "agent_output": {"error": "prompt_context not provided"},
                "agent_history": [],
                "agent_success": False,
                "agent_name": self.name,
            }

        session = await self.run(task_instruction, prompt_context)

        return {
            "agent_output": session.final_answer,
            "agent_history": session.history,
            "agent_success": session.success,
            "agent_name": self.name,
            "raw_session": session,
        }


# ============================================================================
# 工厂函数
# ============================================================================
def create_agent(
    name: str,
    llm: BaseChatModel,
    conda_env_name: str,
    prompt_manager: Optional[PromptManager] = None,
    max_steps: int = 16,
) -> BaseReActAgent:
    """
    创建 Agent 实例。

    参数:
        name: Agent 名称。
        llm: LangChain Chat Model。
        conda_env_name: Conda 环境名称。
        prompt_manager: 提示词管理器 (可选)。
        max_steps: 最大执行步数。

    返回:
        BaseReActAgent 实例。
    """
    tools = get_tools(conda_env_name)
    pm = prompt_manager or PromptManager()

    return BaseReActAgent(
        name=name,
        llm=llm,
        tools=tools,
        prompt_manager=pm,
        max_steps=max_steps,
    )
