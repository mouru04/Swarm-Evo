"""
Agent Pool 模块。

负责管理和创建 Agent 实例。
已重构以支持 LangGraph 架构。
"""

from typing import Dict, List, Any, Optional

from langchain_core.language_models import BaseChatModel

from core.agent.base_agent import BaseReActAgent, create_agent
from core.agent.prompt_manager import PromptManager
from core.agent.tools import get_tools
from utils.logger_system import log_msg


class AgentPool:
    """
    AgentPool - Agent 注册表。

    功能:
    - 保存所有 BaseReActAgent 实例
    - 支持从配置批量构建 Agent
    - 提供 get(name) 用于获取 Agent

    注意:
    - 不再负责执行 Agent（执行由 LangGraph 控制）
    - 不再负责 orchestration / run_async
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        初始化 AgentPool。

        参数:
            llm: LangChain Chat Model 实例。
        """
        self.agents: Dict[str, BaseReActAgent] = {}
        self.llm = llm

    def add(self, agent: BaseReActAgent) -> None:
        """
        添加 Agent 到池中。

        参数:
            agent: BaseReActAgent 实例。
        """
        self.agents[agent.name] = agent
        log_msg("INFO", f"Agent '{agent.name}' 注册完成")

    def get(self, name: str) -> BaseReActAgent:
        """
        获取指定名称的 Agent。

        参数:
            name: Agent 名称。

        返回:
            BaseReActAgent 实例。

        异常:
            KeyError: 如果 Agent 不存在。
        """
        if name not in self.agents:
            log_msg("ERROR", f"Agent '{name}' 不存在于 AgentPool")
            raise KeyError(f"Agent '{name}' not found in pool")
        return self.agents[name]

    def list_agents(self) -> List[str]:
        """
        列出所有已注册的 Agent 名称。

        返回:
            Agent 名称列表。
        """
        return list(self.agents.keys())

    @classmethod
    def from_configs(
        cls,
        agents_num: int,
        config: Dict[str, Any],
        llm: BaseChatModel,
    ) -> "AgentPool":
        """
        从配置批量创建 Agent 并注册。

        为 LangGraph 场景提供统一初始化入口。

        参数:
            agents_num: 要创建的 Agent 数量。
            config: Agent 配置字典，包含:
                - conda_env_name: Conda 环境名称
                - max_steps: 最大执行步数
            llm: LangChain Chat Model 实例。

        返回:
            AgentPool 实例。
        """
        pool = cls(llm=llm)

        conda_env_name = config.get("conda_env_name", "")
        max_steps = config.get("max_steps", 16)

        # 获取工具列表
        tools = get_tools(conda_env_name)
        log_msg("INFO", f"已加载 {len(tools)} 个工具: {[t.name for t in tools]}")

        for i in range(agents_num):
            agent = BaseReActAgent(
                name=f"Agent_{i}",
                llm=llm,
                tools=tools,
                prompt_manager=PromptManager(),
                max_steps=max_steps,
            )
            pool.add(agent)

        log_msg("INFO", f"AgentPool 创建完成，共 {agents_num} 个 Agent")
        return pool
