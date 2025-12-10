class AgentPool:
    """
    AgentPool (Registry Version for LangGraph):

    ✔ 保存所有 BaseReActAgent 实例
    ✔ 支持从配置构建多个 Agent
    ✔ 提供 get(name) 用于 LangGraph Node 调用

    ✘ 不再负责执行 agent（执行由 LangGraph 控制）
    ✘ 不再负责 orchestration / run_async
    """

    def __init__(self, llm_client=None, logger=None):
        self.agents: Dict[str, BaseReActAgent] = {}
        self.llm_client = llm_client
        self.logger = logger

    # -----------------------------------
    # 添加与获取
    # -----------------------------------
    def add(self, agent: BaseReActAgent):
        self.agents[agent.name] = agent
        if self.logger:
            self.logger.text_log("INFO", f"Agent '{agent.name}' 注册完成")

    def get(self, name: str) -> BaseReActAgent:
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' 不存在于 AgentPool")
        return self.agents[name]

    # -----------------------------------
    # 从 config 批量构建 Agent
    # -----------------------------------
    @classmethod
    def from_configs(cls, agents_num: int, config: Dict, llm_client, logger=None):
        """
        用于批量创建 Agent 并注册。
        为 LangGraph 场景提供统一初始化入口。
        """
        pool = cls(llm_client=llm_client, logger=logger)

        # 加载工具
        tool_list = load_tools(
            config.get("tool_config", []),
            config.get("conda_env_name")
        )

        for i in range(agents_num):
            agent = BaseReActAgent(
                name=f"Agent_{i}",
                model=config.get("model_type"),
                tools=tool_list,
                prompt_manager=PromptManager(),
                max_steps=config.get("max_steps"),
                llm_client=llm_client,
                logger=logger,
            )
            pool.add(agent)

        return pool
