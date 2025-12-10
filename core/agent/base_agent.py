from langchain_core.prompts import ChatPromptTemplate
from utils.toon_format import normalize_toon_output, validate_toon_format, parse_tool_input

class BaseReActAgent:

    # 识别 ActionStep(...) / FinalAnswer(...)
    ACTION_PATTERN = re.compile(
        r"\A\s*ActionStep\s*\(\s*(?P<content>.*)\)\s*\Z",
        re.S
    )

    FINAL_PATTERN = re.compile(
        r"\A\s*FinalAnswer\s*\(\s*(?P<content>.*)\)\s*\Z",
        re.S
    )

    # input: ( key: value )
    TOON_INPUT_PATTERN = re.compile(
        r"input\s*:\s*\(\s*(?P<body>.*)\)",
        re.S
    )

    # task: "..."
    TOON_TASK_PATTERN = re.compile(
        r'task\s*:\s*"(?P<val>.*?)"',
        re.S
    )

    # action: "..."
    TOON_ACTION_PATTERN = re.compile(
        r'action\s*:\s*"(?P<val>.*?)"',
        re.S
    )

    def __init__(
        self,
        name: str,
        model: str,
        tools: List[BaseTool],
        prompt_manager: PromptManager,
        max_steps: int,
        llm_client: Optional[OpenAI],
        logger: Optional[LoggerSystem],
    ):
        self.MAX_LLM_RETRY = 12
        
        self.name = name
        self.model = model
        self.max_steps = max_steps
        self.llm_client = llm_client
        self.logger = logger

        # 工具字典
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

        # 系统 + 用户提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_instruction}"),
            ("user", "{user_instruction}")
        ])
    
    def llm(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        调用 LLM 并解析 TOON 输出：
        - ActionStep(...)
        - FinalAnswer(...)
        """
        for attempt in range(1, self.MAX_LLM_RETRY + 1):
            try:
                self.logger.text_log("INFO", f"Agent '{self.name}' 尝试第 {attempt} 次调用 LLM, 使用模型 {self.model}")

                # 获取 LLM 响应
                resp = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                choices = getattr(resp, "choices", None)
                if not choices:
                    self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，LLM 响应格式错误")
                    raise ValueError("LLM response missing choices")
                
                raw_text = getattr(choices[0].message, "content", None)
                if raw_text is None and isinstance(message, dict):
                    raw_text = message.get("content")
                if raw_text is None:
                    self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，LLM 响应格式错误")
                    raise ValueError("LLM response missing content")
                if not isinstance(raw_text, str):
                    self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，LLM 响应格式错误")
                    raise TypeError("LLM response content type error")

                # TOON 正规化，去除 fenced code
                normalized = self._normalize_toon_output(raw_text)
                # 1) FinalAnswer(...)
                final_match = self.FINAL_PATTERN.fullmatch(normalized)
                if final_match:
                    validate_toon_format(normalized, self.logger)
                    return LLMResponse(
                        raw_output=raw_text,
                        normalized_output=normalized,
                        is_final=True,
                        final_answer=normalized,
                        task=None,
                        action=None,
                        tool_input=None
                    )
                # 2) ActionStep(...)
                action_match = self.ACTION_PATTERN.fullmatch(normalized)
                if action_match:
                    validate_toon_format(normalized, self.logger)
                    content = action_match.group("content")
                    
                    # 提取 task
                    task_match = self.TOON_TASK_PATTERN.search(content)
                    if not task_match:
                        self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，缺乏 task 字段")
                        raise ValueError("ActionStep missing task field")
                    task = task_match.group("val")

                    # 提取 action
                    action_match = self.TOON_ACTION_PATTERN.search(content)
                    if not action_match:
                        self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，缺乏 action 字段")
                        raise ValueError("ActionStep missing action field")
                    action_name = action_match.group("val")

                    # 整段 input: (...)
                    input_start = content.find("input")
                    if input_start < 0:
                        self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，缺乏 input 字段")
                        raise ValueError("ActionStep missing input field")
                    tool_input_text = content[input_start:].strip()

                    return LLMResponse(
                        raw_output=raw_text,
                        normalized_output=normalized,
                        is_final=False,
                        final_answer=None,
                        task=task,
                        action=action_name,
                        tool_input=tool_input_text,
                    )

                    self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，TOON 格式错误")
                    raise ValueError("LLM output is neither ActionStep(...) nor FinalAnswer(...)")
            except Exception as exc:
                if attempt < self.MAX_LLM_RETRY:
                    delay = min(self.RETRY_BASE_DELAY * attempt, 10.0)
                    self.logger.text_log("WARNING", f"Agent '{self.name}' 调用 LLM 失败，第 {attempt} 次重试，延迟 {delay} 秒")
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.logger.text_log("ERROR", f"Agent '{self.name}' 调用 LLM 失败，重试次数已达上限")
                    RuntimeError(f"LLM consecutive requests failed: {exc}")
        
    def _react_loop(self, instruction_text: str, prompt_context: PromptContext) -> AgentSessionResult:
        """
        将原来的 run() 完整迁移到这里，不对逻辑做任何更改。
        """

        history_records: List[Dict[str, Any]] = []
        consecutive_tool_errors: int = 0
        step: int = 0

        while step < self.max_steps:
            print(f"step: {step}")

            # 平铺历史（TOON）
            history = render_history(history_records)

            # 第一阶段：构造提示词
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

            # 第二阶段：调用 LLM（你的 TOON ActionStep/FinalAnswer 解析）
            response = self.llm(chat_messages)

            # 第三阶段：FinalAnswer
            if response.is_final:
                if response.final_answer is None:
                    self.logger.text_log("ERROR", "FinalAnswer missing content")
                    consecutive_tool_errors += 1
                    if consecutive_tool_errors > self.MAX_TOOL_RETRY:
                        self.logger.text_log("ERROR", "FinalAnswer missing content and consecutive retries failed")
                        raise RuntimeError("FinalAnswer missing content and consecutive retries failed")
                    continue

                self.logger.json_log({
                    "step": step,
                    "task": response.task,
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
                        self.logger.text_log("ERROR", "Tool input missing")
                        raise ValueError("Tool input missing")

                    parsed_input, _detected = self._parse_tool_input(response.action, response.tool_input)
                    normalized_input = self._normalize_tool_input(response.action, parsed_input)
                    observation = tool.run(normalized_input)

                except Exception as e:
                    observation = f"Tool execution failed: {e}"
                    self.logger.text_log("ERROR", observation)

            else:
                observation = f"Unknown tool: {response.action}"
                self.logger.text_log("ERROR", observation)

            # 工具错误处理
            if isinstance(observation, str) and (
                observation.startswith("Unknown tool") or
                observation.startswith("Tool execution failed")
            ):
                consecutive_tool_errors += 1
                if consecutive_tool_errors > self.MAX_TOOL_RETRY:
                    self.logger.text_log("ERROR", "Consecutive tool failures, terminating")
                    raise RuntimeError(f"Consecutive tool failures, terminating: {observation}")

                # 加入历史记录（错误）
                self.logger.json_log({
                    "step": step,
                    "task": response.task,
                    "final_answer": None,
                    "action": response.action,
                    "tool_input": response.tool_input,
                    "observation": f"Error: {observation}",
                })

                history_records.append({
                    "step": step,
                    "task": response.task,
                    "action": response.action,
                    "tool_input": response.tool_input,
                    "observation": f"Error: {observation}",
                })
                step += 1
                continue

            # 正常观察
            consecutive_tool_errors = 0

            self.logger.json_log({
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
                "action": response.action,
                "tool_input": response.tool_input,
                "observation": str(observation),
            })
            step += 1

        # 超过最大步数
        self.logger.text_log("WARNING", "达到最大 ReAct 步数，任务未完成")
        return AgentSessionResult(
            final_answer="达到最大 ReAct 步数，任务未完成。",
            history=history_records,
            success=False
        )
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph 节点入口：Agent 被当作节点时会执行这个方法。
        state 必须包含:
            - "task_description": str
            - "prompt_context": PromptContext
        """

        instruction_text = state["task_description"]
        prompt_context = state["prompt_context"]

        # 调用内部 ReAct 推理循环（原 run）
        session = self._react_loop(instruction_text, prompt_context)

        # 将结果合并回 LangGraph state
        return {
            "agent_output": session.final_answer,
            "agent_history": session.history,
            "agent_success": session.success,
            "agent_name": self.name,
            "raw_session": session,      # 如果后续节点需要更完整的 session
        }



