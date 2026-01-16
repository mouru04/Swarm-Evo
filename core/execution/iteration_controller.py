import os
import shutil
import zipfile
import asyncio
import time
import json
import traceback
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from core.execution.pipeline import Pipeline
from core.agent.agent_pool import AgentPool
from core.agent.base_agent import BaseReActAgent
from core.execution.journal import Journal, Node
from core.execution.task_class import Task
from core.agent.prompt_manager import PromptContext
from core.evolution.gene_selector import select_gene_plan
from core.evolution.gene_registry import GeneRegistry
from utils.logger_system import log_msg
from core.optimization.reflector import PromptReflector
from core.optimization.generator import PromptGenerator
from core.optimization.version_manager import AgentVersionManager
from core.optimization.prompt_learner import PromptLearner


@dataclass
class TaskExecutionResult:
    """
    任务执行结果的统一封装
    """
    success: bool
    agent_name: str
    task_type: str
    task_id: str
    agent_output: Dict[str, Any]
    raw_session: Any = None
    error: Optional[str] = None
    result_nodes: List[Node] = field(default_factory=list)
    update_data: Dict[str, Any] = field(default_factory=dict)
    archive_path: Optional[str] = None


class IterationController:
    """
    迭代控制器

    负责管理Agent的执行流程、任务路由和优化触发
    """

    # 类常量
    SCORE_THRESHOLD = 0.65               # 学习/反思阈值
    REFLECTION_TRIGGER_THRESHOLD = 5     # 反思触发使用次数阈值
    TEMPLATE_MAP = {
        'review': 'evaluate_user_prompt.j2',
        'merge': 'merge_user_prompt.j2',
        'explore': 'explore_user_prompt.j2'
    }

    def __init__(
        self,
        agent_pool: AgentPool,
        task_pipeline: Pipeline,
        journal: Journal,
        config: Any,
        competition_description: str = "",
        conda_packages: str = ""

    ):
        self.agent_pool = agent_pool
        self.task_pipeline = task_pipeline
        self.journal = journal
        self.config = config
        self.competition_description = competition_description
        self.conda_packages = conda_packages

        self.current_epoch = 0
        self.start_time = time.time()
        
        # Gene Selection
        self.use_pheromone_gene_selection = config.use_pheromone_gene_selection
        self.gene_registry = GeneRegistry()
        self._gene_registry_updated_nodes: Set[str] = set()

        # 从 agent_pool 获取 llm 和 prompt_manager（假设所有 agent 相同）
        first_agent = next(iter(self.agent_pool.agents.values()), None)
        self.llm = first_agent.llm if first_agent else None
        self.prompt_manager = first_agent.prompt_manager if first_agent else None

        # 初始化版本管理器（传入 prompt_manager 以获取初始模板）
        self.version_manager = AgentVersionManager(
            storage_dir=os.path.join(self.config.mle_bench_workspace_dir, "agent_evolution"),
            prompt_manager=self.prompt_manager
        )

        # 初始化反思器和生成器
        self.reflector = PromptReflector(llm=self.llm, prompt_manager=self.prompt_manager) if self.llm and self.prompt_manager else None
        self.generator = PromptGenerator(
            llm=self.llm,
            prompt_manager=self.prompt_manager,
            version_manager=self.version_manager
        ) if self.llm and self.prompt_manager else None

        # 初始化学习器
        self.learner = PromptLearner(
            version_manager=self.version_manager,
            llm=self.llm,
            prompt_manager=self.prompt_manager,
            learning_threshold=getattr(self.config, 'learning_threshold', 0.1),
            storage_dir=os.path.join(self.config.mle_bench_workspace_dir, "prompt_learning")
        ) if self.llm and self.prompt_manager else None

    async def run_competition(self):
        """主竞争循环"""
        log_msg("INFO", "Starting competition loop...")
        while self.current_epoch < self.config.mle_bench_epoch_limit:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.config.time_limit_seconds:
                log_msg("WARNING", f"已达到时间限制 ({self.config.time_limit_seconds}秒), 停止竞赛循环。当前耗时: {elapsed_time:.2f}秒")
                break

            self.current_epoch += 1
            log_msg("INFO", f"--- Starting Epoch {self.current_epoch} ---")
            await self.run_epoch()

        log_msg("INFO", "Competition loop finished.")

    async def run_epoch(self):
        """
        执行一轮（epoch）

        一轮定义：所有 Agent 各执行一次任务
        任务分配：动态分配，执行完一个后再从 Pipeline 获取下一个任务
        """
        agents = list(self.agent_pool.agents.values())
        log_msg("INFO", f"Epoch {self.current_epoch}: {len(agents)} 个 Agent 轮询执行")

        for idx, agent in enumerate(agents):
            log_msg("INFO", f"Epoch {self.current_epoch}: 准备获取任务 {idx+1}/{len(agents)}")
            task_item = self.task_pipeline.get_task()
            if not task_item:
                log_msg("INFO", f"Epoch {self.current_epoch}: Pipeline 无更多任务，跳过 {agent.name}")
                continue

            task_item['agent_name'] = agent.name
            log_msg("INFO", f"Epoch {self.current_epoch}: {agent.name} ← {task_item['type']} 任务")

            # 串行执行（避免 workspace 冲突）
            log_msg("INFO", f"Epoch {self.current_epoch}: 开始执行 {agent.name} 的任务")
            await self._run_single_task(agent, task_item)
            log_msg("INFO", f"Epoch {self.current_epoch}: {agent.name} 的任务执行完成")

        log_msg("INFO", f"Epoch {self.current_epoch}: 轮询完成")

    async def _run_single_task(self, agent: BaseReActAgent, task: Task):
        """
        执行单个任务的主入口

        流程：
        0. MERGE任务：计算gene_plan
        1. 检查阶段 - explore/merge 任务执行前检查优化条件
        2. 准备阶段 - 构建上下文
        3. 执行阶段 - 调用Agent
        4. 处理阶段 - 根据成功/失败路由到不同的处理器
        5. 完成阶段 - 通知Pipeline并触发后续任务
        """
        task_id = task['id']
        task_type = task['type']
        log_msg("INFO", f"[DEBUG] _run_single_task 开始: {agent.name}, {task_type}, {task_id}")

        try:
            # 第零阶段：MERGE任务计算gene_plan
            if task_type == 'merge':
                payload = task['payload']
                gene_plan = self._maybe_compute_gene_plan(task)
                payload['gene_plan'] = gene_plan
                if gene_plan is None:
                    log_msg("WARNING", f"[MERGE] Task {task_id} running without gene_plan (fallback merge)")

            # 第一阶段：explore/merge 任务执行前检查优化条件
            if task_type in ['explore', 'merge']:
                log_msg("INFO", f"[DEBUG] 检查优化条件...")
                await self._check_and_run_reflection(agent.name, task_type)
                log_msg("INFO", f"[DEBUG] 优化条件检查完成")

            # 第二阶段：准备执行上下文
            log_msg("INFO", f"[DEBUG] 准备执行上下文...")
            prepared_data = self._prepare_task_execution(agent, task)
            log_msg("INFO", f"[DEBUG] 上下文准备完成")

            # 第三阶段：执行Agent任务
            log_msg("INFO", f"[DEBUG] 开始执行Agent任务...")
            execution_result = await self._execute_agent_task(agent, task, prepared_data)
            log_msg("INFO", f"[DEBUG] Agent任务执行完成: success={execution_result.success}")

            # 第四阶段：根据执行结果路由处理
            if execution_result.success:
                log_msg("INFO", f"[DEBUG] 处理成功任务...")
                await self._handle_successful_task(agent, task, execution_result)
                log_msg("INFO", f"[DEBUG] 成功任务处理完成")
            else:
                log_msg("INFO", f"[DEBUG] 处理失败任务...")
                await self._handle_failed_task(agent, task, execution_result)
                log_msg("INFO", f"[DEBUG] 失败任务处理完成")

            # 第五阶段：完成任务（通知Pipeline）
            log_msg("INFO", f"[DEBUG] 完成Pipeline任务...")
            self._complete_task_in_pipeline(task, execution_result)
            log_msg("INFO", f"[DEBUG] Pipeline任务完成")

            log_msg("INFO", f"[DEBUG] _run_single_task 完全结束")

        except Exception as e:
            log_msg("ERROR", f"Agent {agent.name} 执行任务 {task_id} 时发生异常: {e}")
            log_msg("ERROR", traceback.format_exc())

            # 标记任务失败
            self.task_pipeline.complete_task(task_id, result_nodes=[], update_data=None)

    # ========================================================================
    # Gene Selection Methods
    # ========================================================================

    def _maybe_compute_gene_plan(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Gene Selection Logic"""
        if not self.use_pheromone_gene_selection:
            return None
        log_msg("INFO", "[GENE-SELECT] Using pheromone + max-sim selection")

        try:
            self._update_gene_registry_from_journal()
            # 直接调用 gene_selector
            gene_plan = select_gene_plan(
                journal=self.journal,
                gene_registry=self.gene_registry,
                current_step=self.current_epoch,
            )
            self._log_gene_plan(gene_plan)
            return gene_plan

        except Exception as exc:
            log_msg(
                "WARNING",
                "Pheromone gene selection failed.\n"
                f"Exception: {exc}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return None

    def _log_gene_plan(self, gene_plan: Dict[str, Any]) -> None:
        """Log gene plan"""
        parts = []
        labels = [
            ("data", "data_source"),
            ("model", "model_source"),
            ("loss", "loss_source"),
            ("opt", "optimizer_source"),
            ("reg", "regularization_source"),
            ("init", "initialization_source"),
            ("tricks", "tricks_source"),
        ]
        for label, field in labels:
            spec = gene_plan.get(field)
            if isinstance(spec, dict):
                node_id = spec.get("source_node_id", "")
                gene_id = spec.get("gene_id", "")
                display = f"{node_id[:6]}:{gene_id[:6]}"
            else:
                display = "None"
            parts.append(f"{label}={display}")
        log_msg("INFO", "[GENE-PLAN] " + " ".join(parts))

    def _update_gene_registry_from_journal(self) -> None:
        """Update gene registry from journal"""
        for node_id, node in self.journal.nodes.items():
            if node_id in self._gene_registry_updated_nodes:
                continue
            pheromone = None
            if node.metadata:
                pheromone = node.metadata.get("pheromone_node")
            if pheromone is None:
                continue
            self.gene_registry.update_from_reviewed_node(node)
            self._gene_registry_updated_nodes.add(node_id)

    # ========================================================================
    # Prompt Optimization Methods
    # ========================================================================

    def _prepare_task_execution(self, agent: BaseReActAgent, task: Task) -> Dict[str, Any]:
        """
        准备任务执行所需的上下文数据

        Returns:
            包含 prompt_context 和 task_description 的字典
        """

        # 构建Prompt上下文
        prompt_context = self._construct_prompt_context(task)
        task_description = self._get_task_description(task)

        return {
            "prompt_context": prompt_context,
            "task_description": task_description
        }

    async def _execute_agent_task(
        self,
        agent: BaseReActAgent,
        task: Task,
        prepared_data: Dict[str, Any]
    ) -> TaskExecutionResult:
        """
        执行Agent任务

        Returns:
            TaskExecutionResult: 包含执行结果的封装对象
        """
        task_id = task['id']
        task_type = task['type']

        # 统一确定步数限制
        current_max_steps = agent.max_steps
        if task_type == 'review':
            current_max_steps = 1

        agent_input_state = {
            "task_description": prepared_data['task_description'],
            "prompt_context": prepared_data['prompt_context'],
            "max_steps": current_max_steps
        }

        result = await agent(agent_input_state)

        return TaskExecutionResult(
            success=result.get('agent_success', False),
            agent_name=agent.name,
            task_type=task_type,
            task_id=task_id,
            agent_output=result.get('agent_output', {}),
            raw_session=result.get('raw_session'),
            error=result.get('error')
        )

    async def _handle_successful_task(
        self,
        agent: BaseReActAgent,
        task: Task,
        execution_result: TaskExecutionResult
    ):
        """
        处理成功任务的路由器

        根据任务类型分发到具体的处理器
        """
        task_type = task['type']

        # 路由到具体的任务处理器
        if task_type == 'review':
            await self._process_review_task(agent, task, execution_result)
        elif task_type in ['explore', 'merge']:
            await self._process_explore_merge_task(agent, task, execution_result)
        else:
            log_msg("WARNING", f"未知的任务类型: {task_type}")

    async def _handle_failed_task(
        self,
        agent: BaseReActAgent,
        task: Task,
        execution_result: TaskExecutionResult
    ):
        """
        处理失败任务

        失败任务仍然记录prompt使用次数，但不创建节点、不归档文件
        """
        task_id = task['id']
        task_type = task['type']

        log_msg("WARNING", f"任务失败: {task_type} (ID: {task_id}) by {agent.name}")

        # explore/merge任务即使失败也要记录使用次数
        if task_type in ['explore', 'merge']:
            await self.version_manager.record_prompt_usage(agent.name, task_type)

        # 不创建节点、不归档文件

    async def _process_review_task(
        self,
        agent: BaseReActAgent,
        task: Task,
        execution_result: TaskExecutionResult
    ):
        """
        处理Review任务的成功结果

        逻辑：
        1. 提取review数据
        2. 记录到版本管理器（用于优化）
        3. 准备update_data供Pipeline更新节点
        """
        task_id = task['id']
        target_id = task['payload'].get('target_node_id')
        agent_output = execution_result.agent_output

        if not target_id:
            log_msg("WARNING", f"Review任务 {task_id} 缺少target_node_id")
            return

        target_node = self.journal.get_node(target_id)
        if not target_node:
            log_msg("WARNING", f"Review任务 {task_id} 的目标节点 {target_id} 不存在")
            return

        # 准备更新数据
        execution_result.update_data = {
            "score": agent_output.get('score'),
            "summary": agent_output.get('summary', ""),
            "is_bug": agent_output.get('is_bug', False),
            "agent_success": execution_result.success
        }

        # 记录review结果到版本管理器（核心优化需求）
        reviewed_prompt_type = target_node.action_type  # 'explore' 或 'merge'
        original_agent_name = target_node.metadata.get('agent_name', agent.name)
        prompt_version_id = target_node.metadata.get('prompt_version_id')  # 获取原始prompt版本ID
        original_task_id = target_node.metadata.get('task_id')  # 获取原始explore/merge任务ID

        await self.version_manager.record_review_result(
            agent_name=original_agent_name,
            prompt_type=reviewed_prompt_type,
            task_id=original_task_id,
            node_id=target_id,  # 被review的Node ID
            score=agent_output.get('score'),
            has_submission=agent_output.get('has_csv_submission', False),
            version_id=prompt_version_id  # 传递原始版本ID
        )

        log_msg("INFO", f"已记录review结果: {original_agent_name} - {reviewed_prompt_type} - "
                      f"task_id={original_task_id} - node_id={target_id} - "
                      f"version_id={prompt_version_id} - score={agent_output.get('score')}")

    async def _process_explore_merge_task(
        self,
        agent: BaseReActAgent,
        task: Task,
        execution_result: TaskExecutionResult
    ):
        """
        处理Explore/Merge任务的成功结果

        逻辑：
        1. 记录prompt使用次数（无论是否生成节点）
        2. 从执行结果中创建Node
        3. 归档solution文件
        """
        task_id = task['id']
        task_type = task['type']

        # 第一步：记录prompt使用次数（无论是否生成节点）
        log_msg("INFO", f"[DEBUG] 开始记录prompt使用次数...")
        await self.version_manager.record_prompt_usage(agent.name, task_type)
        log_msg("INFO", f"[DEBUG] prompt使用次数记录完成")

        # 第二步：创建节点
        execution_result.result_nodes = self._create_nodes_from_result(
            execution_result, task, agent.name
        )

        if not execution_result.result_nodes:
            log_msg("WARNING", f"{task_type}任务 {task_id} 未生成任何节点，但已记录使用次数")
            # 创建空的TaskReviewRecord
            await self.version_manager.record_task_execution(
                agent_name=agent.name,
                prompt_type=task_type,
                task_id=task_id
            )
            return

        # 第三步：归档文件
        archive_path = self._archive_solution_files(task_id, self.config.mle_bench_workspace_dir)
        if archive_path:
            execution_result.archive_path = archive_path
            for node in execution_result.result_nodes:
                node.archive_path = archive_path

        log_msg("INFO", f"{agent.name} 完成 {task_type} 任务 {task_id}，生成 {len(execution_result.result_nodes)} 个节点")

    def _complete_task_in_pipeline(self, task: Task, execution_result: TaskExecutionResult):
        """
        完成任务的最后一步：通知Pipeline

        Pipeline将：
        1. 更新任务状态
        2. 添加节点到Journal
        3. 创建后续任务（如review）
        """
        task_id = task['id']
        log_msg("INFO", f"[DEBUG] _complete_task_in_pipeline 开始: {task_id}")

        self.task_pipeline.complete_task(
            task_id=task_id,
            result_nodes=execution_result.result_nodes,
            update_data=execution_result.update_data if execution_result.update_data else None
        )

        log_msg("INFO", f"[DEBUG] _complete_task_in_pipeline 完成: {task_id}")

    # ========================================================================
    # Prompt Construction Methods
    # ========================================================================

    def _construct_prompt_context(self, task: Task, step_limit: Optional[int] = None) -> PromptContext:
        """
        构建Prompt上下文

        从task payload和全局配置中提取数据，构建PromptContext对象

        参数:
            task: 任务字典
            step_limit: 步数限制（已根据任务类型确定）
        """
        payload = task.get('payload', {})

        elapsed = time.time() - self.start_time
        remaining = self.config.time_limit_seconds - elapsed

        # 设置默认模板名称
        if payload.get('template_name') is None:
            payload['template_name'] = self.TEMPLATE_MAP.get(task['type'], 'explore_user_prompt.j2')

            # Explore任务继承处理
            if task['type'] == 'explore':
                parent_id = payload.get('parent_id')
                if parent_id:
                    parent_node = self.journal.get_node(parent_id)
                    if parent_node:
                        payload['parent_code'] = parent_node.code
                        payload['parent_feedback'] = parent_node.summary
                        # logs 对应模板中的 parent_history
                        payload['parent_history'] = parent_node.logs
                        payload['parent_score'] = parent_node.score

        # 动态填充数据
        candidates_data = {}
        gene_plan_data = {}
        solution_code = None
        execution_logs = None

        if task['type'] == 'merge':
            # Merge 任务逻辑更新
            gene_plan_data = payload.get('gene_plan')
            # 仍然需要 candidate code 用于 materialization
            candidate_ids = payload.get('candidate_ids', [])
            # 如果 gene_plan 存在，sources 也应该作为 candidates
            if gene_plan_data:
                for spec in gene_plan_data.values():
                    if isinstance(spec, dict) and spec.get("source_node_id"):
                         candidate_ids.append(spec.get("source_node_id"))

            candidate_ids = list(set(candidate_ids))
            for cid in candidate_ids:
                node = self.journal.get_node(cid)
                if node and node.code:
                    candidates_data[cid] = node.code

        elif task['type'] == 'review':
            # 获取被review节点的代码和日志
            target_id = payload.get('target_node_id')
            if target_id:
                node = self.journal.get_node(target_id)
                if node:
                    solution_code = node.code
                    execution_logs = node.logs

        # 如果没有提供step_limit，使用默认值
        if step_limit is None:
            step_limit = 10  # 默认步数限制

        return PromptContext(
            workspace_root=self.config.mle_bench_workspace_dir,
            conda_env_name=self.config.conda_env_name,
            time_limit_seconds=self.config.time_limit_seconds,
            total_iterations=self.config.mle_bench_epoch_limit,
            iteration=self.current_epoch,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            conda_packages=self.conda_packages,
            task_description=self._get_task_description(task),
            step_limit=step_limit,
            parent_code=payload.get('parent_code'),
            parent_feedback=payload.get('parent_feedback'),
            parent_score=payload.get('parent_score'),
            candidates=candidates_data if candidates_data else payload.get('candidates'),
            gene_plan=gene_plan_data if gene_plan_data else payload.get('gene_plan'),
            solution_code=solution_code if solution_code else payload.get('solution_code'),
            execution_logs=execution_logs if execution_logs else payload.get('execution_logs'),
            parent_history=payload.get('parent_history'),
            template_name=payload.get('template_name')
        )

    def _get_task_description(self, task: Task) -> str:
        """
        生成任务描述

        基于任务类型生成人类可读的任务描述，并添加竞赛背景
        """
        t_type = task['type']

        task_instructions = {
            'explore': "Please explore a new solution based on the plan.",
            'merge': "Please merge the selected strategies into a new solution.",
            'review': "Please review the solution and provide feedback."
        }

        task_instruction = task_instructions.get(t_type, f"Execute task of type {t_type}")

        # 添加竞赛背景
        if self.competition_description:
            return f"# Competition Background\n{self.competition_description}\n\n---\n\n# Your Task\n{task_instruction}"

        return task_instruction

    def _create_nodes_from_result(self, execution_result: TaskExecutionResult, task: Task, agent_name: str) -> List[Node]:
        """
        从执行结果中创建Journal节点

        策略：
        1. 从history中提取所有solution.py版本（针对explore/merge）
        2. 如果没有找到，使用final agent output作为fallback

        参数:
            execution_result: 任务执行结果
            task: 任务对象
            agent_name: 执行任务的agent名称

        返回:
            创建的节点列表
        """
        nodes = []
        raw_session = execution_result.raw_session
        task_type = task['type']

        # 处理父节点ID
        parent_ids = []
        if task['type'] == 'merge':
            # Merge node parents = Gene Sources
            gene_plan = task['payload'].get('gene_plan') or {}
            gene_source_ids = [
                spec.get("source_node_id")
                for spec in gene_plan.values()
                if isinstance(spec, dict) and spec.get("source_node_id")
            ]
            # Combine candidate_ids and gene_source_ids
            candidate_ids = task['payload'].get('candidate_ids', [])
            parent_ids = list(set(candidate_ids + gene_source_ids))
        else:
            parent_id = task.get('payload', {}).get('parent_id')
            if parent_id:
                parent_ids = [parent_id]

        # 提取完整日志
        logs = ""
        if raw_session:
            logs = json.dumps([h.get('observation') for h in raw_session.history], ensure_ascii=False)

        # 策略1: 从history中提取solution.py的多个版本
        if task['type'] in ['explore', 'merge'] and raw_session:
            history = raw_session.history
            seen_content = set()

            for i, step in enumerate(history):
                action = step.get('action') or step.get('tool')
                tool_input = step.get('input') or step.get('tool_input', {})

                if action == 'write_file' and isinstance(tool_input, dict):
                    path = tool_input.get('path', '')
                    content = tool_input.get('content', '')

                    if path.endswith('solution.py') and content and content not in seen_content:
                        seen_content.add(content)

                        # 获取当前prompt版本ID（用于review时匹配正确的版本）
                        current_prompt = self.version_manager.get_current_prompt(agent_name, task_type)
                        current_version_id = current_prompt.version_id if current_prompt else None

                        # 查找后续的执行日志
                        execution_log = ""
                        for j in range(i + 1, len(history)):
                            next_step = history[j]
                            next_action = next_step.get('action') or next_step.get('tool')
                            next_input = next_step.get('input') or next_step.get('tool_input', {})

                            if next_action in ['run_python', 'python', 'bash', 'cmd_line', 'execute_script', 'terminal']:
                                execution_log = next_step.get('observation', "")
                                break

                            if next_action == 'write_file' and isinstance(next_input, dict):
                                next_path = next_input.get('path', '')
                                if next_path.endswith('solution.py'):
                                    break

                        # 创建节点
                        node = Node(
                            parent_ids=parent_ids,
                            code=content,
                            score=None,
                            step=self.current_epoch,
                            action_type=task['type'],
                            logs=execution_log if execution_log else logs,
                            metadata={
                                "agent_name": execution_result.agent_name,
                                "task_id": task['id'],
                                "success": execution_result.success,
                                "version": "history_snapshot",
                                "prompt_version_id": current_version_id  # 保存当前prompt版本ID
                            }
                        )
                        nodes.append(node)

        # 策略2: Fallback到final agent output
        if not nodes:
            agent_output = execution_result.agent_output
            code_content = ""

            if isinstance(agent_output, dict):
                code_content = agent_output.get('code', "")

            if code_content:
                # 获取当前prompt版本ID
                current_prompt = self.version_manager.get_current_prompt(agent_name, task_type)
                current_version_id = current_prompt.version_id if current_prompt else None

                node = Node(
                    parent_ids=parent_ids,
                    code=code_content,
                    score=None if not isinstance(agent_output, dict) else agent_output.get('score'),
                    step=self.current_epoch,
                    action_type=task['type'],
                    logs=logs,
                    metadata={
                        "agent_name": execution_result.agent_name,
                        "task_id": task['id'],
                        "success": execution_result.success,
                        "version": "final_output",
                        "prompt_version_id": current_version_id  # 保存当前prompt版本ID
                    }
                )
                nodes.append(node)

        return nodes

    def _archive_solution_files(
        self,
        task_id: str,
        workspace_dir: str
    ) -> Optional[str]:
        """
        归档任务执行过程中的 solution.py 和 submission.csv 文件

        参数:
            task_id: 任务ID
            workspace_dir: 工作目录路径

        返回:
            归档文件的路径，如果文件不存在则返回None
        """
        try:
            solution_path = os.path.join(workspace_dir, "solution.py")
            submission_path = os.path.join(workspace_dir, "submission", "submission.csv")

            solution_exists = os.path.exists(solution_path)
            submission_exists = os.path.exists(submission_path)

            if not solution_exists and not submission_exists:
                log_msg("WARNING", f"任务 {task_id}: 未找到 solution.py 或 submission.csv，跳过归档")
                return None

            # 使用 .archives 隐藏目录
            archive_dir = os.path.join(workspace_dir, ".archives")
            os.makedirs(archive_dir, exist_ok=True)

            # 创建归档文件
            archive_filename = f"{task_id}.zip"
            archive_path = os.path.join(archive_dir, archive_filename)

            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if solution_exists:
                    zipf.write(solution_path, "solution.py")
                    log_msg("INFO", f"已归档 solution.py 到 {archive_path}")

                if submission_exists:
                    zipf.write(submission_path, "submission.csv")
                    log_msg("INFO", f"已归档 submission.csv 到 {archive_path}")

            return archive_path

        except Exception as e:
            log_msg("ERROR", f"归档文件失败 (任务 {task_id}): {e}")
            log_msg("ERROR", traceback.format_exc())
            return None

    async def _check_and_run_reflection(
        self,
        agent_name: str,
        task_type: str
    ):
        """
        检查并执行学习或反思-生成流程

        当prompt使用次数达到阈值时：
        1. 如果综合评分 < SCORE_THRESHOLD，触发学习器（向分数最高的agent学习）
        2. 如果综合评分 >= SCORE_THRESHOLD，触发反思-生成流程

        参数:
            agent_name: agent名称
            task_type: 任务类型 ('explore' 或 'merge')
        """
        # 第一阶段: 检查反思器和生成器是否可用
        if not self.reflector or not self.generator:
            return

        # 第二阶段: 检查是否达到反思条件
        usage_count = self.version_manager.get_current_prompt_usage_count(agent_name, task_type)
        if usage_count < self.REFLECTION_TRIGGER_THRESHOLD:
            return
        log_msg("INFO", f"检查学习/反思流程: {task_type} 任务当前prompt已使用 {usage_count} 次")

        # 获取当前版本的综合评分
        current_version = self.version_manager.get_current_prompt(agent_name, task_type)
        if not current_version:
            log_msg("WARNING", f"无法获取 {agent_name} 的 {task_type} 当前版本")
            return

        composite_score = current_version.composite_score
        log_msg("INFO", f"{agent_name} 的 {task_type} prompt综合评分: {composite_score:.3f} (阈值: {self.SCORE_THRESHOLD})")

        # 根据评分决定学习还是反思
        if composite_score < self.SCORE_THRESHOLD:
            # 评分较低，触发学习器
            log_msg("INFO", f"评分低于阈值，触发学习器: {agent_name} 需要向高分agent学习")
            await self._run_learning_for_agent(agent_name, task_type)
        else:
            # 评分较高，触发反思-生成流程
            log_msg("INFO", f"评分高于阈值，触发反思流程: {agent_name} 进行自我反思和改进")
            await self._run_reflection_generation(agent_name, task_type)

    async def _run_learning_for_agent(
        self,
        agent_name: str,
        task_type: str
    ):
        """
        为特定agent执行学习流程
        当agent的综合评分低于阈值时，向分数最高的agent学习prompt

        参数:
            agent_name: 需要学习的agent名称
            task_type: 任务类型 ('explore' 或 'merge')
        """
        if not self.learner:
            log_msg("WARNING", f"学习器不可用，无法为 {agent_name} 执行学习")
            return

        try:
            # 分析学习机会
            candidates = self.learner.analyze_learning_opportunities(
                prompt_type=task_type,
                min_agents=2
            )

            # 筛选出以当前agent为学生的候选对
            student_candidates = [
                c for c in candidates
                if c.student_agent == agent_name
            ]

            if not student_candidates:
                log_msg("INFO", f"未找到适合 {agent_name} 的学习机会（可能已经是最高分）")
                return

            # 选择最佳候选（分数差距最大的）
            best_candidate = student_candidates[0] 

            if not best_candidate:
                log_msg("WARNING", f"无法为 {agent_name} 选择学习候选")
                return

            log_msg("INFO", f"开始学习: {agent_name} (分数={best_candidate.student_score:.3f}) -> "
                          f"{best_candidate.teacher_agent} (分数={best_candidate.teacher_score:.3f}), "
                          f"差距={best_candidate.score_gap:.3f}")

            # 执行学习
            learning_result = await self.learner.execute_learning(candidate=best_candidate)

            if learning_result.success:
                log_msg("INFO", f"Prompt学习成功: {learning_result.reasoning}")
            else:
                log_msg("WARNING", f"Prompt学习失败: {learning_result.error}")

        except Exception as e:
            log_msg("ERROR", f"学习流程执行失败: {e}")
            log_msg("ERROR", traceback.format_exc())

    async def _run_reflection_generation(
        self,
        agent_name: str,
        task_type: str
    ):
        """
        为特定agent执行反思-生成流程
        当agent的综合评分高于阈值时，进行自我反思和改进

        参数:
            agent_name: agent名称
            task_type: 任务类型 ('explore' 或 'merge')
        """
        try:
            # 第一阶段: 获取当前版本记录
            current_version = self.version_manager.get_current_prompt(agent_name, task_type)

            if not current_version:
                log_msg("WARNING", f"无法获取 {agent_name} 的 {task_type} 当前版本")
                return

            # 第二阶段: 使用反思器分析版本
            reflection_result = await self.reflector.analyze_version(
                version_record=current_version
            )

            log_msg("INFO", f"反思分析完成: {task_type} - 准确率={reflection_result['metrics']['avg_accuracy']:.2f}, "
                          f"生成率={reflection_result['metrics']['avg_generation_rate']:.2f}, "
                          f"综合评分={reflection_result['metrics']['composite_score']:.3f}")

            # 第三阶段: 更新当前版本的reflection
            await self.version_manager.update_version_reflection(
                agent_name=agent_name,
                version_id=current_version.version_id,
                reflection=reflection_result.get('reflection', {})
            )
            log_msg("INFO", f"已更新版本reflection: {current_version.version_id}")

            # 第四阶段: 使用生成器生成新prompt
            current_prompt = current_version.prompt_content
            generation_result = await self.generator.generate_new_prompt(
                agent_name=agent_name,
                prompt_type=task_type,
                current_prompt=current_prompt,
                reflection=reflection_result.get('reflection', {})
            )

            if not generation_result.success:
                log_msg("ERROR", f"生成新prompt失败: {generation_result.error}")
                return

            log_msg("INFO", f"新prompt生成成功: {task_type} - 版本={generation_result.version}, "
                          f"修改项={len(generation_result.changes_made)}")

            # 第五阶段: 应用新prompt
            applied = await self.generator.apply_new_prompt(
                prompt_type=task_type,
                new_prompt=generation_result.new_prompt,
                version=generation_result.version
            )

            if applied:
                log_msg("INFO", f"新prompt已应用: {task_type} 模板已更新为版本 {generation_result.version}")

                # 在版本管理器中记录新版本
                await self.version_manager.record_prompt_version(
                    agent_name=agent_name,
                    version_id=generation_result.version,
                    prompt_type=task_type,
                    prompt_content=generation_result.new_prompt,
                    source="generated",
                    previous_version_id=current_version.version_id
                )
            else:
                log_msg("ERROR", f"应用新prompt失败: {task_type}")

        except Exception as e:
            log_msg("ERROR", f"反思流程执行失败: {e}")
            log_msg("ERROR", traceback.format_exc())
