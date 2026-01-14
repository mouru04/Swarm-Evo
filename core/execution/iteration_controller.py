import os
import shutil
import zipfile
import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Set
import uuid
from core.execution.pipeline import Pipeline
from core.agent.agent_pool import AgentPool
from core.agent.base_agent import BaseReActAgent
from core.execution.journal import Journal, Node
from core.agent.prompt_manager import PromptContext
from core.evolution.gene_selector import select_gene_plan
from core.evolution.gene_registry import GeneRegistry
from utils.logger_system import log_msg
from core.execution.task_class import Task

class IterationController:
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
        
        # [PR #2 Feature] Gene Selection
        self.use_pheromone_gene_selection = config.use_pheromone_gene_selection
        self.gene_registry = GeneRegistry()
        self._gene_registry_updated_nodes: Set[str] = set()

    async def run_competition(self):
        log_msg("INFO", "Starting competition loop...")
        while self.current_epoch < self.config.mle_bench_epoch_limit:
            # [Main Feature] Check time limit
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
        执行一轮（epoch）。

        一轮定义：所有 Agent 各执行一次任务。
        任务分配：动态分配，执行完一个后再从 Pipeline 获取下一个任务，
                  可及时响应新插入的任务（如 explore 完成后自动插入的 review）。
        """
        agents = list(self.agent_pool.agents.values())
        log_msg("INFO", f"Epoch {self.current_epoch}: {len(agents)} 个 Agent 轮询执行")

        for agent in agents:
            # 动态获取任务（可能是刚插入的 review/merge）
            task_item = self.task_pipeline.get_task()
            if not task_item:
                log_msg("INFO", f"Epoch {self.current_epoch}: Pipeline 无更多任务，跳过 {agent.name}")
                continue

            task_item['agent_name'] = agent.name
            log_msg("INFO", f"Epoch {self.current_epoch}: {agent.name} ← {task_item['type']} 任务")

            # 串行执行（避免 workspace 冲突）
            await self._run_single_task(agent, task_item)

        log_msg("INFO", f"Epoch {self.current_epoch}: 轮询完成")


    async def _run_single_task(self, agent: BaseReActAgent, task: Dict[str, Any]):
        """
        Execute a single task with a specific agent.
        """
        task_id = task['id']
        task_type = task['type']
        log_msg("INFO", f"Agent {agent.name} starting task {task_id} ({task_type})")
        
        # ---- 初始化----
        result_nodes = []
        update_data = None

        # =====================================================
        # 1.MERGE 任务：在调用 LLM 之前完成“决策层工作” (PR #2 Feature)
        # =====================================================
        if task_type == 'merge':
            payload = task['payload']

            # ---- merge -- gene selection ----
            gene_plan = self._maybe_compute_gene_plan(task)
            payload['gene_plan'] = gene_plan  # 允许为 None（一般不会为none）

            if gene_plan is None:
                log_msg(
                    "WARNING",
                    f"[MERGE] Task {task_id} running without gene_plan (fallback merge)"
                )

        # =====================================================
        # 2.统一确定步数限制
        # =====================================================
        current_max_steps = agent.max_steps
        if task_type == 'review':
            current_max_steps = 1

        # =====================================================
        # 3.构造 PromptContext
        # =====================================================
        prompt_context = self._construct_prompt_context(task, step_limit=current_max_steps)
        task_description = self._get_task_description(task)
        
        agent_input_state = {
            "task_description": task_description,
            "prompt_context": prompt_context,
            "max_steps": current_max_steps
        }

        try:
            result = await agent(agent_input_state)
            
            # 2. 根据任务类型处理结果

            # ---- REVIEW ----
            if task_type == 'review':
                target_id = task['payload'].get('target_node_id')
                if target_id:
                    agent_output = result.get('agent_output', {})
                    update_data = {
                        "score": agent_output.get('score'),
                        "summary": agent_output.get('summary', ""),
                        "is_bug": agent_output.get('is_bug', False),
                        "agent_success": result.get('agent_success')
                    }
                    log_msg("INFO", f"Review task {task_id} completed. Update data: {update_data}")
                else:
                     log_msg("WARNING", f"Review task target node {target_id} not found.")

            # ---- MERGE / EXPLORE ----
            # Explore / Merge 任务: 创建新 Node 对象但不直接添加
            else:
                result_nodes = self._create_nodes_from_result(result, task)
                
                # [Main Feature] Archive solution.py and submission.csv
                archive_path = self._archive_solution_files(task_id, self.config.mle_bench_workspace_dir)
                for node in result_nodes:
                    node.archive_path = archive_path

                log_msg(
                    "INFO", 
                    f"Agent {agent.name} finished task {task_id}. "
                    f"Generated {len(result_nodes)} Nodes: {[n.id for n in result_nodes]}"
                )

            # 3. 完成任务，并将结果传递给 Pipeline 处理 (Journal 更新, 后续任务触发)
            self.task_pipeline.complete_task(
                task_id=task_id,
                result_nodes=result_nodes if result_nodes else [], 
                update_data=update_data
            )
            
        except Exception as e:
            log_msg("ERROR", f"Agent {agent.name} failed task {task_id}: {e}")
            import traceback
            log_msg("ERROR", traceback.format_exc())

    def _maybe_compute_gene_plan(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # [PR #2 Feature] Gene Selection Logic
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
            import traceback

            log_msg(
                "WARNING",
                "Pheromone gene selection failed.\n"
                f"Exception: {exc}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return None

    def _log_gene_plan(self, gene_plan: Dict[str, Any]) -> None:
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

    def _construct_prompt_context(self, task: Dict[str, Any], step_limit: int) -> PromptContext:
        """
        Build PromptContext from task payload and global config.
        
        参数:
            task: 任务字典
            step_limit: 步数限制（已根据任务类型确定）
        """
        payload = task.get('payload', {})
        
        elapsed = time.time() - self.start_time
        remaining = self.config.time_limit_seconds - elapsed

        if payload.get('template_name') == None:
            if task['type'] == 'review':
                payload['template_name']  = "evaluate_user_prompt.j2"
            elif task['type'] == 'merge':
                payload['template_name']  = "merge_user_prompt.j2"
            else:
                payload['template_name'] = "explore_user_prompt.j2"
                
                # [NEW Logic] Explore 任务继承处理
                # 如果 payload 中携带 parent_id，则加载该节点作为上下文
                parent_id = payload.get('parent_id')
                if parent_id:
                    parent_node = self.journal.get_node(parent_id)
                    if parent_node:
                        # 填充继承数据
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
            target_id = payload.get('target_node_id')
            if target_id:
                node = self.journal.get_node(target_id)
                if node:
                    solution_code = node.code
                    execution_logs = node.logs # 假设 logs 存在 Node 中

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
            
            # 显式传递 parent_history (对应 logs)
            parent_history=payload.get('parent_history'),
            
            template_name=payload.get('template_name') 
        )

    def _get_task_description(self, task: Dict[str, Any]) -> str:
        """
        Generate a human-readable task description based on task type.
        Prepends competition description as background context.
        """
        t_type = task['type']
        if t_type == 'select':
            task_instruction = "Please select the best solution strategy from the candidates."
        elif t_type == 'explore':
            task_instruction = "Please explore a new solution based on the plan."
        elif t_type == 'merge':
            task_instruction = "Please merge the selected strategies into a new solution."
        elif t_type == 'review':
            task_instruction = "Please review the solution and provide feedback."
        elif t_type == 'debug':
            task_instruction = "Please debug the current solution."
        else:
            task_instruction = f"Execute task of type {t_type}"
        
        return task_instruction

    def _create_nodes_from_result(self, result: Dict[str, Any], task: Dict[str, Any]) -> List[Node]:
        """
        Convert agent execution result into a list of Journal Nodes.
        Extracts multiple versions of solution.py if available in history.
        """
        nodes = []
        raw_session = result.get('raw_session')
        
        # Parent handling
        parent_ids = []
        if task['type'] == 'merge':
             # Merge node parents = All Candidates + Gene Sources
             gene_plan = task['payload'].get('gene_plan') or {}
             parent_ids = list({
                spec.get("source_node_id")
                for spec in gene_plan.values()
                if isinstance(spec, dict) and spec.get("source_node_id")
             })
             # Also include any explicit candidates if needed, but gene sources are primary
        else:
             parent_id = task.get('payload', {}).get('parent_id')
             if parent_id:
                 parent_ids = [parent_id]
        
        # Extract logs once
        logs = ""
        if raw_session:
            logs = json.dumps([h.get('observation') for h in raw_session.history], ensure_ascii=False)

        # Strategy 1: Parse history for write_file to solution.py (Only for Explore tasks)
        if task['type'] in ['explore', 'merge'] and raw_session:
            history = raw_session.history
            seen_content = set()
            
            # Iterate through history to find all write_file actions for solution.py
            for i, step in enumerate(history):
                action = step.get('action') or step.get('tool')
                tool_input = step.get('input') or step.get('tool_input', {})
                
                if action == 'write_file' and isinstance(tool_input, dict):
                    path = tool_input.get('path', '')
                    content = tool_input.get('content', '')
                    
                    if path.endswith('solution.py') and content:
                        if content not in seen_content:
                            seen_content.add(content)
                            
                            # Look ahead for execution logs
                            execution_log = ""
                            for j in range(i + 1, len(history)):
                                next_step = history[j]
                                next_action = next_step.get('action') or next_step.get('tool')
                                next_input = next_step.get('input') or next_step.get('tool_input', {})
                                
                                # Check for execution commands
                                if next_action in ['run_python', 'python', 'bash', 'cmd_line', 'execute_script', 'terminal']:
                                    # Very loose check: if it runs python, assume it runs the solution
                                    # Or check if 'solution.py' is in the command?
                                    # For now, take the first execution result as related.
                                    execution_log = next_step.get('observation', "")
                                    break
                                
                                # Stop if we hit another write to solution.py
                                if next_action == 'write_file' and isinstance(next_input, dict):
                                    next_path = next_input.get('path', '')
                                    if next_path.endswith('solution.py'):
                                        break

                            # Create a node for this version
                            node = Node(
                                parent_ids=parent_ids,
                                code=content,
                                score=None, 
                                step=self.current_epoch,
                                action_type=task['type'],
                                logs=execution_log if execution_log else logs, # Prefer specific execution log, fallback to full session
                                metadata={
                                    "agent_name": result.get('agent_name'),
                                    "task_id": task['id'],
                                    "success": result.get('agent_success'),
                                    "version": "history_snapshot"
                                }
                            )
                            nodes.append(node)
                            
        # Strategy 2: Fallback to final agent output if no nodes found (or for Merge tasks)
        if not nodes:
            agent_output = result.get('agent_output', {})
            code_content = ""
            if isinstance(agent_output, dict):
                code_content = agent_output.get('code', "")
            
            # If code is present, create a node
            if code_content:
                node = Node(
                    parent_ids=parent_ids,
                    code=code_content,
                    score=None if not isinstance(agent_output, dict) else agent_output.get('score'),
                    step=self.current_epoch,
                    action_type=task['type'],
                    logs=logs,
                    metadata={
                        "agent_name": result.get('agent_name'),
                        "task_id": task['id'],
                        "success": result.get('agent_success'),
                        "version": "final_output"
                    }
                )
                nodes.append(node)
                
        return nodes
    
    def _archive_solution_files(self, task_id: str, workspace_dir: str) -> Optional[str]:
        """
        Compress solution.py and submission.csv into a zip file in a hidden directory.
        Returns the absolute path to the archive if successful, else None.
        """
        archive_dir = os.path.join(workspace_dir, ".archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        files_to_archive = ["solution.py", "submission.csv"]
        found_files = []
        
        # Check existence
        for filename in files_to_archive:
            file_path = os.path.join(workspace_dir, filename)
            if os.path.exists(file_path):
                found_files.append(filename)
        
        if not found_files:
            return None
            
        archive_name = f"{task_id}.zip"
        archive_path = os.path.join(archive_dir, archive_name)
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for filename in found_files:
                    file_path = os.path.join(workspace_dir, filename)
                    zipf.write(file_path, arcname=filename)
            
            # log_msg("INFO", f"Archived {found_files} to {archive_path}")
            return archive_path
        except Exception as e:
            log_msg("ERROR", f"Failed to archive files for task {task_id}: {e}")
            return None
