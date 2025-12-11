import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from core.execution.pipeline import Pipeline
from core.agent.agent_pool import AgentPool
from core.agent.base_agent import BaseReActAgent
from core.execution.journal import Journal, Node
from core.agent.prompt_manager import PromptContext
from utils.logger_system import log_msg

class IterationController:
    def __init__(
        self,
        agent_pool: AgentPool,
        task_pipeline: Pipeline,
        journal: Journal,
        config: Any
    ):
        self.agent_pool = agent_pool
        self.task_pipeline = task_pipeline
        self.journal = journal
        self.config = config

        self.current_epoch = 0
        self.start_time = time.time()

    async def run_competition(self):
        log_msg("INFO", "Starting competition loop...")
        while self.current_epoch < self.config.mle_bench_epoch_limit:
            self.current_epoch += 1
            log_msg("INFO", f"--- Starting Epoch {self.current_epoch} ---")
            await self.run_epoch()
        
        log_msg("INFO", "Competition loop finished.")

    async def run_epoch(self):
        """
        Run one epoch:
        1. Identify available agents.
        2. Fetch tasks from pipeline.
        3. Execute tasks concurrently.
        4. Save results to journal.
        """
        tasks_coroutines = []
        
        agents = list(self.agent_pool.agents.values())
        
        # Concurrency control: Limit the number of active agents
        max_concurrent = self.config.max_concurrent_agents
        active_agents = agents[:max_concurrent]
        
        log_msg("INFO", f"Epoch {self.current_epoch}: concurrency limit = {max_concurrent}, active agents = {len(active_agents)}")

        for agent in active_agents:
            task_item = self.task_pipeline.get_task()
            if task_item:
                task_item['agent_name'] = agent.name
                tasks_coroutines.append(self._run_single_task(agent, task_item))
            else:
                pass
        
        if tasks_coroutines:
            log_msg("INFO", f"Epoch {self.current_epoch}: Executing {len(tasks_coroutines)} tasks.")
            await asyncio.gather(*tasks_coroutines)
        else:
            log_msg("INFO", f"Epoch {self.current_epoch}: No tasks to execute.")
            await asyncio.sleep(1)

    async def _run_single_task(self, agent: BaseReActAgent, task: Dict[str, Any]):
        """
        Execute a single task with a specific agent.
        """
        task_id = task['id']
        log_msg("INFO", f"Agent {agent.name} starting task {task_id} ({task['type']})")
        
        prompt_context = self._construct_prompt_context(task)
        
        task_description = self._get_task_description(task)
        
        agent_input_state = {
            "task_description": task_description,
            "prompt_context": prompt_context
        }

        try:
            result = await agent(agent_input_state)
            
            node = self._create_node_from_result(result, task)
            self.journal.add_node(node)
            
            self.task_pipeline.complete_task(task_id)
            
            log_msg("INFO", f"Agent {agent.name} finished task {task_id}. Node {node.id} added.")
            
        except Exception as e:
            log_msg("ERROR", f"Agent {agent.name} failed task {task_id}: {e}")

    def _construct_prompt_context(self, task: Dict[str, Any]) -> PromptContext:
        """
        Build PromptContext from task payload and global config.
        """
        payload = task.get('payload', {})
        
        elapsed = time.time() - self.start_time
        remaining = self.config.time_limit_seconds - elapsed
        
        return PromptContext(
            workspace_root=self.config.mle_bench_workspace_dir,
            conda_env_name=self.config.conda_env_name,
            time_limit_seconds=self.config.time_limit_seconds,
            total_iterations=self.config.mle_bench_epoch_limit,
            iteration=self.current_epoch,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            conda_packages="", 
            task_description=self._get_task_description(task),
            
            parent_code=payload.get('parent_code'),
            parent_feedback=payload.get('parent_feedback'),
            candidates=payload.get('candidates'),
            gene_plan=payload.get('gene_plan'),
            solution_code=payload.get('solution_code'),
            execution_logs=payload.get('execution_logs'),
            
            template_name=payload.get('template_name') 
        )

    def _get_task_description(self, task: Dict[str, Any]) -> str:
        """
        Generate a human-readable task description based on task type.
        """
        t_type = task['type']
        if t_type == 'select':
            return "Please select the best solution strategy from the candidates."
        elif t_type == 'explore':
            return "Please explore a new solution based on the plan."
        elif t_type == 'merge':
            return "Please merge the selected strategies into a new solution."
        elif t_type == 'review':
             return "Please review the solution and provide feedback."
        elif t_type == 'debug':
             return "Please debug the current solution."
        else:
            return f"Execute task of type {t_type}"

    def _create_node_from_result(self, result: Dict[str, Any], task: Dict[str, Any]) -> Node:
        """
        Convert agent execution result into a Journal Node.
        """
        agent_output = result.get('agent_output', {})
        raw_session = result.get('raw_session') 
        
        code_content = ""
        if isinstance(agent_output, dict):
            code_content = agent_output.get('code', "")
            
        score = None
        if isinstance(agent_output, dict):
             score = agent_output.get('score')
        
        parent_id = task.get('payload', {}).get('parent_id')
        parent_ids = [parent_id] if parent_id else []

        logs = ""
        if raw_session:
            logs = json.dumps([h.get('observation') for h in raw_session.history], ensure_ascii=False)

        return Node(
            parent_ids=parent_ids,
            code=code_content,
            score=score,
            step=self.current_epoch,
            action_type=task['type'],
            logs=logs,
            metadata={
                "agent_name": result.get('agent_name'),
                "task_id": task['id'],
                "success": result.get('agent_success')
            }
        )
