import time
import uuid
import random
import math
from threading import Lock
from typing import List, Optional, Dict, Any

from core.execution.task_class import Task
from core.execution.journal import Journal, Node
from core.evolution.pheromone import ensure_node_stats, compute_node_pheromone
from utils.config import get_config
from utils.logger_system import log_msg


class Pipeline:
    """
    任务执行管道类
    
    负责管理任务的生命周期，包括任务的生成、分发、完成和后续任务的追加。
    支持多线程并发访问。
    """

    def __init__(self, journal: Journal):
        self.lock = Lock()
        self.tasks: List[Task] = []
        self.temporary_storage: Dict[str, Any] = {}
        self.journal = journal
        self.config = get_config()

    def initialize(self):
        """
        初始化管道，添加初始的 explore 任务。
        """
        with self.lock:
            log_msg("INFO", f"Initializing pipeline with {self.config.init_task_num} explore tasks.")
            for _ in range(self.config.init_task_num):
                self._add_task_internal(self._create_task("explore"))

    def _create_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None) -> Task:
        """
        创建新任务实例。

        Args:
            task_type: 任务类型（explore/select/merge/review等）
            payload: 任务负载数据，若为None则使用空字典

        Returns:
            Task: 新创建的任务对象
        """
        payload_dict = payload if payload is not None else {}
        return Task(
            id=str(uuid.uuid4()),
            type=task_type,
            priority=0,
            payload=payload_dict,
            status="pending",
            created_at=time.time(),
            agent_name=None,
            dependencies=None
        )

    def add_task(self, task: Task):
        """
        在结尾插入一个任务
        """
        with self.lock:
            self._add_task_internal(task)

    def _add_task_internal(self, task: Task):
        """
        内部添加任务方法（不加锁）
        """
        self.tasks.append(task)
        # log_msg("DEBUG", f"Task {task['id']} (type={task['type']}) added to pipeline.")

    def _prepend_task_internal(self, task: Task):
        """
        内部优先添加任务方法（添加到队首，不加锁）
        """
        self.tasks.insert(0, task)
        # log_msg("DEBUG", f"Task {task['id']} (type={task['type']}) prepended to pipeline.")

    def get_task(self) -> Optional[Task]:
        """
        在开头取出一个未分配任务。
        如果当前没有未分配任务，则根据配置自动生成一批新任务。
        """
        with self.lock:
            # 1. 查找第一个 pending 任务
            pending_tasks = [t for t in self.tasks if t['status'] == 'pending']

            # 2. 如果没有 pending 任务，生成新任务
            if not pending_tasks:
                log_msg("INFO", "No pending tasks found. Generating new tasks...")
                new_tasks = self._generate_new_tasks()
                for task in new_tasks:
                    self._add_task_internal(task)
                pending_tasks = new_tasks

            # 3. 再次检查（理论上应该有了，除非 step_task_num 为 0）
            if not pending_tasks:
                return None

            # 4. 取出第一个任务 (FIFO)
            task_to_run = pending_tasks[0]
            # 注意：此处不修改 status 为 running，通常由调用者（Agent）在确认接收后修改，
            # 或者在这里修改也可以。根据 User 描述 "取出一个未分配任务"，
            # 为了防止被重复取出，这里应该标记为 running 或者 assigned?
            # 这里的语义通常是 "Claim" 任务。如果不改状态，下次 get_task 还会取到它。
            # 所以必须修改状态。
            task_to_run['status'] = 'running'
            
            return task_to_run

    def _generate_new_tasks(self) -> List[Task]:
        """
        根据 explore_ratio 和 epoch_task_num 生成任务
        """
        new_tasks = []
        count = self.config.epoch_task_num

        for _ in range(count):
            if random.random() < self.config.explore_ratio:
                task_type = "explore"
                parent_node = self._select_parent_node_by_pheromone()
                parent_id = parent_node.id if parent_node else None
                payload = {"parent_id": parent_id} if parent_id else {}
                new_tasks.append(self._create_task(task_type, payload))
            else:
                task_type = "select"
                new_tasks.append(self._create_task(task_type))
        return new_tasks

    def _select_parent_node_by_pheromone(self) -> Optional[Node]:
        """
        使用信息素采样选择父节点。

        Returns:
            Optional[Node]: 选中的父节点，若无有效节点则返回None
        """
        valid_nodes = [
            node for node in self.journal.nodes.values()
            if node.score is not None and not node.is_buggy
        ]
        if not valid_nodes:
            return None

        current_step = max((n.step for n in self.journal.nodes.values()), default=0)
        score_min = self.journal.score_min
        score_max = self.journal.score_max

        pheromones = []
        for node in valid_nodes:
            ensure_node_stats(node)
            pheromone = node.metadata.get("pheromone_node")
            if pheromone is None:
                pheromone = compute_node_pheromone(
                    node,
                    current_step=current_step,
                    score_min=score_min,
                    score_max=score_max,
                )
                node.metadata["pheromone_node"] = pheromone
            pheromones.append(pheromone)

        temperature = 3.0
        logits = [p / temperature for p in pheromones]
        max_logit = max(logits)
        weights = [math.exp(l - max_logit) for l in logits]
        parent_node = random.choices(valid_nodes, weights=weights, k=1)[0]

        ensure_node_stats(parent_node)
        parent_node.metadata["usage_count"] += 1
        pheromone_value = parent_node.metadata.get("pheromone_node")
        if pheromone_value is None:
            log_msg("ERROR", f"父节点 {parent_node.id} 的pheromone_node为None，无法使用信息素值进行日志记录")
        log_msg(
            "INFO",
            f"Selected parent {parent_node.id} with pheromone "
            f"{pheromone_value:.4f}",
        )
        return parent_node

    def reorder_tasks(self):
        """
        根据时间先后对所有任务重新排序
        """
        with self.lock:
            self.tasks.sort(key=lambda t: t['created_at'])

    def complete_task(self, task_id: str, result_nodes: Optional[List[Node]] = None, update_data: Optional[Dict[str, Any]] = None):
        """
        完成一个任务，并根据特定规则添加后续任务。

        Args:
            task_id: 任务 ID
            result_nodes: 如果是 explore/merge 任务，传入新创建的 Node 对象列表
            update_data: 如果是 review 任务，传入更新数据 (score, summary, etc.)

        规则:
        - explore -> review (Priority Insert)
        - select -> merge (Priority Insert) -> Merge 任务创建逻辑在此处或 Controller 触发
        - merge -> review (Priority Insert)
        """
        nodes_to_add = result_nodes if result_nodes is not None else []

        with self.lock:
            # 查找并更新任务状态
            task = next((t for t in self.tasks if t['id'] == task_id), None)
            if not task:
                log_msg("WARNING", f"Attempted to complete unknown task {task_id}")
                return
            
            task['status'] = 'completed'
            # log_msg("INFO", f"Task {task_id} completed.")

            # --- Node Logic based on Task Type ---
            created_node_ids = []
            
            if task['type'] in ['explore', 'merge']:
                if nodes_to_add:
                    for node in nodes_to_add:
                        self.journal.add_node(node)
                        created_node_ids.append(node.id)
                        log_msg("INFO", f"Pipeline added Node {node.id} for task {task_id}")
            
            elif task['type'] == 'review':
                if update_data:
                    target_id = task['payload'].get('target_node_id')
                    if target_id:
                        node = self.journal.get_node(target_id)
                        if node:
                            node.score = update_data.get('score')
                            node.summary = update_data.get('summary', "")
                            node.is_buggy = update_data.get('is_bug', False)
                            if node.metadata is None:
                                node.metadata = {}
                            node.metadata['review_success'] = update_data.get('agent_success', False)
                            ensure_node_stats(node)
                            success = node.score is not None and not node.is_buggy
                            if success:
                                current_score_min = self.journal.score_min
                                current_score_max = self.journal.score_max
                                if current_score_min is None:
                                    self.journal.score_min = node.score
                                else:
                                    if node.score < current_score_min:  # type: ignore
                                        self.journal.score_min = node.score
                                if current_score_max is None:
                                    self.journal.score_max = node.score
                                else:
                                    if node.score > current_score_max:  # type: ignore
                                        self.journal.score_max = node.score
                            current_step = max((n.step for n in self.journal.nodes.values()), default=0)
                            node.metadata["pheromone_node"] = compute_node_pheromone(
                                node,
                                current_step=current_step,
                                score_min=self.journal.score_min,
                                score_max=self.journal.score_max,
                            )
                            if success:
                                for pid in node.parent_ids:
                                    parent = self.journal.get_node(pid)
                                    if not parent:
                                        continue
                                    ensure_node_stats(parent)
                                    parent.metadata["success_count"] += 1
                            log_msg("INFO", f"Pipeline updated Node {target_id} with score {node.score}")
                            pheromone_value = node.metadata.get("pheromone_node")
                            if pheromone_value is None:
                                log_msg("ERROR", f"节点 {node.id} 的pheromone_node为None，无法使用信息素值进行日志记录")
                            log_msg("INFO", f"[PHEROMONE-UPDATE] Node={node.id[:6]} score={node.score} pheromone={pheromone_value:.4f}")
                        else:
                            log_msg("WARNING", f"Review target node {target_id} not found.")
 
            # --- Follow-up Task Creation ---
            next_task_type = None
            
            if task['type'] == 'explore':
                next_task_type = 'review'
                if created_node_ids:
                    # Create a review task for EACH new node
                    for nid in created_node_ids:
                        payload = {"parent_id": task_id}
                        payload['target_node_id'] = nid
                        payload['template_name'] = "evaluate_user_prompt.j2"
                        
                        new_task = self._create_task(next_task_type, payload)
                        self._prepend_task_internal(new_task)
                else:
                    log_msg("WARNING", "Explore task completed without any result nodes. Skipping Review.")
                    return

            elif task['type'] == 'select':
                # Select -> Merge logic remains in Controller for now.
                pass 

            elif task['type'] == 'merge':
                next_task_type = 'review'
                if created_node_ids:
                    for nid in created_node_ids:
                         payload = {"parent_id": task_id}
                         payload['target_node_id'] = nid
                         payload['template_name'] = "evaluate_user_prompt.j2"
                         
                         new_task = self._create_task(next_task_type, payload)
                         self._prepend_task_internal(new_task)
                else:
                    log_msg("WARNING", "Merge task completed without any result nodes. Skipping Review.")
                    return

    def store_result(self, task_id: str, data: Any):
        """
        暂存任务结果
        """
        with self.lock:
            self.temporary_storage[task_id] = data

    def retrieve_result(self, task_id: str, pop: bool = True) -> Any:
        """
        获取暂存的任务结果
        """
        with self.lock:
            if pop:
                return self.temporary_storage.pop(task_id, None)
            return self.temporary_storage.get(task_id)

    def add_urgent_task(self, task: Task):
        """
        添加高优先级任务（插入队首）
        """
        with self.lock:
            self._prepend_task_internal(task)
