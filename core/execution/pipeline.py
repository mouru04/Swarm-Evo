import time
import uuid
import random
from threading import Lock
from typing import List, Optional, Dict, Any

from core.execution.task_class import Task
from utils.config import get_config
from utils.logger_system import log_msg


class Pipeline:
    """
    任务执行管道类
    
    负责管理任务的生命周期，包括任务的生成、分发、完成和后续任务的追加。
    支持多线程并发访问。
    """

    def __init__(self):
        self.lock = Lock()
        self.tasks: List[Task] = []
        self.config = get_config()

    def initialize(self):
        """
        初始化管道，添加初始的 explore 任务。
        """
        with self.lock:
            log_msg("INFO", f"Initializing pipeline with {self.config.init_task_num} explore tasks.")
            for _ in range(self.config.init_task_num):
                self._add_task_internal(self._create_task("explore"))

    def _create_task(self, task_type: str, payload: Dict[str, Any] = None) -> Task:
        """
        创建新任务实例
        """
        return Task(
            id=str(uuid.uuid4()),
            type=task_type,
            priority=0,
            payload=payload or {},
            status="pending",
            created_at=time.time(),
            agent_name=None
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
            else:
                task_type = "select"
            new_tasks.append(self._create_task(task_type))
        return new_tasks

    def reorder_tasks(self):
        """
        根据时间先后对所有任务重新排序
        """
        with self.lock:
            self.tasks.sort(key=lambda t: t['created_at'])

    def complete_task(self, task_id: str):
        """
        完成一个任务，并根据特定规则添加后续任务。
        
        规则:
        - explore -> review
        - select -> merge
        - merge -> review
        """
        with self.lock:
            # 查找并更新任务状态
            task = next((t for t in self.tasks if t['id'] == task_id), None)
            if not task:
                log_msg("WARNING", f"Attempted to complete unknown task {task_id}")
                return
            
            task['status'] = 'completed'
            # log_msg("INFO", f"Task {task_id} completed.")

            # 根据规则添加后续任务
            next_task_type = None
            payload = {"parent_id": task_id}

            if task['type'] == 'explore':
                next_task_type = 'review'
                payload['review_target'] = 'explore_result' # 示例 payload
            elif task['type'] == 'select':
                next_task_type = 'merge'
                payload['merge_source'] = 'select_result'
            elif task['type'] == 'merge':
                next_task_type = 'review'
                payload['review_target'] = 'merge_result'
            
            if next_task_type:
                new_task = self._create_task(next_task_type, payload)
                self._add_task_internal(new_task)
                # log_msg("INFO", f"Follow-up task {new_task['id']} ({next_task_type}) added.")
