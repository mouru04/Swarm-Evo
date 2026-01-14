"""
Agent版本管理器模块

用于记录和管理每个Agent的prompt版本历史、性能数据和反思建议。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
import asyncio
import time
from utils.logger_system import log_msg


@dataclass
class NodeMetadata:
    """单个Node的Review结果"""
    score: Optional[float]                              # Review分数
    has_submission: bool                                # 是否有提交
    timestamp: str                                      # Review时间 (ISO格式)
    node_id: str                                        # 关联的Node ID（被review的节点）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class TaskReviewRecord:
    """单个explore/merge任务的Review记录"""
    num: int                                            # Review序号
    task_score: float                                   # 任务分数 = node_metadata的平均值
    has_submission: bool                                # 是否有提交 = node_metadata中任一为true则为true
    task_id: str                                        # 关联的任务ID（explore/merge任务）
    node_metadata: List[NodeMetadata] = field(default_factory=list)  # 该任务的所有node的review结果

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理NodeMetadata列表
        data['node_metadata'] = [
            nm.to_dict() if hasattr(nm, 'to_dict') else nm for nm in self.node_metadata
        ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskReviewRecord':
        """从字典创建实例"""
        # 处理node_metadata字段
        node_metadata_data = data.get('node_metadata', [])
        node_metadata = [
            NodeMetadata(**nm) if isinstance(nm, dict) else nm
            for nm in node_metadata_data
        ]
        data['node_metadata'] = node_metadata
        return cls(**data)


@dataclass
class PromptVersionRecord:
    """Prompt版本记录"""
    version_id: str                                      # 版本唯一标识
    prompt_type: str                                     # prompt类型 (explore/merge)
    prompt_content: str                                  # prompt内容
    created_at: str                                      # 创建时间 (ISO格式)
    source: str                                          # 版本来源 (initial/generated/manual/learned)

    # 使用统计
    used_count: int = 0                                  # 该版本使用次数

    # Review记录列表（按explore/merge任务分组）
    review_records: List[TaskReviewRecord] = field(default_factory=list)  # 该版本的所有review记录

    # 综合评分
    composite_score: float = 0.0                        # 综合评分 = 加权(平均生成率, 平均准确率)
    avg_generation_rate: float = 0.0                    # 平均生成率
    avg_accuracy: float = 0.0                           # 平均准确率

    # 反思相关信息
    guidance: str = ""                                   # 反思器生成的改进建议（包含分析、建议、优先级等）
    previous_version_id: Optional[str] = None            # 前一个版本ID

    # 交叉来源信息（用于学习器）
    crossover_source: Optional[Dict[str, str]] = None    # 交叉来源 {"agent": "agent_1", "version_id": "explore_v2"}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理TaskReviewRecord列表
        data['review_records'] = [
            r.to_dict() if hasattr(r, 'to_dict') else r for r in self.review_records
        ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVersionRecord':
        """从字典创建实例"""
        # 处理review_records字段
        review_records_data = data.get('review_records', [])
        review_records = [
            TaskReviewRecord.from_dict(r) if isinstance(r, dict) else r
            for r in review_records_data
        ]
        data['review_records'] = review_records
        return cls(**data)


@dataclass
class AgentEvolutionRecord:
    """Agent进化记录"""
    agent_name: str                                      # Agent名称
    created_at: str                                      # 记录创建时间
    last_updated: str                                    # 最后更新时间

    # Prompt版本历史
    prompt_versions: List[PromptVersionRecord] = field(default_factory=list)  # 版本历史列表

    # 当前状态
    current_explore_version: Optional[str] = None        # 当前explore版本ID
    current_merge_version: Optional[str] = None          # 当前merge版本ID

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理嵌套的dataclass对象
        data['prompt_versions'] = [v.to_dict() for v in self.prompt_versions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentEvolutionRecord':
        """从字典创建实例"""
        # 处理嵌套的dataclass对象
        prompt_versions = [PromptVersionRecord.from_dict(v) for v in data.get('prompt_versions', [])]
        data['prompt_versions'] = prompt_versions
        return cls(**data)


class AgentVersionManager:
    """
    Agent版本管理器

    核心功能：
    1. 记录每个Agent的prompt版本历史
    2. 关联版本与性能数据
    3. 记录反思建议和生成过程
    4. 支持数据持久化
    """

    # 类常量
    TEMPLATE_MAP = {
        "explore": "explore_user_prompt.j2",
        "merge": "merge_user_prompt.j2",
        "select": "select_user_prompt.j2",
        "review": "evaluate_user_prompt.j2"
    }
    ACCURACY_WEIGHT = 0.6                                # 准确率权重
    GENERATION_RATE_WEIGHT = 0.4                         # 生成率权重
    SLOW_IO_THRESHOLD = 0.1                              # 慢IO阈值（秒）

    def __init__(self, storage_dir: str = "workspace/agent_evolution", prompt_manager=None):
        """
        初始化Agent版本管理器

        参数:
            storage_dir: 存储目录路径
            prompt_manager: PromptManager实例，用于获取初始模板内容
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 异步安全的锁
        self.lock = asyncio.Lock()

        # PromptManager（用于获取初始模板）
        self.prompt_manager = prompt_manager

        # 内存中的Agent记录
        self.agent_records: Dict[str, AgentEvolutionRecord] = {}

        # 加载已存在的记录
        self._load_existing_records()

    def _load_existing_records(self):
        """加载已存在的Agent记录"""
        try:
            for record_file in self.storage_dir.glob("agent_*.json"):
                try:
                    with open(record_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    record = AgentEvolutionRecord.from_dict(data)
                    self.agent_records[record.agent_name] = record
                except (json.JSONDecodeError, KeyError) as e:
                    log_msg("WARNING", f"无法加载记录文件 {record_file}: {e}")
        except Exception as e:
            log_msg("WARNING", f"加载现有记录时出错: {e}")

    async def register_agent(self, agent_name: str) -> AgentEvolutionRecord:
        """
        注册一个新的Agent记录

        参数:
            agent_name: Agent名称

        返回:
            AgentEvolutionRecord对象
        """
        async with self.lock:
            if agent_name in self.agent_records:
                return self.agent_records[agent_name]

            now = datetime.now().isoformat()
            record = AgentEvolutionRecord(
                agent_name=agent_name,
                created_at=now,
                last_updated=now
            )

            self.agent_records[agent_name] = record
            self._save_agent_record(agent_name)

            return record

    async def record_prompt_version(
        self,
        agent_name: str,
        version_id: str,
        prompt_type: str,
        prompt_content: str,
        source: str = "generated",
        guidance: str = "",
        previous_version_id: Optional[str] = None,
        crossover_source: Optional[Dict[str, str]] = None
    ) -> PromptVersionRecord:
        """
        记录一个新的prompt版本

        参数:
            agent_name: Agent名称
            version_id: 版本唯一标识
            prompt_type: prompt类型 (explore/merge)
            prompt_content: prompt内容
            source: 版本来源 (initial/generated/manual/learned)
            guidance: 反思建议内容（包含分析、建议、优先级等）
            previous_version_id: 前一个版本ID
            crossover_source: 交叉来源信息 {"agent": "agent_1", "version_id": "explore_v2"}

        返回:
            PromptVersionRecord对象
        """
        # 确保Agent记录存在（在获取锁之前）
        if agent_name not in self.agent_records:
            await self.register_agent(agent_name)

        async with self.lock:
            record = PromptVersionRecord(
                version_id=version_id,
                prompt_type=prompt_type,
                prompt_content=prompt_content,
                created_at=datetime.now().isoformat(),
                source=source,
                guidance=guidance,
                previous_version_id=previous_version_id,
                crossover_source=crossover_source
            )

            # 添加到版本历史
            self.agent_records[agent_name].prompt_versions.append(record)
            self.agent_records[agent_name].last_updated = datetime.now().isoformat()

            # 更新当前版本
            self._set_current_version(agent_name, prompt_type, version_id)

            # 保存到文件
            self._save_agent_record(agent_name)

            return record

    async def record_prompt_usage(self, agent_name: str, prompt_type: str) -> bool:
        """
        记录prompt使用次数

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 (explore/merge)

        返回:
            是否成功记录
        """
        # 确保Agent记录存在（在获取锁之前）
        if agent_name not in self.agent_records:
            await self.register_agent(agent_name)

        async with self.lock:
            start = time.time()
            agent_record = self.agent_records[agent_name]

            # 获取当前版本ID
            current_version_id = self._get_current_version_id(agent_record, prompt_type)

            # 如果没有当前版本，创建初始版本
            if not current_version_id:
                current_version_id = await self._create_initial_version(agent_record, prompt_type)

            # 找到当前版本并增加使用次数
            for version in agent_record.prompt_versions:
                if version.version_id == current_version_id:
                    version.used_count += 1
                    agent_record.last_updated = datetime.now().isoformat()
                    self._save_agent_record(agent_name)
                    elapsed = time.time() - start
                    log_msg("DEBUG", f"record_prompt_usage完成 {agent_name}，耗时 {elapsed:.3f}秒")
                    return True

            return False

    async def record_task_execution(
        self,
        agent_name: str,
        prompt_type: str,
        task_id: str
    ) -> bool:
        """
        记录任务执行（用于没有生成节点的情况）

        创建一个空的TaskReviewRecord，node_metadata为空，
        task_score=0.0, has_submission=False

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 (explore/merge)
            task_id: 任务ID

        返回:
            是否成功记录
        """
        # 确保Agent记录存在
        if agent_name not in self.agent_records:
            await self.register_agent(agent_name)

        async with self.lock:
            agent_record = self.agent_records[agent_name]

            # 获取当前版本ID
            current_version_id = self._get_current_version_id(agent_record, prompt_type)
            if not current_version_id:
                current_version_id = await self._create_initial_version(agent_record, prompt_type)

            # 找到当前版本
            for version in agent_record.prompt_versions:
                if version.version_id == current_version_id:
                    # 检查是否已有该task_id的记录
                    for record in version.review_records:
                        if record.task_id == task_id:
                            # 已存在，不需要重复创建
                            return True

                    # 创建空的TaskReviewRecord
                    num = len(version.review_records) + 1
                    task_record = TaskReviewRecord(
                        num=num,
                        task_score=0.0,
                        has_submission=False,
                        task_id=task_id,
                        node_metadata=[]
                    )
                    version.review_records.append(task_record)

                    # 更新版本的综合评分
                    self._calculate_version_score(version)

                    # 保存
                    agent_record.last_updated = datetime.now().isoformat()
                    self._save_agent_record(agent_name)
                    log_msg("INFO", f"已记录空任务执行: {agent_name} - {prompt_type} - task_id={task_id}")
                    return True

            return False

    async def record_review_result(
        self,
        agent_name: str,
        prompt_type: str,
        task_id: str,
        node_id: str,
        score: Optional[float],
        has_submission: bool,
        version_id: Optional[str] = None
    ) -> bool:
        """
        记录review结果到对应的prompt版本

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 (explore/merge)
            task_id: 关联的任务ID（explore/merge任务）
            node_id: 关联的Node ID（被review的节点）
            score: review分数（可能为None）
            has_submission: 是否有提交
            version_id: 指定的版本ID（如果为None，使用当前版本）

        返回:
            是否成功记录
        """
        # 确保Agent记录存在（在获取锁之前）
        if agent_name not in self.agent_records:
            await self.register_agent(agent_name)

        async with self.lock:
            agent_record = self.agent_records[agent_name]

            # 使用指定的version_id，如果没有则使用当前版本ID
            target_version_id = version_id
            if not target_version_id:
                target_version_id = self._get_current_version_id(agent_record, prompt_type)

            # 如果没有目标版本，创建初始版本
            if not target_version_id:
                target_version_id = await self._create_initial_version(agent_record, prompt_type)

            # 找到目标版本并添加review记录
            for version in agent_record.prompt_versions:
                if version.version_id == target_version_id:
                    # 查找是否已有该task_id的记录
                    task_record = None
                    for record in version.review_records:
                        if record.task_id == task_id:
                            task_record = record
                            break

                    # 如果没有该task_id的记录，创建新的TaskReviewRecord
                    if not task_record:
                        num = len(version.review_records) + 1
                        task_record = TaskReviewRecord(
                            num=num,
                            task_score=0.0,
                            has_submission=False,
                            task_id=task_id
                        )
                        version.review_records.append(task_record)

                    # 创建NodeMetadata并添加到任务记录
                    node_metadata = NodeMetadata(
                        score=score,
                        has_submission=has_submission,
                        timestamp=datetime.now().isoformat(),
                        node_id=node_id
                    )
                    task_record.node_metadata.append(node_metadata)

                    # 更新任务级别的聚合指标
                    self._update_task_record_metrics(task_record)

                    # 更新版本的综合评分
                    self._calculate_version_score(version)

                    # 更新时间戳并保存
                    agent_record.last_updated = datetime.now().isoformat()
                    self._save_agent_record(agent_name)
                    return True

            log_msg("WARNING", f"未找到目标版本 {target_version_id} for {agent_name}")
            return False

    def _calculate_version_score(self, version: PromptVersionRecord) -> None:
        """
        计算版本的综合评分

        综合评分 = ACCURACY_WEIGHT × 平均准确率 + GENERATION_RATE_WEIGHT × 平均生成率

        现在基于任务级别（TaskReviewRecord）计算
        """
        if version.used_count == 0:
            version.composite_score = 0.0
            version.avg_generation_rate = 0.0
            version.avg_accuracy = 0.0
            return

        # 计算平均生成率（基于任务级别）
        submission_count = sum(1 for task_record in version.review_records if task_record.has_submission)
        version.avg_generation_rate = submission_count / version.used_count

        # 计算平均准确率（基于任务级别的task_score）
        valid_task_scores = [task_record.task_score for task_record in version.review_records if task_record.task_score is not None]
        if valid_task_scores:
            version.avg_accuracy = sum(valid_task_scores) / len(valid_task_scores)
        else:
            version.avg_accuracy = 0.0

        # 计算综合评分
        version.composite_score = (
            self.ACCURACY_WEIGHT * version.avg_accuracy +
            self.GENERATION_RATE_WEIGHT * version.avg_generation_rate
        )

    def _update_task_record_metrics(self, task_record: TaskReviewRecord) -> None:
        """
        更新任务记录的聚合指标

        参数:
            task_record: 要更新的任务记录

        更新:
            task_score: node_metadata中所有score的平均值
            has_submission: node_metadata中任一has_submission为true则为true
        """
        if not task_record.node_metadata:
            task_record.task_score = 0.0
            task_record.has_submission = False
            return

        # 计算task_score（所有node_metadata的score平均值）
        valid_scores = [nm.score for nm in task_record.node_metadata if nm.score is not None]
        if valid_scores:
            task_record.task_score = sum(valid_scores) / len(valid_scores)
        else:
            task_record.task_score = 0.0

        # 计算has_submission（任一为true则为true）
        task_record.has_submission = any(nm.has_submission for nm in task_record.node_metadata)

    async def update_version_guidance(
        self,
        agent_name: str,
        version_id: str,
        guidance: str
    ) -> bool:
        """
        更新版本的guidance（反思建议）

        参数:
            agent_name: Agent名称
            version_id: 版本ID
            guidance: 反思建议内容

        返回:
            是否成功更新
        """
        async with self.lock:
            agent_record = self.agent_records.get(agent_name)
            if not agent_record:
                return False

            for version in agent_record.prompt_versions:
                if version.version_id == version_id:
                    version.guidance = guidance
                    agent_record.last_updated = datetime.now().isoformat()
                    self._save_agent_record(agent_name)
                    return True

            return False

    def get_agent_record(self, agent_name: str) -> Optional[AgentEvolutionRecord]:
        """
        获取Agent的完整进化记录

        参数:
            agent_name: Agent名称

        返回:
            AgentEvolutionRecord对象，如果不存在则返回None
        """
        return self.agent_records.get(agent_name)

    def get_current_prompt(self, agent_name: str, prompt_type: str) -> Optional[PromptVersionRecord]:
        """
        获取Agent当前的prompt版本

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 (explore/merge)

        返回:
            PromptVersionRecord对象，如果不存在则返回None
        """
        agent_record = self.agent_records.get(agent_name)
        if not agent_record:
            return None

        version_id = self._get_current_version_id(agent_record, prompt_type)
        if not version_id:
            return None

        # 查找对应的版本
        for version in agent_record.prompt_versions:
            if version.version_id == version_id:
                return version

        return None

    def get_current_prompt_usage_count(self, agent_name: str, prompt_type: str) -> int:
        """
        获取当前prompt的使用次数

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 (explore/merge)

        返回:
            使用次数，如果不存在返回0
        """
        current_version = self.get_current_prompt(agent_name, prompt_type)
        if not current_version:
            return 0

        return current_version.used_count

    def should_trigger_reflection(
        self,
        agent_name: str,
        prompt_type: str,
        threshold: int = 5
    ) -> bool:
        """
        检查是否应该触发反思

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 (explore/merge)
            threshold: 触发反思的使用次数阈值

        返回:
            是否应该触发反思
        """
        usage_count = self.get_current_prompt_usage_count(agent_name, prompt_type)
        return usage_count >= threshold

    def get_all_agent_names(self) -> List[str]:
        """
        获取所有Agent的名称列表

        返回:
            Agent名称列表
        """
        return list(self.agent_records.keys())

    # ==================== 私有辅助方法 ====================

    def _get_current_version_id(self, agent_record: AgentEvolutionRecord, prompt_type: str) -> Optional[str]:
        """
        获取Agent当前版本的ID

        参数:
            agent_record: Agent进化记录
            prompt_type: prompt类型 (explore/merge)

        返回:
            当前版本ID，如果不存在则返回None
        """
        if prompt_type == "explore":
            return agent_record.current_explore_version
        elif prompt_type == "merge":
            return agent_record.current_merge_version
        return None

    def _set_current_version(self, agent_name: str, prompt_type: str, version_id: str) -> None:
        """
        设置Agent当前版本

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 (explore/merge)
            version_id: 版本ID
        """
        if prompt_type == "explore":
            self.agent_records[agent_name].current_explore_version = version_id
        elif prompt_type == "merge":
            self.agent_records[agent_name].current_merge_version = version_id

    async def _create_initial_version(self, agent_record: AgentEvolutionRecord, prompt_type: str) -> str:
        """
        创建初始prompt版本

        参数:
            agent_record: Agent进化记录
            prompt_type: prompt类型 (explore/merge)

        返回:
            创建的版本ID
        """
        version_id = f"{prompt_type}_v1"

        # 从 PromptManager 获取初始模板内容
        initial_prompt_content = ""
        if self.prompt_manager:
            try:
                template_name = self.TEMPLATE_MAP.get(prompt_type, "explore_user_prompt.j2")
                initial_prompt_content = self.prompt_manager.get_template(template_name)
                log_msg("DEBUG", f"从模板 {template_name} 加载初始prompt内容，长度: {len(initial_prompt_content)}")
            except Exception as e:
                log_msg("WARNING", f"无法从PromptManager获取模板 {prompt_type}: {e}")

        current_version = PromptVersionRecord(
            version_id=version_id,
            prompt_type=prompt_type,
            prompt_content=initial_prompt_content,
            created_at=datetime.now().isoformat(),
            source="initial",
            used_count=0
        )
        agent_record.prompt_versions.append(current_version)

        # 更新当前版本
        self._set_current_version(agent_record.agent_name, prompt_type, version_id)

        return version_id

    def _save_agent_record(self, agent_name: str):
        """
        保存Agent记录到文件

        参数:
            agent_name: Agent名称
        """
        if agent_name not in self.agent_records:
            return

        record = self.agent_records[agent_name]
        filename = f"agent_{agent_name}.json"
        filepath = self.storage_dir / filename

        try:
            start = time.time()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
            elapsed = time.time() - start
            if elapsed > self.SLOW_IO_THRESHOLD:
                log_msg("WARNING", f"保存Agent记录耗时 {elapsed:.2f}秒: {agent_name}")
        except Exception as e:
            log_msg("ERROR", f"保存Agent记录失败 {agent_name}: {e}")
