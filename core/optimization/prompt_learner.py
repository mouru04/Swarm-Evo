"""
Prompt学习器模块

功能：
1. 识别高分Agent和低分Agent
2. 让低分Agent向高分Agent学习prompt（使用LLM）
3. 保留低分Agent的自身特色，体现差异性
4. 记录学习历史和效果
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from core.optimization.version_manager import (
    AgentVersionManager,
    PromptVersionRecord
)
from core.optimization.utils import LLMResponseParser, MessageBuilder, FileSaver


@dataclass
class LearningCandidate:
    """学习候选对"""
    student_agent: str                    # 学习者（低分Agent）
    teacher_agent: str                    # 被学习者（高分Agent）
    prompt_type: str                      # prompt类型 (explore/merge)
    student_score: float                  # 学生得分（综合评分）
    teacher_score: float                  # 老师得分（综合评分）
    score_gap: float                      # 分数差距
    student_prompt_id: Optional[str]      # 学生当前prompt ID
    teacher_prompt_id: Optional[str]      # 老师当前prompt ID
    student_version: Optional[PromptVersionRecord] = None  # 学生完整版本记录
    teacher_version: Optional[PromptVersionRecord] = None  # 老师完整版本记录


@dataclass
class LearningResult:
    """学习结果"""
    success: bool                         # 是否成功学习
    student_agent: str                    # 学习者
    teacher_agent: str                    # 被学习者
    prompt_type: str                      # prompt类型
    old_prompt_id: str                    # 原prompt ID
    new_prompt_id: str                    # 新prompt ID
    old_score: float                      # 学习前分数
    timestamp: str                        # 学习时间
    reasoning: str                        # 学习理由（LLM生成）
    error: Optional[str] = None           # 错误信息
    llm_response: Optional[str] = None    # 完整的LLM响应（用于调试）


class PromptLearner:
    """
    Prompt学习器

    实现Agent之间的Prompt学习和进化，基于遗传算法思想。

    核心功能：
    1. 分析Agent性能，识别学习候选对
    2. 执行学习操作（替换或交叉）
    3. 记录学习历史
    4. 追踪学习效果
    """

    def __init__(
        self,
        version_manager: AgentVersionManager,
        llm: BaseChatModel,
        prompt_manager: Any,
        learning_threshold: float = 0.1,     # 学习阈值（分数差距）
        min_episodes: int = 3,                # 最少执行次数才开始学习
        storage_dir: str = "workspace/prompt_learning"
    ):
        """
        初始化Prompt学习器

        参数:
            version_manager: 版本管理器
            llm: LangChain语言模型
            prompt_manager: Prompt管理器
            learning_threshold: 学习阈值（分数差距超过此值才学习）
            min_episodes: 最少执行次数才开始学习
            storage_dir: 学习历史存储目录
        """
        self.version_manager = version_manager
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.learning_threshold = learning_threshold
        self.min_episodes = min_episodes

        # 存储目录
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 学习历史
        self.learning_history: List[LearningResult] = []

    def analyze_learning_opportunities(
        self,
        prompt_type: str,
        min_agents: int = 2
    ) -> List[LearningCandidate]:
        """
        分析学习机会，识别可以学习的候选对

        基于每个prompt版本的综合评分来识别学习机会。

        参数:
            prompt_type: prompt类型 (explore/merge)
            min_agents: 最少Agent数量（少于则不学习）

        返回:
            学习候选对列表，按分数差距降序排序
        """
        # 获取所有Agent名称
        all_agent_names = self.version_manager.get_all_agent_names()

        # 过滤掉没有当前版本的Agent
        eligible_agents = []
        for agent_name in all_agent_names:
            current_prompt = self.version_manager.get_current_prompt(agent_name, prompt_type)
            if current_prompt and current_prompt.used_count >= self.min_episodes:
                eligible_agents.append(agent_name)

        if len(eligible_agents) < min_agents:
            return []

        # 提取每个Agent的当前prompt版本的综合评分
        agent_scores = []
        for agent_name in eligible_agents:
            # 获取当前prompt版本
            current_prompt = self.version_manager.get_current_prompt(agent_name, prompt_type)
            if not current_prompt:
                continue

            # 使用综合评分
            score = current_prompt.composite_score

            agent_scores.append({
                'agent_name': agent_name,
                'score': score,
                'prompt_id': current_prompt.version_id,
                'prompt_version': current_prompt,
                'total_tasks': current_prompt.used_count
            })

        # 生成所有可能的学习候选对（低分 -> 高分）
        candidates = []

        for student in agent_scores:
            for teacher in agent_scores:
                if student['agent_name'] == teacher['agent_name']:
                    continue

                score_gap = teacher['score'] - student['score']

                # 只有分数差距超过阈值才考虑学习
                if score_gap >= self.learning_threshold:
                    candidate = LearningCandidate(
                        student_agent=student['agent_name'],
                        teacher_agent=teacher['agent_name'],
                        prompt_type=prompt_type,
                        student_score=student['score'],
                        teacher_score=teacher['score'],
                        score_gap=score_gap,
                        student_prompt_id=student['prompt_id'],
                        teacher_prompt_id=teacher['prompt_id'],
                        student_version=student['prompt_version'],
                        teacher_version=teacher['prompt_version']
                    )
                    candidates.append(candidate)

        # 按分数差距降序排序
        candidates.sort(key=lambda x: x.score_gap, reverse=True)

        return candidates

    def select_best_learning_candidate(
        self,
        candidates: List[LearningCandidate]
    ) -> Optional[LearningCandidate]:
        """
        从候选对中选择最佳学习对象

        策略：选择分数差距最大的候选对

        参数:
            candidates: 学习候选对列表

        返回:
            最佳学习候选，如果没有则返回None
        """
        if not candidates:
            return None

        return candidates[0]  # 已排序，第一个就是最佳

    async def execute_learning(
        self,
        candidate: LearningCandidate
    ) -> LearningResult:
        """
        执行学习操作（使用LLM）

        让低分Agent的prompt向高分Agent的prompt学习，但保留自身特色。

        参数:
            candidate: 学习候选对

        返回:
            学习结果
        """
        timestamp = datetime.now().isoformat()

        try:
            # 验证候选对
            if not candidate.teacher_version or not candidate.student_version:
                return LearningResult(
                    success=False,
                    student_agent=candidate.student_agent,
                    teacher_agent=candidate.teacher_agent,
                    prompt_type=candidate.prompt_type,
                    old_prompt_id=candidate.student_prompt_id or "",
                    new_prompt_id="",
                    old_score=candidate.student_score,
                    timestamp=timestamp,
                    reasoning="",
                    error="Missing version information"
                )

            # 使用LLM执行学习
            new_prompt_content, reasoning, llm_response = await self._learn_with_llm(candidate)

            # 应用新prompt
            template_name = f"{candidate.prompt_type}_user_prompt.j2"
            self.prompt_manager.set_template(template_name, new_prompt_content)

            # 记录新版本到version_manager
            new_version_id = f"{candidate.prompt_type}_learned_from_{candidate.teacher_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            await self.version_manager.record_prompt_version(
                agent_name=candidate.student_agent,
                version_id=new_version_id,
                prompt_type=candidate.prompt_type,
                prompt_content=new_prompt_content,
                source="learned",
                guidance=f"从 {candidate.teacher_agent} 学习\n\n生成理由: {reasoning}",
                previous_version_id=candidate.student_prompt_id,
                crossover_source={
                    "agent": candidate.teacher_agent,
                    "version_id": candidate.teacher_prompt_id
                }
            )

            # 创建学习结果
            result = LearningResult(
                success=True,
                student_agent=candidate.student_agent,
                teacher_agent=candidate.teacher_agent,
                prompt_type=candidate.prompt_type,
                old_prompt_id=candidate.student_prompt_id or "",
                new_prompt_id=new_version_id,
                old_score=candidate.student_score,
                timestamp=timestamp,
                reasoning=reasoning,
                llm_response=llm_response
            )

            # 保存到历史
            self.learning_history.append(result)
            self._save_learning_result(result)

            return result

        except Exception as e:
            result = LearningResult(
                success=False,
                student_agent=candidate.student_agent,
                teacher_agent=candidate.teacher_agent,
                prompt_type=candidate.prompt_type,
                old_prompt_id=candidate.student_prompt_id or "",
                new_prompt_id="",
                old_score=candidate.student_score,
                timestamp=timestamp,
                reasoning="",
                error=str(e)
            )
            self.learning_history.append(result)
            return result

    async def _learn_with_llm(
        self,
        candidate: LearningCandidate
    ) -> Tuple[str, str, str]:
        """
        使用LLM执行学习，生成融合后的prompt

        LLM会分析学生和老师的prompt，生成一个新prompt：
        - 吸收老师的优点
        - 保留学生的自身特色（体现agent差异性）

        返回:
            (新prompt内容, 学习理由, 完整LLM响应)
        """
        # 获取学习模板
        template_content = self.prompt_manager.get_template("learner_prompt.j2")

        # 准备模板变量
        template_vars = {
            "student_agent": candidate.student_agent,
            "teacher_agent": candidate.teacher_agent,
            "prompt_type": candidate.prompt_type,
            "student_score": candidate.student_score,
            "teacher_score": candidate.teacher_score,
            "score_gap": candidate.score_gap,
            "student_prompt": candidate.student_version.prompt_content if candidate.student_version else "",
            "teacher_prompt": candidate.teacher_version.prompt_content,
            "student_guidance": candidate.student_version.guidance if candidate.student_version else "",
            "teacher_guidance": candidate.teacher_version.guidance if candidate.teacher_version else "",
            "student_metrics": {
                "avg_accuracy": candidate.student_version.avg_accuracy if candidate.student_version else 0.0,
                "avg_generation_rate": candidate.student_version.avg_generation_rate if candidate.student_version else 0.0,
                "composite_score": candidate.student_score
            },
            "teacher_metrics": {
                "avg_accuracy": candidate.teacher_version.avg_accuracy if candidate.teacher_version else 0.0,
                "avg_generation_rate": candidate.teacher_version.avg_generation_rate if candidate.teacher_version else 0.0,
                "composite_score": candidate.teacher_score
            }
        }

        # 构建消息
        messages = MessageBuilder.build_llm_messages(
            template_content=template_content,
            template_vars=template_vars,
            human_message="请生成学习后的新prompt。"
        )

        try:
            # 调用LLM
            response = await self.llm.ainvoke(messages)
            response_content = LLMResponseParser.format_response_content(response)

            # 解析响应
            new_prompt, reasoning = self._parse_learning_response(response_content)

            return new_prompt, reasoning, response_content

        except Exception as e:
            # 如果LLM调用失败，使用简单的融合策略
            fallback_prompt = self._fallback_learning(candidate)
            error_msg = f"LLM调用失败，使用备用策略: {str(e)}"
            return fallback_prompt, error_msg, error_msg

    def _parse_learning_response(self, response_content: str) -> Tuple[str, str]:
        """
        解析LLM的学习响应

        返回:
            (新prompt内容, 学习理由)
        """
        data = LLMResponseParser.extract_json_from_response(response_content)

        if data:
            new_prompt = data.get("new_prompt", response_content)
            reasoning = data.get("reasoning", "")

            # 如果 reasoning 为空，尝试用其他字段生成
            if not reasoning:
                gap_analysis = data.get("gap_analysis", "")
                integration_strategy = data.get("integration_strategy", "")
                if gap_analysis or integration_strategy:
                    reasoning = f"Gap: {gap_analysis}. Strategy: {integration_strategy}"
                else:
                    reasoning = "Learning completed (no detailed reasoning provided)"

            return new_prompt, reasoning
        else:
            # 如果不是JSON，记录警告并使用响应内容
            log_msg("WARNING", f"LLM did not return JSON format. Response preview: {response_content[:200]}...")
            return response_content, "Failed to parse reasoning (LLM response not in JSON format)"

    def _fallback_learning(
        self,
        candidate: LearningCandidate
    ) -> str:
        """
        备用学习策略（当LLM调用失败时）

        简单地使用老师的prompt
        """
        if candidate.teacher_version:
            return candidate.teacher_version.prompt_content
        return ""

    def get_learning_history(
        self,
        agent_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[LearningResult]:
        """
        获取学习历史

        参数:
            agent_name: Agent名称（None表示获取所有）
            limit: 限制返回数量

        返回:
            学习历史列表
        """
        history = self.learning_history

        if agent_name:
            history = [
                record for record in history
                if record.student_agent == agent_name
            ]

        if limit:
            history = history[-limit:]

        return history

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        获取学习统计信息

        返回:
            统计信息字典
        """
        total_learning = len(self.learning_history)
        successful_learning = len([r for r in self.learning_history if r.success])

        # 按Agent统计
        agent_learning_counts = {}
        for record in self.learning_history:
            agent = record.student_agent
            agent_learning_counts[agent] = agent_learning_counts.get(agent, 0) + 1

        return {
            "total_learning_events": total_learning,
            "successful_learning": successful_learning,
            "success_rate": successful_learning / total_learning if total_learning > 0 else 0,
            "agent_learning_counts": agent_learning_counts,
            "learning_threshold": self.learning_threshold
        }

    def _save_learning_result(self, result: LearningResult) -> None:
        """
        保存学习结果到文件

        参数:
            result: 学习结果
        """
        filename = f"learning_{result.student_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        result_dict = {
            "success": result.success,
            "student_agent": result.student_agent,
            "teacher_agent": result.teacher_agent,
            "prompt_type": result.prompt_type,
            "old_prompt_id": result.old_prompt_id,
            "new_prompt_id": result.new_prompt_id,
            "old_score": result.old_score,
            "timestamp": result.timestamp,
            "reasoning": result.reasoning,
            "error": result.error
        }

        # 如果有完整的LLM响应，保存它（用于调试）
        if result.llm_response:
            result_dict["llm_response"] = result.llm_response

        FileSaver.save_result_to_json(
            result=result_dict,
            filename=filename,
            storage_dir=str(self.storage_dir),
            result_type="学习"
        )
