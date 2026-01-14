"""
反思器模块 - 用于分析和改进explore/merge prompt

功能：
1. 收集特定prompt版本的执行结果
2. 计算平均准确率和平均生成率
3. 使用LLM分析prompt效果并提供改进建议
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from core.agent.prompt_manager import PromptManager
from core.optimization.version_manager import PromptVersionRecord
from core.optimization.utils import LLMResponseParser, MessageBuilder, FileSaver


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_accuracy: float  # 平均准确率
    avg_generation_rate: float  # 平均生成率
    total_reviews: int  # 总review数
    successful_reviews: int  # 有分数的review数
    submission_count: int  # 有submission的数量


class PromptReflector:
    """
    Prompt反思器

    分析特定prompt的执行效果，并提供改进建议
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_manager: PromptManager
    ):
        """
        初始化反思器

        参数:
            llm: LangChain语言模型
            prompt_manager: Prompt管理器
        """
        self.llm = llm
        self.prompt_manager = prompt_manager

    async def reflect_on_prompt(
        self,
        prompt_type: str,
        prompt_content: str,
        metrics: PerformanceMetrics,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        对prompt进行反思，生成改进建议

        参数:
            prompt_type: prompt类型 ("explore" 或 "merge")
            prompt_content: 当前prompt内容
            metrics: 性能指标
            additional_context: 额外的上下文信息

        返回:
            包含改进建议的字典
        """
        # 构建反思消息
        messages = self._build_reflection_messages(
            prompt_type=prompt_type,
            prompt_content=prompt_content,
            metrics=metrics,
            additional_context=additional_context or {}
        )

        try:
            # 调用LLM生成反思建议
            response = await self.llm.ainvoke(messages)
            response_content = LLMResponseParser.format_response_content(response)

            # 解析响应
            reflection_result = self._parse_reflection_response(response_content)

            return reflection_result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "suggestions": [],
                "analysis": "反思器调用失败"
            }

    def _build_reflection_messages(
        self,
        prompt_type: str,
        prompt_content: str,
        metrics: PerformanceMetrics,
        additional_context: Dict[str, Any]
    ) -> List:
        """
        构建用于LLM反思的消息
        """
        # 获取反思模板
        template_content = self.prompt_manager.get_template("reflector_prompt.j2")

        # 准备模板变量
        success_rate = (
            metrics.successful_reviews / metrics.total_reviews
            if metrics.total_reviews > 0 else 0
        )
        template_vars = {
            "prompt_type": prompt_type,
            "prompt_content": prompt_content,
            "avg_accuracy": metrics.avg_accuracy,
            "avg_generation_rate": metrics.avg_generation_rate,
            "total_reviews": metrics.total_reviews,
            "successful_reviews": metrics.successful_reviews,
            "submission_count": metrics.submission_count,
            "success_rate": success_rate,
            "additional_context": additional_context
        }

        return MessageBuilder.build_llm_messages(
            template_content=template_content,
            template_vars=template_vars,
            human_message="请基于以上数据分析这个prompt的表现，并提供具体的改进建议。"
        )

    def _parse_reflection_response(self, response_content: str) -> Dict[str, Any]:
        """
        解析LLM的反思响应
        """
        data = LLMResponseParser.extract_json_from_response(response_content)

        if data:
            return data
        else:
            # 如果JSON解析失败，返回文本内容
            return {
                "success": True,
                "suggestions": [],
                "analysis": response_content,
                "raw_response": response_content,
                "parse_error": "JSON解析失败"
            }

    async def analyze_version(
        self,
        version_record: PromptVersionRecord,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        分析特定prompt版本的执行效果，生成改进建议（guidance）

        这是核心方法，用于反思器-生成器循环。

        参数:
            version_record: prompt版本记录（包含review_records）
            additional_context: 额外的上下文信息

        返回:
            包含guidance和详细分析的结果字典
        """
        # 第一阶段: 从版本记录中提取review结果
        review_results = []
        for review in version_record.review_records:
            review_results.append({
                'num': review.num,
                'score': review.score,
                'has_submission': review.has_submission,
                'timestamp': review.timestamp,
                'node_id': review.node_id,
                'task_id': review.task_id
            })

        # 第二阶段: 使用版本记录中已计算的指标
        metrics_from_version = PerformanceMetrics(
            avg_accuracy=version_record.avg_accuracy,
            avg_generation_rate=version_record.avg_generation_rate,
            total_reviews=version_record.used_count,
            successful_reviews=len([r for r in version_record.review_records if r.score is not None]),
            submission_count=len([r for r in version_record.review_records if r.has_submission])
        )

        # 第三阶段: 进行反思分析
        reflection_result = await self.reflect_on_prompt(
            prompt_type=version_record.prompt_type,
            prompt_content=version_record.prompt_content,
            metrics=metrics_from_version,
            additional_context={
                "review_results": review_results,
                "version_id": version_record.version_id,
                **(additional_context or {})
            }
        )

        # 第四阶段: 组合完整结果
        result = {
            "version_id": version_record.version_id,
            "prompt_type": version_record.prompt_type,
            "metrics": {
                "avg_accuracy": metrics_from_version.avg_accuracy,
                "avg_generation_rate": metrics_from_version.avg_generation_rate,
                "composite_score": version_record.composite_score,
                "total_reviews": metrics_from_version.total_reviews,
                "successful_reviews": metrics_from_version.successful_reviews,
                "submission_count": metrics_from_version.submission_count
            },
            "review_records": review_results,
            "reflection": reflection_result,
            "current_prompt": version_record.prompt_content
        }

        # 第五阶段: 保存反思结果到文件
        self._save_reflection_result(result)

        return result

    def _save_reflection_result(self, result: Dict[str, Any]) -> None:
        """
        保存反思结果到文件

        参数:
            result: 反思结果字典
        """
        # 生成文件名：reflection_{prompt_type}_{timestamp}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_type = result["prompt_type"]
        filename = f"reflection_{prompt_type}_{timestamp}.json"

        # 使用公共工具保存
        logs_dir = "workspace/logs"
        FileSaver.save_result_to_json(
            result=result,
            filename=filename,
            storage_dir=logs_dir,
            result_type="反思"
        )
