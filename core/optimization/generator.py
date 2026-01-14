"""
生成器模块 - 用于根据建议生成新版本的prompt

功能：
1. 接收当前prompt内容和改进建议
2. 使用LLM生成新版本的prompt
3. 保存新生成的prompt版本
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from langchain_core.language_models import BaseChatModel
from core.agent.prompt_manager import PromptManager
from core.optimization.version_manager import AgentVersionManager
from core.optimization.utils import LLMResponseParser, MessageBuilder
from utils.logger_system import log_msg


@dataclass
class GenerationResult:
    """生成结果"""
    success: bool  # 是否成功生成
    new_prompt: str  # 新生成的prompt内容
    version: str  # 版本号
    changes_made: List[str]  # 所做的修改列表
    reasoning: str  # 修改理由
    original_prompt: str  # 原始prompt
    suggestions_used: List[str]  # 使用的建议列表
    error: Optional[str] = None  # 错误信息（如果生成失败）
    parse_error: Optional[str] = None  # 解析错误信息


class PromptGenerator:
    """
    Prompt生成器

    根据反思器的建议生成新版本的prompt
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_manager: PromptManager,
        version_manager: AgentVersionManager
    ):
        """
        初始化生成器

        参数:
            llm: LangChain语言模型
            prompt_manager: Prompt管理器
            version_manager: 版本管理器
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.version_manager = version_manager

    async def generate_new_prompt(
        self,
        agent_name: str,
        prompt_type: str,
        current_prompt: str,
        suggestions: List[str],
        analysis: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        根据建议生成新版本的prompt

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型 ("explore" 或 "merge")
            current_prompt: 当前prompt内容
            suggestions: 改进建议列表
            analysis: 反思分析结果
            additional_context: 额外的上下文信息

        返回:
            GenerationResult对象
        """
        # 使用 version_manager 生成新版本号
        new_version = self._generate_version(agent_name, prompt_type)

        # 构建生成消息
        messages = self._build_generation_messages(
            prompt_type=prompt_type,
            current_prompt=current_prompt,
            suggestions=suggestions,
            analysis=analysis,
            additional_context=additional_context or {}
        )

        try:
            # 调用LLM生成新prompt
            response = await self.llm.ainvoke(messages)
            response_content = LLMResponseParser.format_response_content(response)

            # 解析生成结果
            generation_result = self._parse_generation_response(
                response_content, current_prompt, new_version, suggestions
            )

            return generation_result

        except Exception as e:
            return GenerationResult(
                success=False,
                new_prompt="",
                version=new_version,
                changes_made=[],
                reasoning="",
                original_prompt=current_prompt,
                suggestions_used=suggestions,
                error=str(e)
            )

    def _generate_version(self, agent_name: str, prompt_type: str) -> str:
        """
        生成新版本号（使用 version_manager）

        参数:
            agent_name: Agent名称
            prompt_type: prompt类型

        返回:
            版本号字符串
        """
        # 获取现有版本数量
        agent_record = self.version_manager.get_agent_record(agent_name)
        if agent_record:
            version_count = len([
                v for v in agent_record.prompt_versions
                if v.prompt_type == prompt_type
            ])
        else:
            version_count = 0

        # 生成版本号
        version_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prompt_type}_v{version_count}_{timestamp}"

    def _build_generation_messages(
        self,
        prompt_type: str,
        current_prompt: str,
        suggestions: List[str],
        analysis: str,
        additional_context: Dict[str, Any]
    ) -> List:
        """
        构建用于LLM生成的消息
        """
        # 获取生成模板
        template_content = self.prompt_manager.get_template("generator_prompt.j2")

        # 准备模板变量
        template_vars = {
            "prompt_type": prompt_type,
            "current_prompt": current_prompt,
            "suggestions": suggestions,
            "analysis": analysis,
            "additional_context": additional_context
        }

        return MessageBuilder.build_llm_messages(
            template_content=template_content,
            template_vars=template_vars,
            human_message="请根据以上分析建议，生成改进后的prompt版本。"
        )

    def _parse_generation_response(
        self,
        response_content: str,
        original_prompt: str,
        version: str,
        suggestions: List[str]
    ) -> GenerationResult:
        """
        解析LLM的生成响应
        """
        data = LLMResponseParser.extract_json_from_response(response_content)

        if data:
            return GenerationResult(
                success=data.get("success", True),
                new_prompt=data.get("new_prompt", ""),
                version=version,
                changes_made=data.get("changes_made", []),
                reasoning=data.get("reasoning", ""),
                original_prompt=original_prompt,
                suggestions_used=suggestions
            )
        else:
            # 如果JSON解析失败，尝试提取文本内容
            return GenerationResult(
                success=True,
                new_prompt=response_content,
                version=version,
                changes_made=["基于LLM直接生成"],
                reasoning="根据分析建议生成新版本",
                original_prompt=original_prompt,
                suggestions_used=suggestions,
                parse_error="JSON解析失败"
            )

    async def apply_new_prompt(
        self,
        prompt_type: str,
        new_prompt: str,
        version: str  # 保留参数以保持接口兼容性
    ) -> bool:
        """
        应用新生成的prompt，更新到prompt_manager

        参数:
            prompt_type: prompt类型
            new_prompt: 新的prompt内容
            version: 版本号（未使用，保留以保持接口兼容性）

        返回:
            是否成功应用
        """
        try:
            template_name = f"{prompt_type}_user_prompt.j2"
            self.prompt_manager.set_template(
                template_name, new_prompt
            )
            return True

        except Exception as e:
            log_msg("ERROR", f"应用新prompt失败: {e}")
            return False
