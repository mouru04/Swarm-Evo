"""
优化模块公共工具

提供LLM响应解析、消息构建等公共功能
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage
from utils.logger_system import log_msg


class LLMResponseParser:
    """LLM响应解析器"""

    @staticmethod
    def extract_json_from_response(response_content: str) -> Optional[Dict[str, Any]]:
        """
        从LLM响应中提取JSON数据

        支持以下格式：
        1. ```json ... ```
        2. ``` ... ```
        3. 纯JSON

        参数:
            response_content: LLM响应内容

        返回:
            解析后的字典，解析失败返回None
        """
        try:
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_str = response_content[json_start:json_end].strip()
                return json.loads(json_str)
            elif "```" in response_content:
                json_start = response_content.find("```") + 3
                json_end = response_content.find("```", json_start)
                json_str = response_content[json_start:json_end].strip()
                return json.loads(json_str)
            else:
                return json.loads(response_content)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def format_response_content(response) -> str:
        """
        格式化LLM响应内容为字符串

        参数:
            response: LangChain响应对象

        返回:
            字符串格式的响应内容
        """
        return (
            response.content if isinstance(response.content, str)
            else str(response.content)
        )


class MessageBuilder:
    """LLM消息构建器"""

    @staticmethod
    def build_llm_messages(
        template_content: str,
        template_vars: Dict[str, Any],
        human_message: str
    ) -> List:
        """
        构建标准的LLM消息

        参数:
            template_content: 模板内容
            template_vars: 模板变量
            human_message: 人类消息

        返回:
            包含SystemMessage和HumanMessage的列表
        """
        template = Template(template_content)
        system_message = template.render(**template_vars)

        return [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]


class FileSaver:
    """文件保存工具"""

    @staticmethod
    def save_result_to_json(
        result: Dict[str, Any],
        filename: str,
        storage_dir: str = "workspace/logs",
        result_type: str = "result"
    ) -> bool:
        """
        保存结果到JSON文件

        参数:
            result: 结果字典
            filename: 文件名
            storage_dir: 存储目录
            result_type: 结果类型（用于日志）

        返回:
            是否保存成功
        """
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)

        filepath = storage_path / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            log_msg("INFO", f"{result_type}结果已保存到: {filepath}")
            return True
        except Exception as e:
            log_msg("ERROR", f"保存{result_type}结果失败: {e}")
            return False
