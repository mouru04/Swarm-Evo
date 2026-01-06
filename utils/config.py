"""
环境变量配置管理模块

本模块负责读取.env文件中的环境变量，并提供统一的配置访问接口。
使用单例模式确保配置在整个应用中保持一致。
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from utils.logger_system import log_msg


class Config:
    """
    环境变量配置类

    负责加载和管理所有环境变量配置，包括：
    - LLM API配置
    - 日志配置
    - 实验配置
    - MLE-bench配置
    - Conda环境配置

    使用方式:
        from utils.config import get_config
        config = get_config()
        api_key = config.api_key
        conda_env = config.conda_env_name
    """

    _instance: Optional['Config'] = None

    def __new__(cls) -> 'Config':
        """
        单例模式实现

        参数:
            cls: 当前类对象

        返回:
            Config: 单例配置实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        初始化配置，加载环境变量

        参数:
            无

        返回:
            None
        """
        if self._initialized:
            return

        # 加载.env文件
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # 第一阶段：LLM API配置
        self.api_key = self._get_required_env('API_KEY')
        self.api_base = self._get_required_env('API_BASE')
        self.model_name = self._get_required_env('MODEL_NAME')

        # 第二阶段：日志配置
        self.log_level = self._get_required_env('LOG_LEVEL')
        self.log_dir = self._get_required_env('LOG_DIR')

        # 第三阶段：实验配置
        self.random_seed = self._get_required_int_env('RANDOM_SEED')
        self.max_concurrent_agents = self._get_optional_int_env('MAX_CONCURRENT_AGENTS', 1)
        self.agent_timeout = self._get_required_int_env('AGENT_TIMEOUT')
        self.max_retries = self._get_required_int_env('MAX_RETRIES')
        self.agent_num = self._get_required_int_env('AGENT_NUM')
        self.agent_config_dir = self._get_required_env('AGENT_CONFIG_DIR')

        # 第四阶段：MLE-bench配置
        self.mle_bench_competition = self._get_required_env('MLE_BENCH_COMPETITION')
        self.mle_bench_server_url = self._get_required_env('MLE_BENCH_SERVER_URL')
        self.mle_bench_workspace_dir = self._get_required_env('MLE_BENCH_WORKSPACE_DIR')
        self.mle_bench_time_limit = self._get_required_int_env('MLE_BENCH_TIME_LIMIT')
        self.time_limit_seconds = self.mle_bench_time_limit*3600
        self.mle_bench_epoch_limit = self._get_required_int_env('MLE_BENCH_EPOCH_LIMIT')
        self.mle_bench_private_data_dir = self._get_required_env('MLE_BENCH_PRIVATE_DATA_DIR')

        # 第五阶段：Conda环境配置
        self.conda_env_name = self._get_required_env('CONDA_ENV_NAME')

        # 第六阶段：任务执行配置
        self.init_task_num = self._get_required_int_env('INIT_TASK_NUM')
        self.explore_ratio = self._get_required_float_env('EXPLORE_RATIO')
        self.epoch_task_num = self._get_required_int_env('EPOCH_TASK_NUM')

        self._initialized = True

    def _get_required_env(self, key: str) -> str:
        """
        读取必需的字符串环境变量，缺失或为空时抛出错误

        参数:
            key: 环境变量名称

        返回:
            str: 合法的环境变量值
        """
        value = os.getenv(key)
        if value is None or value.strip() == '':
            log_msg("ERROR", f"{key}为必填配置，请在.env文件中设置")
        return value

    def _get_required_int_env(self, key: str) -> int:
        """
        读取必需的整数环境变量，包含数值合法性校验

        参数:
            key: 环境变量名称

        返回:
            int: 转换后的整数值
        """
        value_str = self._get_required_env(key)
        try:
            return int(value_str)
        except ValueError as exc:
            log_msg("ERROR", f"{key}必须为整数，当前值为{value_str}")

    def _get_optional_int_env(self, key: str, default: int) -> int:
        """
        读取可选的整数环境变量，如果不存在或为空则返回默认值。

        参数:
            key: 环境变量名称
            default: 默认值

        返回:
            int: 转换后的整数值或默认值
        """
        value_str = os.getenv(key)
        if value_str is None or value_str.strip() == '':
             return default
        try:
            return int(value_str)
        except ValueError as exc:
             log_msg("WARNING", f"{key}必须为整数，当前值为{value_str}，使用默认值{default}")
             return default

    def _get_required_float_env(self, key: str) -> float:
        """
        读取必需的浮点数环境变量，包含数值合法性校验

        参数:
            key: 环境变量名称

        返回:
            float: 转换后的浮点数值
        """
        value_str = self._get_required_env(key)
        try:
            return float(value_str)
        except ValueError as exc:
            log_msg("ERROR", f"{key}必须为浮点数，当前值为{value_str}")

    def validate(self) -> tuple[bool, str]:
        """
        验证必要的配置是否已设置

        参数:
            无

        返回:
            tuple[bool, str]: (是否验证通过, 错误信息)
        """
        if self.api_key == 'your_api_key_here':
            return False, "请将API_KEY替换为实际的API密钥"

        return True, ""

    def create_llm_client(self) -> OpenAI:
        """
        按照 OpenAI 兼容协议创建统一的 LLM 客户端。

        参数:
            无

        返回:
            OpenAI: 已注入 API Key 与 Base URL 的客户端实例
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.agent_timeout,
            max_retries=self.max_retries
        )

    def create_langchain_llm(self):
        """
        创建 LangChain 兼容的 Chat Model。

        用于新的 LangGraph Agent 架构。

        参数:
            无

        返回:
            BaseChatModel: LangChain Chat Model 实例
        """
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.api_base,
            temperature=0.2,
            request_timeout=self.agent_timeout,
            max_retries=self.max_retries,
        )


# 全局单例访问函数
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    获取配置单例实例

    参数:
        无

    返回:
        Config: 配置对象

    示例:
        >>> from utils.config import get_config
        >>> config = get_config()
        >>> print(config.api_key)
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance