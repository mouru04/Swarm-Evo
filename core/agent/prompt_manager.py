"""
Prompt 管理模块，负责拼装动态补充信息。

本模块已重构以支持 LangGraph 架构：
- 使用 LangChain 的 Message 类型 (SystemMessage, HumanMessage)
- 保留 Jinja2 模板渲染能力
- 支持 explore, select, merge, review 多种任务类型
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from utils.logger_system import log_msg
from utils.directory_tree_generator import DirectoryTreeGenerator
from utils.system_info import get_hardware_description


@dataclass
class PromptContext:
    """
    提示上下文数据结构，描述动态变量与运行阶段。
    
    所有任务类型 (explore, select, merge, review, evaluate) 共享此结构，
    不同任务类型只使用其相关字段。
    """

    # ========== 通用字段 (所有任务都需要) ==========
    workspace_root: str
    conda_env_name: str
    time_limit_seconds: int
    total_iterations: int
    iteration: int
    elapsed_seconds: float
    remaining_seconds: float
    conda_packages: str
    task_description: str

    # ========== 控制字段 ==========
    # 决定使用哪个模板，如 "explore_user_prompt.j2"
    template_name: Optional[str] = None

    # ========== Explore 任务字段 ==========
    parent_code: Optional[str] = None
    parent_feedback: Optional[str] = None
    parent_history: Optional[str] = None
    parent_score: Optional[float] = None

    # ========== Merge/Select 任务字段 ==========
    candidates: Optional[Dict[str, str]] = None
    gene_plan: Optional[Dict[str, Any]] = None

    # ========== Evaluate 任务字段 ==========
    solution_code: Optional[str] = None
    execution_logs: Optional[str] = None


class PromptManager:
    """
    提示词管理器，负责拼接动态补充信息与模板注入。
    
    重构后的核心变化：
    - build_system_message(): 返回 SystemMessage 对象
    - build_initial_messages(): 返回完整的初始消息列表 [SystemMessage, HumanMessage]
    - 保留所有 .j2 模板支持
    - 保留进化算法的模板注入接口 (set_template, get_template)
    """

    def __init__(self, template_dir: str = "benchmark/mle-bench/prompt_templates") -> None:
        """
        初始化提示词管理器。

        初始化时将所有模板从文件系统加载到内存，后续只使用内存模板。

        参数:
            template_dir: 外置模板目录，默认使用 benchmark/mle-bench/prompt_templates。
        """
        self._template_dir = Path(template_dir)
        self._templates: Dict[str, str] = {}
        self._load_all_templates()

    def _load_all_templates(self) -> None:
        """从文件系统加载所有模板到内存。"""
        for template_file in self._template_dir.glob("*.j2"):
            with open(template_file, "r", encoding="utf-8") as f:
                self._templates[template_file.name] = f.read()

    def _get_template(self, name: str) -> Template:
        """
        获取模板对象。
        
        从内存中的_templates获取模板字符串并创建jinja2.Template对象。
        如果模板不存在，通过log_msg记录错误并抛出异常。

        参数:
            name: 模板名称（如 "explore_user_prompt.j2"）

        返回:
            jinja2.Template对象
        """
        if name not in self._templates:
            log_msg("ERROR", f"模板 '{name}' 不存在于 PromptManager 中")
            raise KeyError(f"Template '{name}' not found")
        return Template(self._templates[name])

    def set_template(self, name: str, content: str) -> None:
        """
        设置实例的模板内容。

        供进化算法调用，将变异后的Prompt内容设置到该实例。

        参数:
            name: 模板名称（如 "explore_user_prompt.j2"）
            content: 完整的Jinja2模板字符串
        """
        self._templates[name] = content

    def get_template(self, name: str) -> str:
        """
        获取实例的模板内容。

        参数:
            name: 模板名称

        返回:
            模板字符串
        """
        if name not in self._templates:
            log_msg("ERROR", f"模板 '{name}' 不存在于 PromptManager 中")
            raise KeyError(f"Template '{name}' not found")
        return self._templates[name]

    def reset_template(self, name: str) -> None:
        """
        用基础模板重置私有模板。

        从文件系统重新读取基础模板并覆盖当前的私有模板。
        这是唯一会重新访问文件系统的操作。

        参数:
            name: 模板名称
        """
        template_path = self._template_dir / name
        if not template_path.exists():
            log_msg("ERROR", f"基础模板文件 '{name}' 不存在")
            raise FileNotFoundError(f"Template file '{name}' not found")
        with open(template_path, "r", encoding="utf-8") as f:
            self._templates[name] = f.read()

    # ========================================================================
    # 新增：LangGraph 兼容的 Message 构建方法
    # ========================================================================

    def build_system_message(self, context: PromptContext) -> SystemMessage:
        """
        构建 System Message。

        将任务模板渲染为 SystemMessage，包含：
        - 角色定义
        - 环境信息（目录树、硬件、时间）
        - 任务上下文（explore/select/merge/review 各自的参数）

        参数:
            context: 运行时动态上下文数据。

        返回:
            LangChain SystemMessage 对象。
        """
        # 确定模板名称
        template_name = context.template_name or "explore_user_prompt.j2"
        template = self._get_template(template_name)

        # 准备渲染变量
        render_vars = self._prepare_render_variables(context)

        # 渲染模板
        rendered_content = template.render(**render_vars)

        return SystemMessage(content=rendered_content)

    def build_initial_messages(self, context: PromptContext, task_instruction: str) -> List[BaseMessage]:
        """
        构建初始消息列表，用于 LangGraph 的 input。

        返回格式: [SystemMessage, HumanMessage]
        - SystemMessage: 从 j2 模板渲染的角色 + 环境信息
        - HumanMessage: 具体的任务指令

        参数:
            context: 运行时动态上下文数据。
            task_instruction: 具体的任务指令文本。

        返回:
            消息列表。
        """
        system_msg = self.build_system_message(context)
        human_msg = HumanMessage(content=task_instruction)
        return [system_msg, human_msg]

    def _prepare_render_variables(self, context: PromptContext) -> Dict[str, Any]:
        """
        准备模板渲染所需的所有变量。

        参数:
            context: 运行时动态上下文数据。

        返回:
            渲染变量字典。
        """
        # 确保 workspace_root 是绝对路径
        workspace_root = str(Path(context.workspace_root).resolve())

        # 准备时间信息
        elapsed = self._format_duration(context.elapsed_seconds)
        remaining = self._format_duration(context.remaining_seconds)
        total_time = self._format_duration(context.time_limit_seconds)

        # 获取动态硬件信息
        device_info = get_hardware_description()
        log_msg("INFO", f"Device info: {device_info}")

        # 准备目录树信息
        file_previews = {}
        try:
            tree_gen = DirectoryTreeGenerator(
                workspace_root, 
                ignore_patterns=['.git', '__pycache__', "agent"]
            )
            directory_tree, file_previews = tree_gen.generate()
        except Exception as e:
            directory_tree = f"Error generating directory tree: {e}"
            log_msg("WARNING", f"Directory tree generation failed: {e}")

        return {
            # 通用字段
            "workspace_root": workspace_root,
            "conda_env_name": context.conda_env_name,
            "time_limit": self._format_hours(context.time_limit_seconds),
            "task_description": context.task_description,
            "elapsed": elapsed,
            "remaining": remaining,
            "total_time": total_time,
            "iteration": context.iteration,
            "total_iterations": context.total_iterations,
            "conda_packages": context.conda_packages.strip(),
            "device_info": device_info,
            "directory_tree": directory_tree,
            "file_previews": file_previews,
            # 不再需要 history_block，因为历史由 LangGraph 的 messages 管理
            "history_block": "",

            # Explore 字段
            "parent_code": context.parent_code,
            "parent_feedback": context.parent_feedback,
            "parent_history": context.parent_history,
            "parent_score": context.parent_score,

            # Merge / Select 字段
            "candidates": context.candidates,
            "gene_plan": context.gene_plan,

            # Evaluate 字段
            "solution_code": context.solution_code,
            "execution_logs": context.execution_logs,
        }

    # ========================================================================
    # 兼容性方法：保留旧接口以支持渐进迁移
    # ========================================================================

    def build_system_prompt(self, context: Optional[PromptContext] = None) -> str:
        """
        构建 System Prompt 字符串。
        
        [兼容性方法] 保留以支持旧代码。
        建议使用 build_system_message() 代替。
        """
        template = self._get_template("system_prompt.j2")
        return template.render()

    def build_user_prompt(self, context: PromptContext, history: str) -> str:
        """
        构建 User Prompt 字符串。

        [兼容性方法] 保留以支持旧代码。
        在新架构中，建议使用 build_initial_messages() 代替。

        参数:
            context: 运行时动态上下文数据。
            history: ReAct 执行历史 (在新架构中不再需要手动传递)。
        返回:
            拼接完模板与补充信息后的完整用户提示字符串。
        """
        template_name = context.template_name or "explore_user_prompt.j2"
        template = self._get_template(template_name)
        
        render_vars = self._prepare_render_variables(context)
        # 兼容旧接口：手动添加 history_block
        render_vars["history_block"] = self._build_history_block(history)
        
        return template.render(**render_vars)

    def _build_history_block(self, history: str) -> str:
        """生成历史记录区块，若无历史则返回空字符串。"""
        if not history or not history.strip():
            return "No history available."
        return history

    def _format_duration(self, seconds: float) -> str:
        """将秒数转换为"分秒"文本。"""
        safe_seconds = max(0, int(seconds))
        minutes, remain = divmod(safe_seconds, 60)
        return f"{minutes}分{remain}秒"

    def _format_hours(self, seconds: float) -> str:
        """将秒数转换为小时（保留1位小数）。"""
        return f"{seconds / 3600:.1f}"
