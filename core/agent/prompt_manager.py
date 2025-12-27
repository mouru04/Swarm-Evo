"""Prompt 管理模块，负责拼装动态补充信息。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
from jinja2 import Template

from utils.logger_system import log_msg
from utils.directory_tree_generator import DirectoryTreeGenerator
from utils.system_info import get_hardware_description

@dataclass
class PromptContext:
    """提示上下文数据结构，描述动态变量与运行阶段。"""

    # Common fields
    workspace_root: str
    conda_env_name: str
    time_limit_seconds: int
    total_iterations: int
    iteration: int
    elapsed_seconds: float
    remaining_seconds: float
    conda_packages: str
    task_description: str  # Updated from task_goal as per feedback

    # Control fields
    template_name: Optional[str] = None

    # Explore fields
    parent_code: Optional[str] = None
    parent_feedback: Optional[str] = None
    parent_history: Optional[str] = None

    # Merge/Select fields
    candidates: Optional[Dict[str, str]] = None
    gene_plan: Optional[Dict[str, Any]] = None

    # Evaluate fields
    solution_code: Optional[str] = None
    execution_logs: Optional[str] = None


class PromptManager:
    """提示词管理器，负责拼接动态补充信息与模板注入。
    
    每个PromptManager实例在初始化时将基础模板加载到内存中，
    后续只使用内存中的私有模板，支持每个Agent独立进化Prompt。
    """

    def __init__(self, template_dir: str = "benchmark/mle-bench/prompt_templates") -> None:
        """初始化提示词管理器。

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
        """获取模板对象。
        
        从内存中的_templates获取模板字符串并创建jinja2.Template对象。
        如果模板不存在，通过log_msg记录错误并抛出异常。

        参数:
            name: 模板名称（如 "explore_user_prompt.j2"）

        返回:
            jinja2.Template对象
        """
        if name not in self._templates:
            log_msg("ERROR", f"模板 '{name}' 不存在于 PromptManager 中")
        return Template(self._templates[name])

    def set_template(self, name: str, content: str) -> None:
        """设置实例的模板内容。

        供进化算法调用，将变异后的Prompt内容设置到该实例。

        参数:
            name: 模板名称（如 "explore_user_prompt.j2"）
            content: 完整的Jinja2模板字符串
        """
        self._templates[name] = content

    def get_template(self, name: str) -> str:
        """获取实例的模板内容。

        参数:
            name: 模板名称

        返回:
            模板字符串
        """
        if name not in self._templates:
            log_msg("ERROR", f"模板 '{name}' 不存在于 PromptManager 中")
        return self._templates[name]

    def reset_template(self, name: str) -> None:
        """用基础模板重置私有模板。

        从文件系统重新读取基础模板并覆盖当前的私有模板。
        这是唯一会重新访问文件系统的操作。

        参数:
            name: 模板名称
        """
        template_path = self._template_dir / name
        if not template_path.exists():
            log_msg("ERROR", f"基础模板文件 '{name}' 不存在")
        with open(template_path, "r", encoding="utf-8") as f:
            self._templates[name] = f.read()

    def build_system_prompt(self, context: Optional[PromptContext] = None) -> str:
        """构建 System Prompt。
        
        注意：目前 system_prompt.j2 不接受任何动态变量。
        """
        template = self._get_template("system_prompt.j2")
        return template.render()

    def build_user_prompt(self, context: PromptContext, history: str) -> str:
        """构建 User Prompt。

        参数:
            context: 运行时动态上下文数据。
            history: ReAct 执行历史。
        返回:
            拼接完模板与补充信息后的完整用户提示字符串。
        """
        if not context.template_name:
            template_name = "explore_user_prompt.j2" 
        else:
            template_name = context.template_name

        template = self._get_template(template_name)
        
        # Ensure workspace_root is absolute path
        context.workspace_root = str(Path(context.workspace_root).resolve())
        
        # 准备环境信息
        elapsed = self._format_duration(context.elapsed_seconds)
        remaining = self._format_duration(context.remaining_seconds)
        total_time = self._format_duration(context.time_limit_seconds)
        # 获取动态硬件信息
        device_info = get_hardware_description()
        log_msg("INFO", f"Device info: {device_info}")
        
        # 准备目录树信息
        file_previews = {}
        try:
            tree_gen = DirectoryTreeGenerator(context.workspace_root, ignore_patterns=['.git', '__pycache__',"agent"])
            directory_tree, file_previews = tree_gen.generate()
            # log_msg("INFO", f"Directory tree generated: {directory_tree}")
        except Exception as e:
            # Fallback if generation fails
            directory_tree = f"Error generating directory tree: {e}"

        # 准备历史信息
        history_block = self._build_history_block(history)

        return template.render(
            # Common
            workspace_root=context.workspace_root,
            conda_env_name=context.conda_env_name,
            time_limit=self._format_hours(context.time_limit_seconds),
            
            task_description=context.task_description,
            elapsed=elapsed,
            remaining=remaining,
            total_time=total_time,
            iteration=context.iteration,
            total_iterations=context.total_iterations,
            conda_packages=context.conda_packages.strip(),
            device_info=device_info,
            history_block=history_block,
            directory_tree=directory_tree,
            file_previews=file_previews,

            # Explore
            parent_code=context.parent_code,
            parent_feedback=context.parent_feedback,
            parent_history=context.parent_history,

            # Merge / Select
            candidates=context.candidates,
            gene_plan=context.gene_plan,

            # Evaluate
            solution_code=context.solution_code,
            execution_logs=context.execution_logs
        )

    def _build_history_block(self, history: str) -> str:
        """
        生成历史记录区块，若无历史则返回空字符串。
        """
        if not history or not history.strip():
            return "No history available."
        return history

    def _format_duration(self, seconds: float) -> str:
        """将秒数转换为“分秒”文本。"""
        safe_seconds = max(0, int(seconds))
        minutes, remain = divmod(safe_seconds, 60)
        return f"{minutes}分{remain}秒"

    def _format_hours(self, seconds: float) -> str:
        """将秒数转换为小时（保留1位小数）。"""
        return f"{seconds / 3600:.1f}"
