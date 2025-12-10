"""面向 JSON 的安全工具封装，确保输出结构化且可控。"""

from __future__ import annotations

import codecs
import json
import os
import shutil
import subprocess
import time
from typing import Optional, Any, Dict

from langchain.tools import BaseTool
from langchain_community.tools.file_management import ListDirectoryTool, ReadFileTool

# 全局截断常量，保持工具行为一致
MAX_LIST_ITEMS = 100
MAX_READFILE_CHARS_DEFAULT = 3000
MAX_READFILE_CHARS_CODE = 20000
MAX_STDOUT_CHARS = 4000
MAX_STDERR_CHARS = 2000


def tool_observation(ok: bool, data: Optional[Dict[str, Any]] = None, error: Optional[Dict[str, Any]] = None) -> str:
    """
    包装工具观测结果为 JSON 字符串。

    参数:
        ok: 工具调用是否成功。
        data: 成功时的负载数据。
        error: 失败时的错误负载数据。
    返回:
        ToolObservation JSON 文本。
    """
    payload = {
        "type": "ToolObservation",
        "ok": ok,
        "data": data,
        "error": error
    }
    return json.dumps(payload, ensure_ascii=False)


def _wrap_error(error_type: str, message: str, **extra_fields) -> str:
    """
    统一构造错误观察对象，保证结构一致。

    参数:
        error_type: 错误类型标识。
        message: 错误描述。
        **extra_fields: 附加诊断字段。
    返回:
        封装好的 ToolObservation 错误 JSON 文本。
    """
    error_payload = {
        "type": "ErrorInfo",
        "error_type": error_type,
        "message": message,
        **extra_fields
    }
    return tool_observation(ok=False, error=error_payload)


class SafeListDirectoryTool(BaseTool):
    """安全版目录列举工具，自动截断长列表并输出 JSON。"""

    name: str = "list_directory"
    description: str = "List directory contents and return JSON-formatted output, auto-truncating long results."

    def _run(self, path: str) -> str:
        """
        列举目录内容并返回 JSON 文本，超出上限自动截断。

        参数:
            path: 目标目录路径。
        返回:
            ToolObservation 结构化结果，包含目录统计与截断标记。
        """

        tool = ListDirectoryTool()
        try:
            raw = tool.run({"dir_path": path})
        except Exception as exc:
            return _wrap_error("ListDirectoryError", str(exc), path=path)

        items = raw.strip().split("\n") if raw else []
        total = len(items)
        shown = items[:MAX_LIST_ITEMS]

        data = {
            "type": "DirectoryListing",
            "path": path,
            "total_count": total,
            "shown_count": len(shown),
            "truncated": total > MAX_LIST_ITEMS,
            "files": shown,
        }
        return tool_observation(ok=True, data=data)


class SafeReadFileTool(BaseTool):
    """安全版文件读取工具，兼顾大文件截断与错误提示。"""

    name: str = "read_file"
    description: str = "Read file content (auto-truncate large files) and output as JSON."

    def _run(self, file_path: str) -> str:
        """
        读取文件并在必要时截断输出。

        参数:
            file_path: 待读取的文件路径。
        返回:
            ToolObservation 结构，成功时包含读取预览与截断信息。
        """

        tool = ReadFileTool()

        try:
            size = os.path.getsize(file_path)
        except Exception as exc:
            return _wrap_error("FileNotFound", str(exc), path=file_path)

        # Determine limit based on extension
        limit = MAX_READFILE_CHARS_CODE if file_path.endswith(".py") else MAX_READFILE_CHARS_DEFAULT

        if size > limit:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read(limit)
            except Exception as exc:
                return _wrap_error("LargeFileReadError", str(exc), path=file_path)

            data = {
                "type": "FileReadResult",
                "path": file_path,
                "size_bytes": size,
                "truncated": True,
                "content": content,
            }
            return tool_observation(ok=True, data=data)

        try:
            raw = tool.run({"file_path": file_path})
        except Exception as exc:
            return _wrap_error("ReadFileError", str(exc), path=file_path)

        data = {
            "type": "FileReadResult",
            "path": file_path,
            "size_bytes": size,
            "truncated": False,
            "content": raw,
        }
        return tool_observation(ok=True, data=data)


class SafeShellTool(BaseTool):
    """带截断的 Shell 执行工具，失败时返回完整 stderr。"""

    name: str = "terminal"
    description: str = (
        "Execute a shell command with smart truncation. "
        "If the command succeeds, stdout/stderr are truncated. "
        "If it fails (exit != 0), full stderr is returned for debugging."
    )
    conda_env_name: str = ""

    def __init__(self, conda_env_name: Optional[str] = None, **kwargs: object) -> None:
        """
        初始化 Shell 工具，绑定目标 Conda 环境名称。

        参数:
            conda_env_name: 需要自动激活的 Conda 环境名称。
            **kwargs: 透传给父类的其他配置。
        返回:
            None
        """

        super().__init__(conda_env_name=conda_env_name or "", **kwargs)

    def _run(self, command: str) -> str:
        """
        执行 Shell 命令并提供截断控制。

        参数:
            command: 需执行的命令行字符串（自动前置 Conda 激活）。
        返回:
            ToolObservation，成功时包含截断的输出，失败时返回完整错误信息。
        """

        env_name = self.conda_env_name.strip()
        if not env_name:
            return _wrap_error("ShellExecutionError", "conda_env_name 未配置，无法执行命令")

        conda_exe = os.environ.get("CONDA_EXE")
        resolved_conda = conda_exe if conda_exe else "conda"
        if shutil.which(resolved_conda) is None:
            return _wrap_error("ShellExecutionError", f"未找到可用的 conda 可执行文件：{resolved_conda}")

        activation_chain = (
            f'eval "$({resolved_conda} shell.bash hook)" '
            f"&& conda activate {env_name} "
            f"&& {command}"
        )

        try:
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", activation_chain],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate()
            exit_code = proc.returncode
        except Exception as exc:
            return _wrap_error("ShellExecutionError", str(exc))

        if exit_code == 0:
            stdout_preview = stdout[:MAX_STDOUT_CHARS] if stdout else ""
            stderr_preview = stderr[:MAX_STDERR_CHARS] if stderr else ""

            data = {
                "type": "ShellResult",
                "command": command,
                "exit_code": exit_code,
                "stdout_preview": stdout_preview,
                "stderr_preview": stderr_preview,
                "stdout_truncated": bool(stdout and len(stdout) > MAX_STDOUT_CHARS),
                "stderr_truncated": bool(stderr and len(stderr) > MAX_STDERR_CHARS),
            }
            return tool_observation(ok=True, data=data)

        full_stderr = stderr if stderr else ""
        stdout_preview = stdout[:MAX_STDOUT_CHARS] if stdout else ""
        return _wrap_error(
            "ShellCommandError",
            f"Command exited with {exit_code}",
            stderr=full_stderr,
            stdout_preview=stdout_preview,
        )


class SafeWriteFileTool(BaseTool):
    """写文件工具，兼容转义字符并输出 JSON。"""

    name: str = "write_file"
    description: str = (
        "Write text to a file (overwrite). Accepts escaped strings (with \\n, "
        "\\t, \\r, \\\" etc.) and automatically decodes them into raw multiline text. "
        "Works even when content is double-escaped or wrapped in quotes."
    )

    def decode_content(self, content: str) -> str:
        """
        解码转义文本为原始多行字符串。

        参数:
            content: 可能包含转义符的文本。
        返回:
            解码后的纯文本。
        """

        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]

        content = content.replace("\\\\n", "\\n").replace("\\\\t", "\\t").replace("\\\\r", "\\r")

        try:
            decoded = codecs.decode(content, "unicode_escape")
        except Exception:
            decoded = content.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\n")

        return decoded.replace("\r\n", "\n").replace("\r", "\n")

    def _run(self, path: str | None = None, content: str | None = None, file_path: str | None = None, text: str | None = None) -> str:
        """
        写文件（覆盖）。兼容 path/content 与 file_path/text 两种字段。

        参数:
            path: 写入文件路径，优先使用。
            content: 写入内容（可带转义）。
            file_path: 可选的文件路径备用字段。
            text: 可选的内容备用字段。
        返回:
            ToolObservation，包含写入字节数与内容预览。
        """

        resolved_path = path or file_path
        resolved_content = content if content is not None else text

        if resolved_path is None:
            return _wrap_error("WriteFileError", "缺少 path 参数", path=str(resolved_path))
        if resolved_content is None:
            return _wrap_error("WriteFileError", "缺少 content 参数", path=str(resolved_path))

        try:
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            decoded = self.decode_content(resolved_content)
            
            # Snapshot Logic for solution.py
            if os.path.basename(resolved_path) == "solution.py":
                try:
                    snapshot_dir = os.path.join(os.path.dirname(resolved_path), ".snapshots")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    snapshot_path = os.path.join(snapshot_dir, f"solution_{timestamp}.py")
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        f.write(decoded)
                except Exception as e:
                    # Snapshot failure should not block main write
                    pass

            with open(resolved_path, "w", encoding="utf-8") as handle:
                handle.write(decoded)
        except Exception as exc:
            return _wrap_error("WriteFileError", str(exc), path=resolved_path)

        preview = decoded[:2000]
        data = {
            "type": "WriteFileResult",
            "path": resolved_path,
            "bytes_written": len(decoded.encode("utf-8")),
            "preview": preview,
        }
        return tool_observation(ok=True, data=data)
