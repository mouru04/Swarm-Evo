"""TOON 格式化与历史渲染工具模块。"""

from __future__ import annotations

from typing import Any, Dict, List


def escape_string(text: str) -> str:
    """
    转义双引号，确保字符串可安全嵌入 TOON。

    参数:
        text: 需要转义的原始文本。
    返回:
        处理后的安全字符串，非字符串输入会被转换为字符串。
    """

    if not isinstance(text, str):
        return str(text)
    return text.replace("\"", "\\\"")


def convert_input_to_toon(tool_input: Any) -> str:
    """
    将工具输入转换为 TOON 文本，支持字符串、列表与字典。

    参数:
        tool_input: 工具输入对象，可为字符串、列表、字典或基础类型。
    返回:
        对应的 TOON 文本表示，便于插入到历史记录。
    """

    if isinstance(tool_input, str):
        return f"\"{escape_string(tool_input)}\""

    if isinstance(tool_input, dict):
        lines = []
        for key, value in tool_input.items():
            value_str = convert_input_to_toon(value)
            lines.append(f"    {key}: {value_str}")
        return "(\n" + "\n".join(lines) + "\n)"

    if isinstance(tool_input, list):
        items = ", ".join(convert_input_to_toon(item) for item in tool_input)
        return f"[{items}]"

    return str(tool_input)


def format_step_toon(entry: Dict[str, Any]) -> str:
    """
    将单条 ReAct 历史记录渲染为 ReActStep TOON 对象，统一使用 tool 字段展示工具名称。

    参数:
        entry: 包含 step、task、action、tool_input、observation 的记录字典。
    返回:
        符合 TOON 规范的 ReActStep 文本。
    """

    step = entry.get("step", 0)
    task = escape_string(entry.get("task", ""))
    tool_name = escape_string(entry.get("tool", entry.get("action", "")))
    tool_input = entry.get("tool_input", "")
    observation = entry.get("observation", "")

    return (
        "ReActStep (\n"
        f"    step: {step}\n"
        f"    task: \"{task}\"\n"
        f"    tool: \"{tool_name}\"\n"
        f"    input: {convert_input_to_toon(tool_input)}\n"
        f"    observation: {observation}\n"
        ")"
    )


def render_history(records: List[Dict[str, Any]]) -> str:
    """
    批量渲染历史记录，若无记录返回空字符串。

    参数:
        records: ReAct 步骤字典列表。
    返回:
        拼接后的历史文本，每条记录之间以空行分隔。
    """

    if not records:
        return ""
    return "\n\n".join(format_step_toon(entry) for entry in records)