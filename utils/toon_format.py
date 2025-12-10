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
        f"    action: \"{tool_name}\"\n"
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


import re

TOON_HEADER_RE = re.compile(r"(ActionStep|FinalAnswer)\s*\(")

def normalize_toon_output(output: str) -> str:
    """最稳健、工程级的 TOON 正规化: 清洗、定位、括号平衡解析"""

    if not output:
        return ""

    text = output.strip()

    # 1. 清洗 fenced code
    if text.startswith("```"):
        # 去头部的 ```
        text = text.split("```", 1)[-1].strip()
        # 去尾部 ```
        if "```" in text:
            text = text.split("```", 1)[0].strip()

    # 2. 找 TOON 起点
    m = TOON_HEADER_RE.search(text)
    if not m:
        return text

    start = m.start()

    # 3. 括号计数，实现安全截断
    depth = 0
    i = start
    seen_first_paren = False

    while i < len(text):
        c = text[i]

        if c == "(":
            depth += 1
            seen_first_paren = True
        elif c == ")":
            depth -= 1
            if seen_first_paren and depth == 0:
                return text[start:i+1].strip()

        i += 1

    # 如果到文末都没有平衡，返回从 start 开始的全部内容
    return text[start:].strip()

def validate_single_root(output: str, root_type: str, logger: LoggerSystem):
    if output.startswith(root_type):
        if output.count(root_type) != 1:
            logger.text_log("ERROR", f"TOON 格式错误：{root_type}(...) 只能出现一次")
            raise ValueError(f"{root_type}(...) must appear exactly once")

def validate_toon_format(output: str, logger: LoggerSystem):
    out = output.strip()
    if out.startswith("ActionStep"):
        validate_single_root(out, "ActionStep", logger)
    elif out.startswith("FinalAnswer"):
        validate_single_root(out, "FinalAnswer", logger)
    else:
        logger.text_log("ERROR", f"TOON 格式错误：{out}")
        raise ValueError("Invalid TOON root object")

def parse_tool_input(self, action: str, raw_input: str) -> Tuple[Any, str]:
    """
    解析 TOON 的 input 字段：
        
    input: (
        path: "/data"
        recursive: true
    )

    转换为 Python dict，供工具直接使用。
    """

    text = raw_input.strip()
    detection_hint = text

    # 写文件特殊处理：使用贪婪匹配抓取 input: ( ... ) 全体，避免被正文中的右括号截断
    if action == "write_file":
        m_write = re.search(r"input\s*:\s*\((?P<body>.*)", text, re.S)
        if not m_write:
            return text, text

        body = m_write.group("body")

        path_match = re.search(r'^\s*(?:path|file_path)\s*:\s*([\'"])(?P<p>.*?)\1', body, re.M)
        path_val = path_match.group("p") if path_match else None

        content_match = re.search(r'content\s*:\s*(?P<c>.*)', body, re.S)
        if not content_match:
            raise ValueError("write_file tool input missing content section")

        content_raw = content_match.group("c")

        # 去除结尾单独一行的 ")"
        content_lines = content_raw.rstrip().splitlines()
        while content_lines and content_lines[-1].strip() == ")":
            content_lines.pop()
        content_raw = "\n".join(content_lines).rstrip()

        def _strip_wrapper(val: str) -> str:
            if val.startswith('"""') or val.startswith("'''"):
                delim = val[:3]
                inner = val[3:]
                if inner.endswith(delim):
                    inner = inner[:-3]
                return inner
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                return val[1:-1]
            return val

        content_val = _strip_wrapper(content_raw)

        if path_val is None:
            raise ValueError("write_file tool input missing path field")

        result = {"path": path_val, "content": content_val}
        detection_hint = str(result)
        return result, detection_hint

    # 非写文件依旧使用默认 input: ( ... ) 非贪婪匹配
    m = self.TOON_INPUT_PATTERN.search(text)
    if not m:
        # fallback：可能是 input: "xxx"
        str_match = re.search(r'input\s*:\s*"(?P<val>.*?)"', text)
        if str_match:
            val = str_match.group("val")
            return val, val
        return text, text

    body = m.group("body").strip()

    result = {}

    body_lines = body.splitlines()
    idx = 0

    while idx < len(body_lines):
        line = body_lines[idx].strip()
        if not line or ":" not in line:
            idx += 1
            continue

        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # 三引号多行字符串
        if val.startswith('"""') or val.startswith("'''"):
            delimiter = val[:3]
            content_part = val[3:]
            collected: List[str] = []
            if content_part:
                collected.append(content_part)

            idx += 1
            closed = False
            while idx < len(body_lines):
                current_line = body_lines[idx]
                if current_line.strip().endswith(delimiter):
                    end_pos = current_line.rfind(delimiter)
                    collected.append(current_line[:end_pos])
                    closed = True
                    break
                collected.append(current_line)
                idx += 1

            if not closed:
                raise ValueError("write_file content missing closing triple quotes")

            val = "\n".join(collected)
        # 普通引号包裹但跨多行的字符串
        elif (val.startswith('"') and not val.endswith('"')) or (val.startswith("'") and not val.endswith("'")):
            delimiter = val[0]
            collected: List[str] = []
            if len(val) > 1:
                collected.append(val[1:])  # 去掉首引号

            idx += 1
            closed = False
            while idx < len(body_lines):
                current_line = body_lines[idx]
                if current_line.endswith(delimiter):
                    collected.append(current_line[:-1])
                    closed = True
                    break
                collected.append(current_line)
                idx += 1

            if not closed:
                raise ValueError("write_file content missing closing quote")

            val = "\n".join(collected)
        else:
            # 字符串
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            # 布尔
            elif val in ("true", "false"):
                val = val == "true"
            else:
                # 尝试数字
                try:
                    val = int(val)
                except:
                    try:
                        val = float(val)
                    except:
                        # 保留原始
                        val = val

        result[key] = val
        idx += 1

    detection_hint = str(result)
    return result, detection_hint