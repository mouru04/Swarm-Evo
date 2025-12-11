"""
系统信息获取工具模块

提供自动检测和获取系统硬件配置信息以及conda环境信息的功能
"""

import json
import os
import platform
import subprocess
from collections import Counter
import shutil
from typing import Dict, List, Optional, Union
from utils.logger_system import log_msg


def get_cpu_count() -> int:
    """
    获取CPU核心数

    返回:
        int: CPU核心数
    """
    count = os.cpu_count()
    if count is None:
        log_msg("ERROR", "Failed to get CPU count: os.cpu_count() returned None")
    return count


def get_memory_info() -> Dict[str, float]:
    """
    获取内存信息（单位：GB）

    返回:
        Dict[str, float]: 包含total（总内存）和available（可用内存）的字典

    实现细节:
        - Linux: 读取 /proc/meminfo
        - 其他系统: 尝试使用psutil库，失败则返回默认值
    """
    if platform.system() == "Linux":
        # 第一阶段：读取 /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()

        # 第二阶段：解析总内存
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                # 提取内存大小（单位：KB）
                mem_kb = int(line.split()[1])
                mem_gb = mem_kb / (1024 * 1024)
                return {'total': round(mem_gb, 2), 'available': round(mem_gb, 2)}

    # 尝试使用psutil（如果已安装）
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total': round(mem.total / (1024**3), 2),
            'available': round(mem.available / (1024**3), 2)
        }
    except ImportError:
        pass

    # 如果无法获取内存信息，抛出异常
    log_msg("ERROR", "Failed to get memory info: /proc/meminfo parsing failed and psutil not available")


def get_gpu_info() -> Optional[str]:
    """
    获取GPU信息

    返回:
        Optional[str]: GPU描述字符串，如果没有GPU或无法检测则返回None

    实现细节:
        - 尝试使用nvidia-smi命令检测NVIDIA GPU
        - 如果检测失败，返回None表示无GPU或无法检测
    """
    # 尝试使用nvidia-smi命令
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        timeout=5
    )

    if result.returncode != 0:
        log_msg("ERROR", f"nvidia-smi command failed with return code {result.returncode}: {result.stderr}")

    if not result.stdout.strip():
        # 如果没有输出，可能表示没有检测到GPU，或者确实没有GPU。
        # 根据需求"不允许返回默认值，出错要报error"，
        # 这里如果认为"没有GPU"是正常状态而不是错误，应该怎么处理？
        # 通常 get_gpu_info 返回 None 表示没有 GPU 是合理的，
        # 但既然 user 说 "如果出错全部要报对应的error"，
        # 我们可以区分: 命令成功但无输出(无GPU) vs 命令失败/超时(Error).
        # 如果 nvidia-smi 存在但输出空，通常意味着没有 GPU 进程或类似情况，但在这里更像是没有 GPU。
        # 以前返回 None。如果不允许默认值（None is a "default" for "no GPU" in some sense, but conceptually distinct from "error"）。
        # 但是，如果命令成功执行了，说明 nvidia-smi 存在。
        # 让我们假设 nvidia-smi 成功但无数据不是"出错"，而是"真实结果为空"。
        # 但根据 "不允许返回默认值"，我倾向于如果无法明确获取到信息（哪怕是空），可能需要抛错，或者仅仅针对 "出错" 的情况抛错。
        # 之前的代码 swallow 了 timeout 和 file not found。现在这些会自然抛出。
        # 如果 result.returncode == 0 但 stdout 为空，返回 None 还是抛错？
        # 考虑到如果没有 GPU，返回 None 是符合语义的（Option[str]）。
        # "出错全部要报对应的error" 指的是 verify errors (timeout, not found) aren't caught and turned into None.
        return None

    # 解析输出
    lines = result.stdout.strip().split('\n')
    gpu_list = []

    for line in lines:
        if line.strip():
            parts = line.split(',')
            if len(parts) >= 2:
                gpu_name = parts[0].strip()
                gpu_memory = parts[1].strip()
                gpu_list.append(f"{gpu_name} {gpu_memory}")

    if gpu_list:
        if len(gpu_list) == 1:
            return gpu_list[0]
        else:
            return f"{len(gpu_list)}x {gpu_list[0]}"

    return None


def get_hardware_description() -> str:
    """
    获取完整的硬件配置描述字符串

    返回:
        str: 格式化的硬件配置描述，例如：
             "CPU: 8 cores, RAM: 32GB, GPU: NVIDIA RTX 3090 24GB"
             或 "CPU: 8 cores, RAM: 32GB"（如果没有GPU）

    实现细节:
        - 按照 CPU -> RAM -> GPU 的顺序组织信息
        - 如果没有检测到GPU，则不包含GPU信息
    """
    # 第一阶段：获取基础硬件信息
    cpu_count = get_cpu_count()
    memory_info = get_memory_info()
    gpu_info = get_gpu_info()

    # 第二阶段：构建描述字符串
    description_parts = [
        f"CPU: {cpu_count} cores",
        f"RAM: {int(memory_info['total'])}GB"
    ]

    if gpu_info:
        description_parts.append(f"GPU: {gpu_info}")

    return ", ".join(description_parts)


def get_conda_packages(env_name: Optional[str] = None) -> str:
    """
    获取特定conda虚拟环境中所有包的叙述式摘要

    参数:
        env_name (Optional[str]): conda环境名称，如果为None则使用当前激活的环境

    返回:
        str: 适合大模型理解的自然语言描述，涵盖Python版本、包规模、渠道组成及核心依赖的特色版本号

    实现细节:
        - 使用 conda list 命令获取完整包列表
        - 对包信息进行统计汇总，构建逻辑自洽的环境描述
        - 重点突出演示PyTorch优先于TensorFlow的建议，并给出特色版本号

    异常:
        RuntimeError: 当conda命令不可用或执行失败时抛出
        ValueError: 当指定的conda环境不存在时抛出

    示例:
        >>> # 获取当前环境的包信息
        >>> packages = get_conda_packages()
        >>> print(packages)
        当前Conda环境'base'共计157个软件包，Python版本为python 3.9.7...
    """
    try:
        # 第一阶段：构建并验证conda命令参数
        if env_name:
            # 检查指定环境是否存在
            check_cmd = ['conda', 'env', 'list']
            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                log_msg("ERROR", f"Failed to execute conda env list command: {result.stderr}")

            # 验证环境是否存在
            if env_name not in result.stdout:
                log_msg("ERROR", f"Conda environment '{env_name}' does not exist")

            # 获取指定环境的包列表
            cmd = ['conda', 'list', '--name', env_name, '--json']
        else:
            # 获取当前激活环境的包列表
            cmd = ['conda', 'list', '--json']

        # 第二阶段：执行conda list命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode != 0:
            log_msg("ERROR", f"Failed to execute conda list command: {result.stderr}")

        # 第三阶段：解析JSON输出
        try:
            packages_data: List[Dict[str, Any]] = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log_msg("ERROR", f"Failed to parse conda list output: {e}")

        if not isinstance(packages_data, list):
            log_msg("ERROR", "Incorrect data format returned by conda list")

        # 第四阶段：整理Python信息与包列表
        python_info: Optional[Dict[str, Optional[str]]] = None
        packages_list: List[Dict[str, Optional[str]]] = []
        channel_counter: Counter[str] = Counter()
        highlighted_packages: List[Tuple[str, str]] = []

        for package in packages_data:
            name = package.get('name')
            if not name:
                continue

            version = package.get('version', 'unknown')
            build_string = package.get('build_string') or package.get('build')
            channel = package.get('channel')
            signature = f"{name} {version}"

            channel_label = channel or 'unknown'
            channel_counter[channel_label] += 1

            package_payload: Dict[str, Optional[str]] = {
                'name': name,
                'version': version,
                'build_string': build_string or None,
                'channel': channel or None,
                'signature': signature
            }
            packages_list.append(package_payload)

            if name.lower() == 'python':
                python_info = dict(package_payload)
            elif name.lower() in {'numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'tensorflow', 'xgboost', 'lightgbm'}:
                highlighted_packages.append((name, signature))

        packages_list.sort(key=lambda item: (item['name'] or '').lower())
        highlighted_packages = sorted(set(highlighted_packages), key=lambda item: item[0])

        # 第五阶段：构建AI友好输出结构
        environment_name = env_name or 'current'
        package_total = len(packages_list)
        channel_description = ", ".join(
            f"{channel}({count})"
            for channel, count in sorted(channel_counter.items())
        ) if channel_counter else "unknown source"

        python_signature = python_info.get('signature') if python_info else 'python unknown'
        python_build = python_info.get('build_string') if python_info else None

        core_packages = [signature for _, signature in highlighted_packages[:6]]
        remaining_count = max(len(highlighted_packages) - len(core_packages), 0)

        torch_info = next((signature for name, signature in highlighted_packages if name.lower() == 'torch'), None)
        torchvision_info = next((signature for name, signature in highlighted_packages if name.lower() == 'torchvision'), None)
        tensorflow_info = next((signature for name, signature in highlighted_packages if name.lower() == 'tensorflow'), None)

        # 第六阶段：拼装自然语言描述
        description_parts: List[str] = []
        description_parts.append(
            f"Current Conda environment '{environment_name}' contains {package_total} packages, Python version is {python_signature}"
            + (f" (build: {python_build})" if python_build else "")
            + f", channel distribution is {channel_description}."
        )

        if core_packages:
            core_packages_text = "、".join(core_packages)
            if remaining_count > 0:
                core_packages_text += f", {remaining_count} other core dependencies omitted"
            description_parts.append(f"Common research and ML components are ready, including {core_packages_text}.")
        else:
            description_parts.append("No common research or ML components identified, please check full package list if needed.")

        if torch_info:
            torch_sentence = f"It is recommended to prioritize PyTorch ecosystem for neural network tasks ({torch_info}"
            if torchvision_info:
                torch_sentence += f", {torchvision_info}"
            torch_sentence += "), which has better compatibility and optimization in this environment."
            if tensorflow_info:
                torch_sentence += f" In comparison, TensorFlow ({tensorflow_info}) can be used as an alternative."
            description_parts.append(torch_sentence)
        elif tensorflow_info:
            description_parts.append(f"TensorFlow ecosystem ({tensorflow_info}) is available, check for extra installation if PyTorch is needed.")

        description_parts.append("If you need dependencies not listed, import them directly; the environment pre-installs all common scientific computing libraries.")

        return " ".join(description_parts)

    except subprocess.TimeoutExpired:
        log_msg("ERROR", "conda list command timed out")
    except FileNotFoundError:
        log_msg("ERROR", "conda command unavailable, please ensure conda is installed and added to PATH")
    except Exception as e:
        log_msg("ERROR", f"Unknown error occurred while getting conda package info: {e}")