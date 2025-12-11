"""模板文件处理工具模块。

目前仅负责构建 MLE-bench workspace 目录与说明文档。
"""

import shutil
from pathlib import Path
from utils.logger_system import log_msg
from typing import Optional
from .config import Config
from .system_info import get_hardware_description

def build_workspace(
    config: Config,
    workspace_dir: Optional[str] = None,
    competition: Optional[str] = None
) -> str:
    """
    构建完整的 MLE-bench workspace 目录结构

    参数:
        config: Config 配置对象（包含所有 MLE-bench 相关配置）
        workspace_dir: workspace 目标目录路径（可选，默认使用 config.mle_bench_workspace_dir）
        competition: 竞赛名称（可选，默认使用 config.mle_bench_competition）

    返回:
        str: description.md 的内容

    异常:
        FileNotFoundError: 当源文件或目录不存在时抛出
        ValueError: 当参数无效时抛出

    工作空间结构:
        ./workspace/
        ├── description.md            # 竞赛描述(只读) - 从竞赛数据复制
        ├── /data/                    # 竞赛数据(只读) - 从竞赛数据复制
        ├── /submission/              # 提交输出(读写)
        ├── /logs/                    # 日志输出(读写)
        └── /code/                    # 代码输出(读写)
    """
    # 第一阶段：从 Config 对象获取配置
    workspace_dir = workspace_dir or config.mle_bench_workspace_dir
    competition = competition or config.mle_bench_competition
    workspace_path = Path(workspace_dir).expanduser()

    # 第二阶段：定位竞赛数据目录（兼容不同数据组织形式）
    public_data_dir = Path("dataset/public").expanduser()
    candidate_data_dirs = [
        public_data_dir / competition / 'prepared' / 'public',
        public_data_dir / competition,
        public_data_dir
    ]
    competition_data_dir = None
    for candidate in candidate_data_dirs:
        if candidate.exists():
            competition_data_dir = candidate
            break
    if competition_data_dir is None:
        log_msg("ERROR", f"无法在以下目录中找到竞赛数据: {[str(path) for path in candidate_data_dirs]}")

    # 第三阶段：验证源目录和文件存在性
    competition_path = competition_data_dir
    description_source = competition_path / "description.md"
    if not description_source.exists():
        log_msg("ERROR", f"description.md 不存在: {description_source}")

    # 第四阶段：创建 workspace 目录结构
    workspace_path.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    data_dir = workspace_path / "data"
    submission_dir = workspace_path / "submission"
    logs_dir = workspace_path / "logs"
    code_dir = workspace_path / "code"

    data_dir.mkdir(exist_ok=True)
    submission_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    code_dir.mkdir(exist_ok=True)

    # 第五阶段：复制 description.md
    description_target = workspace_path / "description.md"
    shutil.copy2(description_source, description_target)

    # 第六阶段：复制竞赛数据文件到 data/ 目录
    for item in competition_path.iterdir():
        if item.name != "description.md":
            target_path = data_dir / item.name
            if item.is_file():
                shutil.copy2(item, target_path)
            elif item.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(item, target_path)

    # 第七阶段：读取并返回 description.md 内容
    with open(description_source, 'r', encoding='utf-8') as f:
        description_content = f.read()

    return description_content