"""
MLE-Bench Adapter for Swarm-Evo

本模块是 Swarm-Evo 项目与 MLE-Bench 评测框架的适配层。
在 MLE-Bench 容器环境中作为入口点运行，负责：
1. 环境变量适配与配置初始化
2. 工作空间设置
3. Agent 系统初始化
4. 竞赛执行与结果输出
"""

import os
import asyncio
import shutil
import json
from pathlib import Path
from typing import Optional


async def run_adapter():
    """
    MLE-Bench 适配器主入口函数。
    
    执行流程:
        1. 环境变量预设（必须在 import config 之前）
        2. 初始化日志系统
        3. 加载配置
        4. 创建 Agent 系统组件
        5. 运行竞赛
        6. 输出结果文件
    """
    
    # =========================================
    # 第一阶段：环境变量预设
    # 必须在导入 utils.config 之前完成
    # =========================================
    
    competition_id = os.environ.get("COMPETITION_ID", "unknown_competition")
    
    # MLE-Bench 容器环境变量默认值
    # 这些值会被容器传入的环境变量覆盖
    env_defaults = {
        # LLM API 配置 - 需要从容器环境获取或硬编码
        "API_KEY": os.environ.get("API_KEY", "your_api_key_here"),
        "API_BASE": os.environ.get("API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "qwen-max"),
        
        # 日志配置
        "LOG_LEVEL": "INFO",
        "LOG_DIR": "/home/logs",
        
        # 实验配置
        "RANDOM_SEED": "42",
        "MAX_CONCURRENT_AGENTS": "1",
        "AGENT_TIMEOUT": "3600",
        "AGENT_NUM": os.environ.get("AGENT_NUM", "1"),
        "AGENT_CONFIG_DIR": "/home/agent/core/config/agent.json",
        
        # MLE-Bench 配置
        "MLE_BENCH_COMPETITION": competition_id,
        "MLE_BENCH_SERVER_URL": "http://localhost:5000",
        "MLE_BENCH_WORKSPACE_DIR": "/home",
        "MLE_BENCH_TIME_LIMIT": os.environ.get("TIME_LIMIT", "12"),
        "MLE_BENCH_EPOCH_LIMIT": os.environ.get("STEP_LIMIT", "100"),
        "MLE_BENCH_PRIVATE_DATA_DIR": "/private/data",
        
        # Conda 环境
        "CONDA_ENV_NAME": os.environ.get("CONDA_ENV_NAME", "base"),
        
        # 任务执行配置
        "INIT_TASK_NUM": "2",
        "EXPLORE_RATIO": "0.6",
        "EPOCH_TASK_NUM": "1",
    }
    
    for key, value in env_defaults.items():
        if key not in os.environ or os.environ.get(key, "").strip() == "":
            os.environ[key] = value
    
    # =========================================
    # 第二阶段：导入项目模块（在环境变量设置之后）
    # =========================================
    
    from utils.config import get_config
    from utils.logger_system import init_logger, log_msg
    from core.agent.agent_pool import AgentPool
    from core.execution.pipeline import Pipeline
    from core.execution.journal import Journal
    from core.execution.iteration_controller import IterationController
    
    # =========================================
    # 第三阶段：初始化日志系统
    # =========================================
    
    logs_dir = Path("/home/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    init_logger(str(logs_dir))
    log_msg("INFO", "=" * 50)
    log_msg("INFO", "Swarm-Evo MLE-Bench Adapter 启动")
    log_msg("INFO", f"Competition: {competition_id}")
    log_msg("INFO", "=" * 50)
    
    # =========================================
    # 第四阶段：MLE-Bench 工作空间设置
    # =========================================
    
    # MLE-Bench 标准目录结构
    workspace_dir = Path("/home")
    data_dir = workspace_dir / "data"
    submission_dir = workspace_dir / "submission"
    code_dir = workspace_dir / "code"
    
    # 确保输出目录存在
    submission_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    
    log_msg("INFO", f"工作目录: {workspace_dir}")
    log_msg("INFO", f"数据目录: {data_dir}")
    log_msg("INFO", f"提交目录: {submission_dir}")
    
    # =========================================
    # 第五阶段：加载配置
    # =========================================
    
    try:
        config = get_config()
        log_msg("INFO", "配置加载成功")
    except Exception as e:
        log_msg("ERROR", f"配置加载失败: {e}")
        return
    
    # =========================================
    # 第六阶段：读取竞赛描述
    # =========================================
    
    description_content = ""
    description_candidates = [
        workspace_dir / "description.md",  # MLE-Bench 默认位置: /home/description.md
        data_dir / "description.md",
    ]
    
    for desc_path in description_candidates:
        if desc_path.exists():
            try:
                description_content = desc_path.read_text(encoding="utf-8")
                log_msg("INFO", f"竞赛描述已加载: {desc_path}")
                break
            except Exception as e:
                log_msg("WARNING", f"无法读取 {desc_path}: {e}")
    
    if not description_content:
        log_msg("WARNING", "未找到竞赛描述文件，使用默认描述")
        description_content = f"""
# Kaggle Competition: {competition_id}

## Data Location
Competition data is available at: {data_dir}

## Submission
Save your predictions to: {submission_dir / "submission.csv"}
"""
    
    # =========================================
    # 第七阶段：创建 LLM 客户端和 Agent Pool
    # =========================================
    
    try:
        llm = config.create_langchain_llm()
        log_msg("INFO", "LangChain LLM 客户端创建成功")
    except Exception as e:
        log_msg("ERROR", f"LLM 客户端创建失败: {e}")
        return
    
    # 加载 Agent 配置
    agent_config_path = Path("/home/agent/core/config/agent.json")
    if not agent_config_path.exists():
        # 回退到项目内路径
        agent_config_path = Path("core/config/agent.json")
    
    if agent_config_path.exists():
        try:
            with open(agent_config_path, "r", encoding="utf-8") as f:
                agent_config_dict = json.load(f)
            agent_config_dict["conda_env_name"] = config.conda_env_name
            log_msg("INFO", f"Agent 配置加载成功: {agent_config_path}")
        except Exception as e:
            log_msg("ERROR", f"Agent 配置加载失败: {e}")
            return
    else:
        log_msg("WARNING", f"Agent 配置文件不存在: {agent_config_path}，使用默认配置")
        agent_config_dict = {
            "model_type": config.model_name,
            "max_steps": 50,
            "tool_config": [],
            "conda_env_name": config.conda_env_name,
        }
    
    try:
        agent_pool = AgentPool.from_configs(
            agents_num=config.agent_num,
            config=agent_config_dict,
            llm=llm
        )
        log_msg("INFO", f"AgentPool 创建成功，共 {config.agent_num} 个 Agent")
    except Exception as e:
        log_msg("ERROR", f"AgentPool 创建失败: {e}")
        return
    
    # =========================================
    # 第八阶段：创建 Journal 和 Pipeline
    # =========================================
    
    try:
        journal = Journal()
        log_msg("INFO", "Journal 初始化完成")
        
        pipeline = Pipeline(journal)
        pipeline.initialize()
        log_msg("INFO", "Pipeline 初始化完成")
    except Exception as e:
        log_msg("ERROR", f"Journal/Pipeline 初始化失败: {e}")
        return
    
    # =========================================
    # 第九阶段：创建并运行 IterationController
    # =========================================
    
    try:
        controller = IterationController(
            agent_pool=agent_pool,
            task_pipeline=pipeline,
            journal=journal,
            config=config,
            competition_description=description_content
        )
        log_msg("INFO", "IterationController 创建成功")
        
        log_msg("INFO", "开始竞赛执行...")
        await controller.run_competition()
        log_msg("INFO", "竞赛执行完成")
        
    except Exception as e:
        log_msg("ERROR", f"竞赛执行失败: {e}", exc_info=True)
    
    # =========================================
    # 第十阶段：输出结果文件（MLE-Bench 合规）
    # =========================================
    
    log_msg("INFO", "整理输出文件...")
    
    submission_file = submission_dir / "submission.csv"
    
    # 1. 检查提交文件
    # [NEW Logic] 优先尝试从最佳方案恢复
    try:
        best_node = journal.get_best_node()
        if best_node and best_node.archive_path and os.path.exists(best_node.archive_path):
            log_msg("INFO", f"正在从归档恢复最佳方案: {best_node.archive_path}")
            import zipfile
            with zipfile.ZipFile(best_node.archive_path, 'r') as zip_ref:
                zip_ref.extractall(workspace_dir)
            log_msg("INFO", "✅ 最佳方案文件已恢复到工作目录")
    except Exception as e:
        log_msg("WARNING", f"尝试恢复最佳方案失败: {e}")

    if submission_file.exists():
        log_msg("INFO", f"✅ 提交文件存在: {submission_file}")
    else:
        log_msg("WARNING", "❌ 提交文件未生成! 尝试兜底查找...")
        # 尝试从工作目录查找
        for candidate in workspace_dir.glob("**/submission.csv"):
            try:
                shutil.copy2(candidate, submission_file)
                log_msg("INFO", f"从 {candidate} 复制提交文件")
                break
            except Exception as e:
                log_msg("WARNING", f"复制失败: {e}")
    
    # 2. 复制代码到 /home/code
    solution_candidates = [
        workspace_dir / "solution.py",
        workspace_dir / "code" / "solution.py",
    ]
    for src in solution_candidates:
        if src.exists():
            try:
                shutil.copy2(src, code_dir / "solution.py")
                log_msg("INFO", f"已复制 solution.py 到 {code_dir}")
                break
            except Exception as e:
                log_msg("WARNING", f"复制 solution.py 失败: {e}")
    
    # 3. 复制快照目录
    snapshots_src = workspace_dir / ".snapshots"
    if snapshots_src.exists():
        try:
            shutil.copytree(snapshots_src, code_dir / ".snapshots", dirs_exist_ok=True)
            log_msg("INFO", f"已复制 .snapshots 到 {code_dir}")
        except Exception as e:
            log_msg("WARNING", f"复制 .snapshots 失败: {e}")
    
    # 4. 复制 Journal 到日志目录
    journal_src = workspace_dir / "journal.json"
    if journal_src.exists():
        try:
            shutil.copy2(journal_src, logs_dir / "journal.json")
            log_msg("INFO", f"已复制 journal.json 到 {logs_dir}")
        except Exception as e:
            log_msg("WARNING", f"复制 journal.json 失败: {e}")
    
    # 5. 展示最佳结果
    try:
        best_node = journal.get_best_node()
        if best_node:
            log_msg("INFO", f"最佳方案 ID: {best_node.id}, Score: {best_node.score}")
        else:
            log_msg("WARNING", "未找到有效方案")
    except Exception as e:
        log_msg("WARNING", f"获取最佳方案失败: {e}")
    
    log_msg("INFO", "=" * 50)
    log_msg("INFO", "Swarm-Evo MLE-Bench Adapter 结束")
    log_msg("INFO", "=" * 50)


if __name__ == "__main__":
    print("\n🚀 启动 Swarm-Evo MLE-Bench 适配器\n")
    asyncio.run(run_adapter())
