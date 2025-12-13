import asyncio
import shutil
import os
import json
from pathlib import Path

from utils.config import get_config
from utils.system_info import get_conda_packages
from utils.build_workspace import build_workspace
from utils.logger_system import LoggerSystem, init_logger, logger, log_msg
from core.agent.agent_pool import AgentPool
from core.execution.pipeline import Pipeline
from core.execution.journal import Journal
from core.execution.iteration_controller import IterationController
from utils.logger_system import logger as global_logger 


async def main_mle_bench_competition() -> None:
    """
    MLE-bench竞赛主执行函数

    执行流程:
        1. 构建workspace
        2. 构建并初始化日志系统
        3. 验证环境配置
        4. 获取系统信息（conda包等）
        5. 创建AgentPool
        6. 创建IterationController并运行竞赛
        7. 展示执行结果
    """
    
    # 提前加载配置，因为构建workspace也需要config
    try:
        config = get_config()
    except Exception as e:
        print(f"加载配置失败: {e}")
        return

    # 删除workspace目录
    try:
        shutil.rmtree(Path("workspace"))
    except FileNotFoundError:
        pass

    try:
        # 第一阶段：构建workspace
        print("\n[1/7] 构建workspace...")
        description_content = build_workspace(config)
        print(f"✅ workspace 构建成功: {config.mle_bench_workspace_dir}")
    except Exception as e:
        print(f"workspace 构建失败: {e}")
        return

    try:
        # 第二阶段：构建并初始化日志系统
        print("\n[2/7] 构建并初始化日志系统...")
        log_dir = os.path.join(config.mle_bench_workspace_dir, "logs")
        init_logger(log_dir)
        log_msg("INFO", "Logger initialized")
        log_msg("INFO", "✅ 日志系统构建成功")
    except Exception as e:
        print(f"Logger构建失败: {e}")
        return

    try:
        # 第三阶段：验证环境配置
        log_msg("INFO", "\n[3/7] 验证环境配置...")
        # config已在开头加载
        is_valid, error_msg = config.validate()
        if not is_valid:
            log_msg("ERROR", f"❌ 配置验证失败: {error_msg}")
            log_msg("WARNING", "提示: 请确保.env文件中配置了必要的API密钥")
            return
        log_msg("INFO", "✅ 环境配置验证通过")
    except Exception as e:
        log_msg("ERROR", f"环境配置失败: {e}")

    try:
        # 第四阶段：获取系统信息
        log_msg("INFO", "\n[4/7] 获取系统环境信息...")

        conda_packages = get_conda_packages(config.conda_env_name)
        log_msg("INFO", f"✅ Conda环境 '{config.conda_env_name}' 包信息获取成功")
    except Exception as e:
        log_msg("ERROR", f"获取系统环境信息失败: {e}")
        return
    
    try:
        # 第五阶段：创建AgentPool
        log_msg("INFO", "\n[5/7] 创建AgentPool...")
        
        # 1. 初始化 LLM 客户端
        llm_client = config.create_llm_client()

        # 2. 读取 Agent 配置
        agent_config_path = Path("core/config/agent.json")
        if not agent_config_path.exists():
             log_msg("ERROR", f"Agent配置文件未找到: {agent_config_path}")
        
        with open(agent_config_path, "r", encoding="utf-8") as f:
            agent_config_dict = json.load(f)

        # 3. 补充环境配置
        agent_config_dict["conda_env_name"] = config.conda_env_name

        # 4. 批量创建并注册
        agent_pool = AgentPool.from_configs(
            agents_num=config.agent_num,
            config=agent_config_dict,
            llm_client=llm_client
        )
        
        log_msg("INFO", f"✅ AgentPool 创建成功, 已注册 {config.agent_num} 个 Agent")
    except Exception as e:
        log_msg("ERROR", f"AgentPool 创建失败: {e}")
        return

    try:
        # 第六阶段：创建IterationController并运行竞赛
        log_msg("INFO", "\n[6/7] 启动竞赛循环...")

        # 1. 创建 Journal
        journal = Journal()
        log_msg("INFO", "✅ Journal 初始化完成")

        # 2. 创建 Pipeline
        pipeline = Pipeline(journal)
        pipeline.initialize()
        log_msg("INFO", "✅ Pipeline 初始化完成")

        # 3. 创建 Controller
        controller = IterationController(
            agent_pool=agent_pool,
            task_pipeline=pipeline,
            journal=journal,
            config=config,
            competition_description=description_content
        )

        # 4. 运行
        await controller.run_competition()
        log_msg("INFO", "✅ 竞赛执行完成")

    except Exception as e:
        log_msg("ERROR", f"竞赛执行失败: {e}")
        return

    try:
        # 第七阶段：展示结果 (占位)
        log_msg("INFO", "\n[7/7] 结果展示...")
        # TODO: 从 journal 读取并展示最佳结果
        best_node = journal.get_best_node()
        if best_node:
             log_msg("INFO", f"最佳方案 ID: {best_node.id}, Score: {best_node.score}")
        else:
             log_msg("WARNING", "未找到有效方案")

    except Exception as e:
        log_msg("ERROR", f"结果展示失败: {e}")

    

if __name__ == "__main__":
    print("\n🚀 启动MLE-bench竞赛自主执行系统\n")
    asyncio.run(main_mle_bench_competition())