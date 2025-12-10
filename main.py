import asyncio
import shutil
import os
from pathlib import Path

from utils.config import get_config
from utils.system_info import get_conda_packages
from utils.build_workspace import build_workspace
from utils.logger_system import LoggerSystem

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
        logger = LoggerSystem(log_dir)
        logger.text_log("INFO", "Logger initialized")
        print("✅ 日志系统构建成功")
    except Exception as e:
        print(f"Logger构建失败: {e}")
        return

    try:
        # 第三阶段：验证环境配置
        logger.text_log("INFO", "\n[3/7] 验证环境配置...")
        # config已在开头加载
        is_valid, error_msg = config.validate()
        if not is_valid:
            logger.text_log("ERROR", f"❌ 配置验证失败: {error_msg}")
            logger.text_log("WARNING", "提示: 请确保.env文件中配置了必要的API密钥")
            return
        logger.text_log("INFO", "✅ 环境配置验证通过")
    except Exception as e:
        logger.text_log("ERROR", f"环境配置失败: {e}")

    try:
        # 第四阶段：获取系统信息
        logger.text_log("INFO", "\n[4/7] 获取系统环境信息...")

        conda_packages = get_conda_packages(config.conda_env_name)
        logger.text_log("INFO", f"✅ Conda环境 '{config.conda_env_name}' 包信息获取成功")
    except Exception as e:
        logger.text_log("ERROR", f"获取系统环境信息失败: {e}")
        return
    
    try:
        # 第五阶段：创建AgentPool
        logger.text_log("INFO", "\n[5/7] 创建AgentPool...")
        agent_pool = AgentPool(llm_client=llm_client, logger=logger)
        logger.text_log("INFO", "✅ AgentPool 创建成功")
    except Exception as e:
        logger.text_log("ERROR", f"AgentPool 创建失败: {e}")
        return
    

if __name__ == "__main__":
    print("\n🚀 启动MLE-bench竞赛自主执行系统\n")
    asyncio.run(main_mle_bench_competition())