import asyncio
import shutil
from pathlib import Path

from utils.config import get_config
from utils.system_info import get_conda_packages
from utils.build_workspace import build_workspace

async def main_mle_bench_competition() -> None:
    """
    MLE-bench竞赛主执行函数

    执行流程:
        1. 验证环境配置
        2. 获取系统信息（conda包等）
        3. 加载Agent配置（draft, debug, improve）
        4. 创建AgentPool
        5. 创建IterationController并运行竞赛
        6. 展示执行结果
    """
    
    # 删除workspace目录
    try:
        shutil.rmtree(Path("workspace"))
    except FileNotFoundError:
        pass

    try:
        # 第一阶段：验证环境配置
        print("\n[1/7] 验证环境配置...")
        config = get_config()
        is_valid, error_msg = config.validate()
        if not is_valid:
            print(f"❌ 配置验证失败: {error_msg}")
            print("提示: 请确保.env文件中配置了必要的API密钥")
            return
        print("✅ 环境配置验证通过")
    except:
        print("环境配置失败")

    try:
        # 第二阶段：获取系统信息
        print("\n[2/7] 获取系统环境信息...")

        conda_packages = get_conda_packages(config.conda_env_name)
        print(f"✅ Conda环境 '{config.conda_env_name}' 包信息获取成功")
    except Exception as e:
        print(f"获取系统环境信息失败: {e}")
        return
    
    try:
        # 第三阶段：构建workspace
        print("\n[3/7] 构建workspace...")
        description_content = build_workspace(config)
        print(f"✅ workspace 构建成功: {config.mle_bench_workspace_dir}")
    except Exception as e:
        print(f"workspace 构建失败: {e}")
        return

if __name__ == "__main__":
    print("\n🚀 启动MLE-bench竞赛自主执行系统\n")
    asyncio.run(main_mle_bench_competition())