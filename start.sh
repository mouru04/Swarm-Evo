#!/bin/bash

# 切换到 agent 目录，确保相对路径正确
cd /home/agent

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:/home/agent

# 初始化 conda shell 功能，使 conda activate 在非交互式脚本中生效
eval "$(conda shell.bash hook)"
conda activate agent

# Create output directories immediately to prevent "Not Found" errors if the script crashes
mkdir -p /home/code
mkdir -p /home/logs

# Run the adapter script
python /home/agent/run_mle_adapter.py
