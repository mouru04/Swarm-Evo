# Swarm-Evo Agents

## 简介

Swarm-Evo Agents 是一个开源的代码库。起作用是构建给予群体智能与进化算法想结合的Agent系统以解决传统Agent系统在面对复杂代码问题时的局限性。Swarm-Evo Agents 是一个开源的多智能体系统（Multi-Agent System, MAS）框架，旨在将 群体智能（Swarm Intelligence） 与 进化算法（Evolutionary Algorithms） 深度融合，构建可在复杂代码任务中持续进化、自我优化的智能体集群。

该框架旨在突破传统单智能体或静态规则驱动系统在处理大规模、结构化、多阶段代码问题时的瓶颈，通过 进化式策略迭代、基因级别的模型合成与群体行为协作，实现更稳健、更高性能的自动化编程与机器学习工程能力。

## 测试基准

我们所构建的Swarm-Evo Agents框架将在以下两个benchmark进行性能测试：

### MLE-bench

MLE-bench 是由 OpenAI 构建的机器学习工程能力评估基准，涵盖 75 个真实 Kaggle 竞赛，要求 Agent 能完整执行从数据理解、特征工程到模型训练与提交文件生成的全流程，并在标准化的 Docker 容器环境中按照严格的提交格式完成自动化评测。其竞赛数据基于人类选手的排行榜建立评测基线，当前最佳系统（o1-preview + AIDE）在 pass@1 下达到 16.9% 奖牌率，通过 pass@8 可进一步提升至 34.1%。为了适应该任务，系统通常需要包含数据分析、特征工程、建模与优化等多类协作 Agent，各自负责数据分布理解、特征构造、模型选择训练与参数调优等能力；此外官方还提供了 Lite 版本数据集（22 个竞赛、约 158GB）以及完整的 Agent 开发技术规范供开发者遵循。

### SWE-bench

SWE-bench 是一个针对软件工程问题修复的标准化评测基准，收录 2,294 个真实 GitHub issue，需要 Agent 能理解多文件代码库、定位缺陷、生成修复 patch 并通过单元测试的自动化验证，因此系统通常由代码理解、Bug 定位、代码修复与测试等协作 Agent 共同完成任务。评测完全自动化，使用明确的成功率指标衡量性能；目前排行榜上最佳系统在 Verified 子集上约实现 45% 的解决率，而单一大模型的基线水平大约为 20%，体现出多 Agent 协作在代码修复场景中的显著优势。

## 技术栈

Swarm-Evo Agents 框架基于 Python 3.12 开发，使用了以下技术栈：


| 类别 | 工具/库 | 说明 |
| :--- | :--- | :--- |
| 语言 | Python 3.12 | 主要开发语言（严格版本要求） |
| Agent框架 | LangGraph | 多Agent协作框架 |
| LLM接口 | Openai接口 | 所有兼容OpenAI格式的模型 |
| 并发机制 | asyncio | Python标准异步执行框架 |
| 依赖管理 | miniconda3 | Python包管理器 |

## 环境配置

1. 虚拟环境设置

```bash
# 进入项目目录
cd Swarm-Evo

conda create -n swarm-evo python=3.12 -y
conda activate swarm-evo

# 同步安装依赖包
pip install -r requirements.txt

## 拉取数据
pip install dvc
dcv pull 

# 安装agent环境供Agent系统使用
conda create -n agent python=3.10 -y
pip install -r requirements_agent.txt #linux系统，如果是mac系统需要自己准备requirement.txt文件
```

2. 环境变量配置
项目使用.env文件管理环境变量。复制.env.example创建你的配置文件：

```bash
cp .env.example .env
```

3. 配置MLE-bench

配置MLE-bench需要下载一个特殊的库`https://github.com/openai/mle-bench.git`

```bash
# 下载mle-bench
cd ..
git clone https://github.com/openai/mle-bench.git
cd mle-bench

# 构建基础镜像（如果遇到错误GPT一下，那过错误是要修改一下Dockerfile;基础镜像只需要构建一次）
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .

# 从 https://www.kaggle.com/account 下载kaggle.json
# 将凭证文件放置到指定目录
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 下载mle-bench数据集
conda create -n mlebench python=3.11
conda activate mlebench
pip install -e .
mlebench prepare --lite

# 构建Swarm-Evo镜像 （每一次更新代码都需要重新构建镜像）
rsync -av --progress --exclude='workspace' --exclude='workspace/.dvc' ../Swarm-Evo/ ./agents/swarm-evo/
docker build --no-cache -t swarm-evo ./agents/swarm-evo

# 运行程序
API_KEY="Your-api-key"  API_BASE="https://open.bigmodel.cn/api/coding/paas/v4"  MODEL_NAME="glm-4.6" python run_agent.py --agent-id swarm-evo --competition-set experiments/splits/low.txt --n-workers 4
```


