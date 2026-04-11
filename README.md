# 带有语义缓存的高效 AI Agent

本项目实现了一个基于 LangGraph + RedisVL 语义缓存的 Agent 工作流。
目标是通过语义命中复用，降低重复问题的 LLM 成本并提升响应效率。

## 前置要求

1. Python 3.11+（推荐）
2. 本机已安装并可启动 Redis（监听 `localhost:6379`）
3. 可用的 ARK API Key

## 快速启动

### 1) 创建并激活虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 安装并启动 Redis（针对 macOS 系统）

如果你用 Homebrew：

```bash
brew install redis
brew services start redis
redis-cli ping
```

`redis-cli ping` 返回 `PONG` 表示 Redis 已正常启动。

如果你不想用服务模式，也可以前台启动：

```bash
redis-server
```

### 4) 配置环境变量（`.env`）

在项目根目录新建 `.env` 文件，写入以下内容：

```env
# Redis
REDIS_URL=redis://localhost:6379
CACHE_NAME=semantic-cache
CACHE_DISTANCE_THRESHOLD=0.3
CACHE_TTL_SECONDS=3600

# 运行必需：火山引擎 ARK
ARK_API_KEY=your_ark_api_key_here

# 可选：如果你使用 Tavily 检索
TAVILY_API_KEY=your_tavily_api_key_here

# 可选：LangSmith 链路追踪
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
```

注意：API Key 直接填纯文本，不要额外加引号。

### 5) 运行项目

```bash
python main.py
```

可选：如果你希望同时在控制台打印每个场景的结果：

```bash
SHOW_CONSOLE_RESULTS=true python main.py
```

## 输出结果

运行后会在 `outputs/` 下生成：

1. 场景汇总 CSV
2. LLM 调用明细 CSV
3. 全量结果 JSON

说明：以上文件名为固定名称（`run_summary.csv`、`llm_usage.csv`、`run_results.json`），每次运行会覆盖上一次结果，不会累积历史文件。

## 当前结构（核心）

1. `main.py`: 唯一入口，负责启动流程
2. `app/bootstrap.py`: 知识库初始化
3. `cache/bootstrap.py`: 语义缓存初始化（含 `hydrate` 预热）
4. `app/scenarios.py`: 场景定义
5. `app/workflow_runner.py`: 场景执行、成本统计、导出与分析
6. `agent/`: LangGraph 节点、边、工具与工作流编排
7. `cache/`: 语义缓存封装、评估与配置
