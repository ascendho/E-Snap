# 智能客服 AI Agent

一个基于 LangGraph 和 FastAPI 构建的企业级、高效率 AI 智能体，支持多领域架构，并通过 RedisVL 实现语义缓存。

本项目展示了构建生产级大模型（LLM）应用的完整方案，将前端展示与智能后端编排进行了无缝整合。

## 🌟 核心特性
- **语义缓存 (Semantic Caching):** 利用 RedisVL 和专用的嵌入模型 (`ep-m-2026...`) 对语义相似的问题进行缓存和快速返回，大幅降低 Token 消耗和响应延迟。
- **工作流编排 (Workflow Orchestration):** 基于 LangGraph 驱动，Agent 在回复用户前能够进行复杂的推理、迭代反思以及多步检索研究。
- **全栈集成 (Full-Stack Integration):** FastAPI 一体化地提供复杂的后端 AI 逻辑服务和现代化的毛玻璃风格前端页面。无需处理跨域（CORS）烦恼，大大降低部署复杂度。
- **高品质用户体验:** 进阶版 UI 支持暗黑/明亮模式智能切换，拥有实时的发光 "Thinking" 思考状态提示，以及精美的组件。

## 📂 架构与目录结构
有别于常规的玩具项目，本代码库拥抱领域驱动设计（DDD）。我们不是简单地将代码分为 `frontend` 和 `backend`，**`src/` 目录正是整个应用的大脑核心。**

```text
.
├── frontend/             # 前端静态资源 (HTML, CSS, JS 及毛玻璃特效 UI)
├── data/                 # 数据集 (包含电商客服白皮书知识库等)
├── tests/                # 测试套件 (测试代码和评测脚本)
└── src/                  # 核心后端逻辑 (Python)
    ├── api/              # 网关与路由 (FastAPI 服务器、路由挂载)
    ├── workflow/         # 智能工作流 (LangGraph Node、Edge、状态和提示词)
    ├── cache/            # 缓存与性能 (RedisVL 语义缓存集成)
    ├── knowledge/        # 知识库层 (文档解析机制、Embedding 嵌入)
    └── common/           # 基础设施层 (日志配置、环境变量安全)
```

## 🚀 极速启动与部署指南

### 1. 环境准备
- **Python 3.11+**
- **Redis Stack Server** (重要：由于本系统依赖 RedisVL 提供的向量和语义搜索能力，普通的 `redis` 无法运行。请必须安装 `redis-stack` 或带有 `search` 扩展的 Redis 服务器，例如 macOS 下使用 `brew install redis-stack` 后用 `redis-stack-server` 启动，或直接通过 Docker 运行 `redis/redis-stack:latest`)
- **大模型 API Key**（如: 火山引擎/豆包模型 ARK_API_KEY，或其他接入渠道）

### 2. 安装依赖
建议使用虚拟环境以避免依赖冲突：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置环境变量
复制根目录下的示例环境变量文件并填入您的真实密钥：
```bash
cp .env.example .env
```
（确保配置了对应的 LLM 接口密钥及 REDIS 访问地址）。

### 4. 启动统一服务
整个前端体系已纯净挂载入 FastAPI 应用中，您仅需一条命令即可启动完整的全栈系统！
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```

### 5. 开始体验
打开浏览器访问 **[http://127.0.0.1:8000](http://127.0.0.1:8000)** 即可开始与您的智能客服智能体对话！

### 6. 运行测试
PYTHONPATH=. .venv/bin/python src/main.py

---

## 📚 知识库更新指南 (重要)

如果在此后需要修改或扩展机器人的知识覆盖范围（例如修改了 `data/raw_docs.md` 平台白皮书），您需要执行以下操作来使修改生效：
1. **清理旧版缓存**: 删除旧的向量存储/语义缓存以防拿到脏数据。
2. **重启系统**: 重启 `uvicorn` 服务，以便 Agent 重新读取和预处理更新后的 `raw_docs.md`。

