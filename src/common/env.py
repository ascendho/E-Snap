import os
import getpass
from dotenv import load_dotenv, find_dotenv

# ==========================================
# 核心环境与安全密钥加载工具
# ==========================================

def load_env():
    """
    加载 .env 配置文件到当前进程的系统环境变量中。
    """
    load_dotenv(find_dotenv())

load_env()

# ==========================================
# 全局配置 (Single Source of Truth)
# ==========================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_NAME = os.getenv("CACHE_NAME", "semantic-cache")

# 当前实现采用逻辑两级缓存：
# - L1: exact fast path（归一化后的字符串精确命中）
# - L2: semantic cache（当前仍由 RedisVL 承载）
CACHE_L1_EXACT_ENABLED = os.getenv("CACHE_L1_EXACT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
CACHE_L2_DISTANCE_THRESHOLD = float(os.getenv("CACHE_L2_DISTANCE_THRESHOLD", os.getenv("CACHE_DISTANCE_THRESHOLD", "0.2")))

# 向量存储后端配置。目前代码实现仍落在 Redis；若后续迁移到 Qdrant，可优先替换这两个面向角色的配置。
CACHE_VECTOR_BACKEND = os.getenv("CACHE_VECTOR_BACKEND", "redis")
RAG_VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "redis")

# 向后兼容旧名字，避免现有调用点漂移。
CACHE_DISTANCE_THRESHOLD = CACHE_L2_DISTANCE_THRESHOLD

ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ANALYSIS_MODEL_NAME = os.getenv("ANALYSIS_MODEL_NAME", "ep-m-20260411093114-9hftc")
RESEARCH_MODEL_NAME = os.getenv("RESEARCH_MODEL_NAME", "deepseek-v3-2-251201")


# ==========================================
# 环境变量解析工具
# ==========================================

def to_bool_env(name: str, default: bool = False) -> bool:
    """读取并安全解析布尔类型的环境变量。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def set_ark_key():
    """
    确保火山引擎（ARK）大模型的 API Key 可用；若缺失则触发交互式安全输入或报错。
    """
    if not os.getenv("ARK_API_KEY"):
        import sys
        if sys.stdin.isatty():
            os.environ["ARK_API_KEY"] = getpass.getpass("请输入你的 ARK API key: ")
        else:
            raise ValueError("Critical Error: 线上环境缺少 'ARK_API_KEY' 或 'REDIS_URL' 环境变量配置！请在部署平台控制台添加。")

