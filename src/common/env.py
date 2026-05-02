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
# 全局配置（单一真源）
# ==========================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_NAME = os.getenv("CACHE_NAME", "semantic-cache")

# 当前实现采用逻辑两级缓存：
# - L1: 进程内快速路径（归一化后的字符串精确命中、near_exact、edit_distance）
# - L2: semantic cache（当前仍由 RedisVL 承载）
# 这不是“L1 单独向量库 + L2 单独键值库”的双库结构，
# 而是同一套运行时中的逻辑双层封装。
CACHE_L1_EXACT_ENABLED = os.getenv("CACHE_L1_EXACT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
CACHE_L1_EDIT_DISTANCE_ENABLED = os.getenv("CACHE_L1_EDIT_DISTANCE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
CACHE_L1_PROMOTION_ENABLED = os.getenv("CACHE_L1_PROMOTION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
# L1 只保留 FAQ pinned 项和被证明足够热的运行时问答。
# 默认阈值为 2，表示某条运行时问答在 L2 被复用两次后，才值得进入 L1 快速路径。
CACHE_L1_PROMOTION_THRESHOLD = int(os.getenv("CACHE_L1_PROMOTION_THRESHOLD", "2"))
# `CACHE_L1_MAX_ENTRIES` 的语义是“L1 总预算”。
# FAQ seed 会以 pinned 形式常驻；剩余预算才分配给运行时热点项。
CACHE_L1_MAX_ENTRIES = int(os.getenv("CACHE_L1_MAX_ENTRIES", "128"))
# 默认保持 1 作为安全基线：
# - 1 足以兜住轻微 typo / OCR 噪声
# - 更大的阈值更容易把不同中文商品问法误判成同题
CACHE_L1_EDIT_DISTANCE_MAX_DISTANCE = int(os.getenv("CACHE_L1_EDIT_DISTANCE_MAX_DISTANCE", "1"))
CACHE_L2_DISTANCE_THRESHOLD = float(os.getenv("CACHE_L2_DISTANCE_THRESHOLD", os.getenv("CACHE_DISTANCE_THRESHOLD", "0.2")))

# 向量存储后端配置。目前代码实现仍落在 Redis；若后续迁移到 Qdrant，可优先替换这两个面向角色的配置。
CACHE_VECTOR_BACKEND = os.getenv("CACHE_VECTOR_BACKEND", "redis")
RAG_VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "redis")

# 向后兼容旧名字，避免现有调用点漂移。
CACHE_DISTANCE_THRESHOLD = CACHE_L2_DISTANCE_THRESHOLD

ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ANALYSIS_MODEL_NAME = os.getenv("ANALYSIS_MODEL_NAME", "ep-m-20260411093114-9hftc")
RESEARCH_MODEL_NAME = os.getenv("RESEARCH_MODEL_NAME", "deepseek-v3-2-251201")
ANALYSIS_INPUT_PRICE_RMB_PER_1K = float(os.getenv("ANALYSIS_INPUT_PRICE_RMB_PER_1K", "0.0006"))
ANALYSIS_OUTPUT_PRICE_RMB_PER_1K = float(os.getenv("ANALYSIS_OUTPUT_PRICE_RMB_PER_1K", "0.0036"))
ANALYSIS_CACHED_INPUT_PRICE_RMB_PER_1K = float(os.getenv("ANALYSIS_CACHED_INPUT_PRICE_RMB_PER_1K", "0.00012"))
RESEARCH_INPUT_PRICE_RMB_PER_1K = float(os.getenv("RESEARCH_INPUT_PRICE_RMB_PER_1K", "0.0020"))
RESEARCH_OUTPUT_PRICE_RMB_PER_1K = float(os.getenv("RESEARCH_OUTPUT_PRICE_RMB_PER_1K", "0.0030"))
RESEARCH_CACHED_INPUT_PRICE_RMB_PER_1K = float(os.getenv("RESEARCH_CACHED_INPUT_PRICE_RMB_PER_1K", "0.00040"))


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
            raise ValueError(
                "Critical Error: 缺少 'ARK_API_KEY' 环境变量。"
                "请在 .env 或部署平台控制台配置后重试。"
            )

