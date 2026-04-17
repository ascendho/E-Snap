import re
from pathlib import Path
from redisvl.utils.vectorize import HFTextVectorizer
from knowledge.indexer import create_knowledge_base_from_texts, _split_markdown_into_structured_chunks
from common.env import REDIS_URL # 统一从 env.py 中读取 Redis 链接

RAW_DOCS_MD_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_docs.md"

def init_app_knowledge_base():
    """
    初始化并构建应用程序的知识库。
    """
    embeddings = HFTextVectorizer(model="BAAI/bge-large-zh-v1.5")
    markdown_text = RAW_DOCS_MD_PATH.read_text(encoding="utf-8")
    
    # 强制知识库建设必须以结构化数组为入口
    raw_docs = _split_markdown_into_structured_chunks(markdown_text)

    _, _, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,
        source_id="customer_support_docs",
        redis_url=REDIS_URL,
    )

    return kb_index, embeddings
