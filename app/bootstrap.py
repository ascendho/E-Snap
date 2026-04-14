import re
from pathlib import Path

from redisvl.utils.vectorize import HFTextVectorizer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from agent.knowledge_base_utils import create_knowledge_base_from_texts, _split_markdown_into_structured_chunks

RAW_DOCS_MD_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_docs.md"

def init_app_knowledge_base():
    """构建演示用知识库并返回索引与向量模型。"""
    embeddings = HFTextVectorizer(model="BAAI/bge-large-zh-v1.5")
    markdown_text = RAW_DOCS_MD_PATH.read_text(encoding="utf-8")
    raw_docs = _split_markdown_into_structured_chunks(markdown_text)

    _, _, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,
        source_id="customer_support_docs",
        redis_url="redis://localhost:6379",

        # 已经在文本切块阶段做了分层结构化，入库时不需要再切块了，否则会破坏原有的层级信息。
        skip_chunking=True,
    )

    # embeddings 是一个加载了神经网络权重（MiniLM 模型）的 Python 对象。
    # 作用：它提供了一个 embed_query() 方法，可以将输入文本转换为向量（数字数组），以便在 Redis 向量数据库中进行相似度搜索。

    # kb_index 是一个“检索器对象”或者说是“向量数据库的索引连接器”。
    # 它脑子里记住了整个 Redis 数据库中所有文档片段（Chunks）的位置。
    # 作用：你给它一串数字（向量），它跑到 Redis 里算出最接近的几个文本块，然后原封不动地拿回来给你。
    return kb_index, embeddings
