import re
"""
知识库工具模块。
提供从文本快速构建 Redis 向量知识库的简化能力。
"""

import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import redis
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.index import SearchIndex

# 初始化日志记录器
logger = logging.getLogger("kb-utils")

class KnowledgeBaseManager:
    """管理基于 Redis 的知识库索引及其生命周期。"""
    
    def __init__(self, redis_client: redis.Redis, embeddings: Optional[HFTextVectorizer] = None):
        """
        初始化知识库管理器。
        
        Args:
            redis_client: Redis 客户端实例。
            embeddings: 向量化工具（如果为 None，则默认使用 HuggingFace 的轻量级模型）。
        """
        self.redis_client = redis_client
        # 默认使用 sentence-transformers 模型进行文本向量化
        self.embeddings = embeddings or HFTextVectorizer(model="BAAI/bge-large-zh-v1.5")
        # 用于跟踪当前活跃的索引（以 URL 的哈希值为键）
        self.active_indexes = {} 
    
    def create_knowledge_base(
        self, 
        source_id: str, 
        content, 
        chunk_size: int = 2500, 
        chunk_overlap: int = 250,
        skip_chunking: bool = False
    ) -> Tuple[bool, str, Optional[SearchIndex]]:
        """
        从输入内容创建知识库向量索引。
        
        Args:
            source_id: 内容来源标识符（如 URL 或自定义 ID）。
            content: 输入内容。可以是字符串（将自动切片）或字符串列表（直接使用）。
            chunk_size: 文本切片的大小（字符数）。
            chunk_overlap: 切片之间的重叠大小。
            skip_chunking: 如果为 True，则将整段字符串视为单个切片。
            
        Returns:
            元组 (是否成功, 提示消息, 搜索索引对象)
        """
        try:
            if not content:
                return False, "No content to process", None
            
            # 基于 source_id 生成 8 位 MD5 哈希作为索引名，确保 Redis 键名合法且不冲突
            source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
            index_name = f"kb-{source_hash}"
            
            # --- 步骤 1: 文本预处理与切片 ---
            if isinstance(content, list):
                # 如果已经是列表，支持：
                # 1) ["text1", "text2"]
                # 2) [{"content": "...", "metadata": {...}}]
                normalized_chunks = []
                for item in content:
                    if isinstance(item, str):
                        normalized_chunks.append({"content": item, "metadata": {}})
                    elif isinstance(item, dict):
                        chunk_content = str(item.get("content", "") or "").strip()
                        if not chunk_content:
                            continue
                        raw_meta = item.get("metadata", {})
                        metadata = raw_meta if isinstance(raw_meta, dict) else {}
                        normalized_chunks.append({"content": chunk_content, "metadata": metadata})
                    else:
                        logger.warning(f"列表中不支持的块类型: {type(item)}")

                text_chunks = normalized_chunks
                has_structured_metadata = any(chunk.get("metadata") for chunk in text_chunks)
                source_type = "structured_list" if has_structured_metadata else "text_list"
                logger.info(f"使用提供的 {len(text_chunks)} 个文本块列表")
            elif isinstance(content, str):
                if skip_chunking:
                    # 不切片，直接作为单条记录
                    text_chunks = [{"content": content, "metadata": {}}]
                    source_type = "text_single"
                    logger.info("将整个文本作为一个数据块")
                else:
                    # 使用 LangChain 的递归字符切片器，尝试按段落、句子保留语境
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    
                    docs = [Document(page_content=content, metadata={"source_id": source_id, "source": "text"})]
                    doc_chunks = splitter.split_documents(docs)
                    text_chunks = [
                        {"content": chunk.page_content, "metadata": chunk.metadata or {}}
                        for chunk in doc_chunks
                    ]
                    source_type = "text_chunked"
                    logger.info(f"将文本切分为 {len(text_chunks)} 个数据块")
            else:
                return False, f"Unsupported content type: {type(content)}", None
            
            # --- 步骤 2: 定义 Redis 向量索引 Schema ---
            # 包含：文本内容、来源 ID、来源类型、切片索引以及 1024 维的向量字段
            schema = {
                "index": {"name": index_name, "prefix": f"kb:{source_hash}:"},
                "fields": [
                    {"name": "content", "type": "text"},          # 原始文本内容
                    {"name": "source_id", "type": "tag"},         # 来源标签
                    {"name": "source_type", "type": "tag"},       # 类型标签
                    {"name": "header_1", "type": "text"},         # Markdown 一级标题
                    {"name": "header_2", "type": "text"},         # Markdown 二级标题
                    {"name": "header_3", "type": "text"},         # Markdown 三级标题
                    {"name": "is_announcement", "type": "tag"},   # 是否公告块
                    {"name": "chunk_index", "type": "numeric"},   # 切片顺序编号
                    {
                        "name": "content_vector",
                        "type": "vector",
                        "attrs": {
                            "dims": 1024,                  # 向量维度 (bge-large-zh-v1.5)
                            "distance_metric": "cosine",  # 距离算法：余弦相似度
                            "algorithm": "hnsw",          # 算法：HNSW (适合高维度、高性能检索)
                            "datatype": "float32",
                        },
                    },
                ],
            }
            
            # --- 步骤 3: 在 Redis 中创建并初始化索引 ---
            kb_index = SearchIndex.from_dict(schema, redis_client=self.redis_client)
            # overwrite=True 确保每次处理同一个 URL 时会覆盖旧的索引数据
            kb_index.create(overwrite=True)
            
            # --- 步骤 4: 生成嵌入向量（Embedding）并装载数据 ---
            payload = []
            for i, chunk in enumerate(text_chunks):
                try:
                    text_chunk = str(chunk.get("content", "") or "").strip()
                    if not text_chunk:
                        continue
                    metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                    # 调用模型将文本转为向量字节流
                    embedding = self.embeddings.embed(text_chunk, as_buffer=True)
                    payload.append({
                        "content": text_chunk,
                        "source_id": source_id,
                        "source_type": source_type,
                        "header_1": str(metadata.get("header_1", "") or ""),
                        "header_2": str(metadata.get("header_2", "") or ""),
                        "header_3": str(metadata.get("header_3", "") or ""),
                        "is_announcement": "true" if bool(metadata.get("is_announcement", False)) else "false",
                        "chunk_index": i,
                        "content_vector": embedding,
                    })
                except Exception as e:
                    logger.warning(f"数据块 {i} 向量化失败: {e}")
                    continue
            
            if not payload:
                return False, "Failed to create embeddings for content", None
            
            # 将封装好的负载批量写入 Redis
            kb_index.load(payload)
            
            # 记录到活跃索引字典中，方便管理
            self.active_indexes[source_hash] = {
                "index": kb_index,
                "source_id": source_id,
                "source_type": source_type,
                "chunks": len(text_chunks),
                "created_at": time.time()
            }
            
            success_msg = f"✅ 知识库创建成功，包含 {len(text_chunks)} 个数据块 ({source_type})"
            logger.info(success_msg)
            return True, success_msg, kb_index
            
        except Exception as e:
            error_msg = f"❌ Knowledge base creation failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    

def create_knowledge_base_from_texts(
    texts: List[Union[str, Dict[str, Any]]],
    source_id: str = "custom_texts",
    redis_url: str = "redis://localhost:6379",
    skip_chunking: bool = True
) -> Tuple[bool, str, Optional[SearchIndex]]:
    """
    便捷函数：直接从文本列表创建知识库。
    
    Args:
        texts: 文本字符串列表。
        source_id: 来源标识。
        redis_url: Redis 连接地址。
        skip_chunking: 是否跳过分块处理。
        
    Returns:
        元组 (是否成功, 提示消息, 搜索索引对象)
    """
    redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
    kb_manager = KnowledgeBaseManager(redis_client)
    return kb_manager.create_knowledge_base(source_id, texts, skip_chunking=skip_chunking)

def _split_markdown_into_structured_chunks(markdown_text: str):
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    """按 Markdown 层级切块，并对超长段落进行递归兜底切块。"""
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True,
    )
    header_docs = header_splitter.split_text(markdown_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    structured_chunks = []
    for doc in header_docs:
        child_docs = recursive_splitter.split_documents([doc])
        for child in child_docs:
            content = child.page_content.strip()
            if not content:
                continue
            metadata = {
                "header_1": child.metadata.get("header_1", ""),
                "header_2": child.metadata.get("header_2", ""),
                "header_3": child.metadata.get("header_3", ""),
                "is_announcement": bool(
                    re.search(r"(?m)^\s*>", content)
                    or "最新系统公告" in content
                    or "黑五" in content and "补偿" in content
                ),
            }
            structured_chunks.append({"content": content, "metadata": metadata})
    return structured_chunks
