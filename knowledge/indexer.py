import re  # 导入正则表达式库，用于文本模式匹配（如识别公告内容）

"""
知识库工具模块。
提供从文本快速构建 Redis 向量知识库的简化能力。
该模块集成了文本切分、向量化、Redis 索引管理等 RAG（检索增强生成）核心流程。
"""

import logging  # 日志记录
import hashlib  # 用于生成唯一哈希值，确保索引名称的一致性
import time     # 用于记录创建时间
from typing import Dict, Any, List, Optional, Tuple, Union  # 静态类型提示

import redis  # Redis 官方 Python 客户端
from langchain.schema import Document  # LangChain 的标准文档对象格式
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 递归字符文本切分器
from redisvl.utils.vectorize import HFTextVectorizer  # RedisVL 提供的 HuggingFace 向量化工具
from redisvl.index import SearchIndex  # RedisVL 的搜索索引管理类

# 初始化日志记录器，方便追踪处理流程和排查错误
logger = logging.getLogger("kb-utils")

class KnowledgeBaseManager:
    """
    管理基于 Redis 的知识库索引及其生命周期。
    负责从原始文本到 Redis 向量库的完整转换过程。
    """
    
    def __init__(self, redis_client: redis.Redis, embeddings: Optional[HFTextVectorizer] = None):
        """
        初始化知识库管理器。
        
        Args:
            redis_client: 已连接的 Redis 客户端实例。
            embeddings: 向量化工具实例。如果不提供，默认加载 BAAI 的 BGE 强力中文模型。
        """
        self.redis_client = redis_client
        
        # 默认使用 BAAI/bge-large-zh-v1.5，这是一个高性能的中文向量化模型，输出维度为 1024
        self.embeddings = embeddings or HFTextVectorizer(model="BAAI/bge-large-zh-v1.5")
        
        # 用于在内存中缓存当前正在使用的索引信息，键为 source_id 的哈希值
        self.active_indexes = {} 
    
    def create_knowledge_base(
        self, 
        source_id: str, 
        content
    ) -> Tuple[bool, str, Optional[SearchIndex]]:
        """
        核心方法：将输入内容（字符串或列表）转化为 Redis 中的向量搜索索引。
        
        Args:
            source_id: 唯一标识符，例如文档的 URL 或文件名。
            content: 待处理内容。支持字符串（需切分）或列表（已预处理好的块）。
            
        Returns:
            Tuple: (成功标志 bool, 结果描述消息 str, 索引对象 SearchIndex)
        """
        try:
            # 校验：如果没有内容则无法创建
            if not content:
                return False, "No content to process", None
            
            # --- 唯一名称生成 ---
            # 基于 source_id 生成 8 位 MD5 哈希作为索引名。
            # 这样做可以确保：1. Redis 键名符合规范；2. 同样的 source_id 始终对应同一个索引。
            source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
            index_name = f"kb-{source_hash}"
            
            # 此函数里面的分块逻辑可以考虑去除？
            # --- 步骤 1: 文本预处理与标准化 ---
            if isinstance(content, list):
                # 处理列表输入（通常是已经手动切分并带有元数据的数据）
                normalized_chunks = []
                for item in content:
                    if isinstance(item, str):
                        # 纯字符串列表，无元数据
                        normalized_chunks.append({"content": item, "metadata": {}})
                    elif isinstance(item, dict):
                        # 字典列表，需提取 content 字段，并保留 metadata
                        chunk_content = str(item.get("content", "") or "").strip()
                        if not chunk_content:
                            continue
                        raw_meta = item.get("metadata", {})
                        metadata = raw_meta if isinstance(raw_meta, dict) else {}
                        normalized_chunks.append({"content": chunk_content, "metadata": metadata})
                    else:
                        logger.warning(f"列表中不支持的块类型: {type(item)}")

                text_chunks = normalized_chunks
                # 判断是否有结构化元数据，用于后续标识 source_type
                has_structured_metadata = any(chunk.get("metadata") for chunk in text_chunks)
                source_type = "structured_list" if has_structured_metadata else "text_list"
                logger.info(f"使用提供的 {len(text_chunks)} 个文本块列表")
                
            else:
                return False, f"Unsupported content type: {type(content)}. Expected a list of chunks.", None
            
            # --- 步骤 2: 定义 Redis 向量索引 Schema（结构） ---
            # 该结构决定了数据如何在 Redis 中存储以及哪些字段可以被检索
            schema = {
                "index": {
                    "name": index_name,                # 索引名称
                    "prefix": f"kb:{source_hash}:"     # Redis 内部存储的键前缀
                },
                "fields": [
                    {"name": "content", "type": "text"},          # 全文搜索字段：存储原始文本
                    {"name": "source_id", "type": "tag"},         # 标签字段：用于精确过滤（如多租户隔离）
                    {"name": "source_type", "type": "tag"},       # 标签字段：标识来源类型
                    {"name": "header_1", "type": "text"},         # 文本标题层级，方便检索时显示上下文
                    {"name": "header_2", "type": "text"},
                    {"name": "header_3", "type": "text"},
                    {"name": "is_announcement", "type": "tag"},   # 标签字段：用于快速筛选公告类内容
                    {"name": "chunk_index", "type": "numeric"},   # 数字字段：存储切片顺序，方便还原上下文
                    {
                        "name": "content_vector",                 # 向量字段：存储文本的数学特征
                        "type": "vector",
                        "attrs": {
                            "dims": 1024,                         # 维度：必须与 bge-large-zh 模型一致
                            "distance_metric": "cosine",         # 度量：使用余弦相似度计算相关性
                            "algorithm": "hnsw",                 # 算法：HNSW 是一种高效的近似最近邻搜索算法
                            "datatype": "float32",                # 浮点数存储
                        },
                    },
                ],
            }
            
            # --- 步骤 3: 在 Redis 中创建并初始化索引 ---
            kb_index = SearchIndex.from_dict(schema, redis_client=self.redis_client)
            # overwrite=True：如果已存在同名索引，则先删除再创建，确保数据最新
            kb_index.create(overwrite=True)
            
            # --- 步骤 4: 并行向量化（Embedding）并封装 Payload ---
            payload = []
            for i, chunk in enumerate(text_chunks):
                try:
                    text_chunk = str(chunk.get("content", "") or "").strip()
                    if not text_chunk:
                        continue
                        
                    metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                    
                    # 调用 BGE 模型将文本转化为字节流（as_buffer=True 节省传输开销）
                    embedding = self.embeddings.embed(text_chunk, as_buffer=True)
                    
                    # 构建存入 Redis 的完整字典
                    payload.append({
                        "content": text_chunk,
                        "source_id": source_id,
                        "source_type": source_type,
                        "header_1": str(metadata.get("header_1", "") or ""),
                        "header_2": str(metadata.get("header_2", "") or ""),
                        "header_3": str(metadata.get("header_3", "") or ""),
                        # RedisVL 对 bool 处理较严格，此处转为字符串 tag
                        "is_announcement": "true" if bool(metadata.get("is_announcement", False)) else "false",
                        "chunk_index": i,
                        "content_vector": embedding,
                    })
                except Exception as e:
                    logger.warning(f"数据块 {i} 向量化失败: {e}")
                    continue
            
            if not payload:
                return False, "Failed to create embeddings for content", None
            
            # --- 步骤 5: 将数据批量装载（Upsert）到 Redis ---
            kb_index.load(payload)
            
            # 将索引信息存入内存管理器，记录元数据
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
            # 异常捕获，防止程序崩溃
            error_msg = f"❌ Knowledge base creation failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    

def create_knowledge_base_from_texts(
    texts: List[Union[str, Dict[str, Any]]],
    source_id: str,
    redis_url: str
) -> Tuple[bool, str, Optional[SearchIndex]]:
    """
    顶层便捷函数：允许调用者通过简单的文本列表快速初始化 Redis 知识库。
    
    Args:
        texts: 文本列表，可以是纯字符串，也可以是包含 content 和 metadata 的字典。
        source_id: 业务定义的唯一标识。
        redis_url: Redis 连接字符串。
    """
    # 创建非解码模式的 Redis 客户端（向量二进制数据需要原始字节流）
    redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
    kb_manager = KnowledgeBaseManager(redis_client)
    return kb_manager.create_knowledge_base(source_id, texts)

def _split_markdown_into_structured_chunks(markdown_text: str):
    """
    内部辅助函数：专门处理 Markdown 文档。
    采用两阶段切分法：
    1. 结构化切分：按 Markdown 标题 (#, ##, ###) 将文档切成逻辑段落。
    2. 兜底切分：如果某个标题下的段落依然过长（超过 500 字），则进行递归切分。
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    
    # 阶段 1：定义识别的标题层级
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True, # 切分后去掉原始内容中的标题符号，因为标题已存入元数据
    )
    header_docs = header_splitter.split_text(markdown_text)

    # 阶段 2：定义细粒度切分规则
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # 单块最大 500 字，适合语义检索的细粒度需求
        chunk_overlap=80,    # 保持 80 字重叠，防止切断关键语义
        separators=["\n\n", "\n", "。", "！", "？", " ", ""], # 切分优先级顺序
    )

    structured_chunks = []
    # 遍历标题切分后的文档块
    for doc in header_docs:
        # 对每一个大块进一步细切
        child_docs = recursive_splitter.split_documents([doc])
        for child in child_docs:
            content = child.page_content.strip()
            if not content:
                continue
            
            # --- 核心逻辑：元数据提取 ---
            metadata = {
                # 从 MarkdownHeaderTextSplitter 结果中继承标题信息
                "header_1": child.metadata.get("header_1", ""),
                "header_2": child.metadata.get("header_2", ""),
                "header_3": child.metadata.get("header_3", ""),
                # 特色功能：通过正则表达式或关键词识别该块是否为“公告”类重要内容
                "is_announcement": bool(
                    re.search(r"(?m)^\s*>", content) # Markdown 引用符号通常用于公告
                    or "最新系统公告" in content
                    or ("黑五" in content and "补偿" in content) # 针对特定业务场景的关键词识别
                ),
            }
            # 构建最终结构化数据
            structured_chunks.append({"content": content, "metadata": metadata})
            
    return structured_chunks