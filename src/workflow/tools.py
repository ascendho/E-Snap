"""
深度研究工作流的工具模块。
该模块负责知识库语义检索的具体实现：
1) 将自然语言查询转为向量嵌入（Embedding）；
2) 在 Redis 向量索引中执行向量近邻搜索（Vector Search）；
3) 对结果进行处理，返回带相关度分数的格式化文本。
"""

import logging  # 导入日志模块，用于记录工具的运行状态和调试信息
from typing import Any  # 导入 Any 类型，用于通用的类型标注

from langchain_core.tools import tool  # 导入 LangChain 的 tool 装饰器，将普通函数转换为 Agent 可调用的工具
from langchain_openai import ChatOpenAI  # 导入 OpenAI 模型接口（此处主要用于类型参考或扩展）
from redisvl.utils.vectorize import HFTextVectorizer  # 导入 RedisVL 的向量化工具类
from redisvl.query import VectorQuery  # 导入 RedisVL 的向量查询构建器

# 配置模块日志记录器，专门用于在控制台追踪智能体调用此工具的具体行为
logger = logging.getLogger("agentic-workflow")

# --- 全局状态管理 ---
# 这些变量在模块加载时为 None，必须在程序启动阶段通过 initialize_tools() 进行注入。
# 使用全局变量是为了让被 @tool 装饰的函数能够访问这些共享的连接资源。
kb_index = None  # 用于存储 RedisVL 的 SearchIndex 实例，负责执行实际的 Redis 命令
embeddings = None  # 用于存储向量化模型实例，负责将文本转换为数学向量


def initialize_tools(
    knowledge_base_index: Any, openai_embeddings: HFTextVectorizer
):
    """
    初始化工具模块所需的依赖项。
    
    原因说明：
    由于 LangChain 的 @tool 装饰器定义的工具通常是单例或全局的，
    但在生产环境下，Redis 连接和模型加载通常在主程序启动时完成，
    因此我们通过此函数将这些“重型资源”动态注入到本工具模块中。

    Args:
        knowledge_base_index: 已连接并配置好的 RedisVL SearchIndex 实例。
        openai_embeddings: 已加载权重的向量化模型实例（如 BGE 或 OpenAI Embedding）。
    """
    # 这里采用“启动期注入、运行期复用”的模式，而不是每次调用工具都重新构建依赖。
    # 原因是索引连接和向量模型都属于重对象，应该在进程级别保持单例语义。
    global kb_index, embeddings      # 申明使用全局变量
    kb_index = knowledge_base_index  # 注入索引实例
    embeddings = openai_embeddings   # 注入模型实例


@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    在 Redis 向量知识库中搜索与问题最相关的信息片段。
    
    这是供智能体（Agent）调用的标准工具。
    智能体在发现自己无法直接回答问题时，会自发决定调用此函数，
    将“原始查询”转化为“语义向量”，并在 Redis 毫秒级返回最匹配的知识。

    Args:
        query: 用户的问题或需要检索的关键词。
        top_k: 指定返回最相关的文本块数量，默认取前 3 条以平衡信息量与 Token 消耗。

    Returns:
        一段格式化的字符串，包含检索到的文本内容、层级标题及其相关度分数。
    """
    # 【安全性检查】：确保资源已注入。如果未初始化直接调用，返回提示而非让系统崩溃。
    if not kb_index or not embeddings:
        return "错误：知识库组件（索引或模型）尚未初始化，请联系系统管理员。"

    # 在日志中记录搜索请求。这里的 query 是 LLM 为工具调用生成/转发的检索词，不一定等于用户原句。
    logger.info(
        f"🔍 触发知识库检索工具 | 工具检索词: '{query}' | 返回条数: {top_k}"
    )

    try:
        # --- 步骤 1: 文本向量化 ---
        # 使用初始化时注入的模型，将字符串转换成一个高维浮点数数组（向量）
        query_vector = embeddings.embed(query)

        # --- 步骤 2: 构建混合检索语义 (Vector KNN + Text BM25) ---
        # 这里故意保留两条信号：
        # - 向量检索负责召回“意思接近”的内容
        # - BM25 负责抓商品名、规则原文、专有词等字面强匹配
        # 对客服知识库来说，这种混合检索通常比只做单一路径更稳。
        # 1. 向量相似度查询 (语义搜索)
        vec_query = VectorQuery(
            vector=query_vector,           # 搜索的“靶向”向量
            vector_field_name="content_vector", # Redis 中存储向量的字段名，需与 Indexer 配置一致
            return_fields=[                # 指定需要从 Redis 中检索并返回的字段
                "content",                 # 原始文本内容
                "vector_distance",         # 向量距离（用于计算相似度）
                "header_1",                # Markdown 一级标题
                "header_2",                # Markdown 二级标题
                "header_3",                # Markdown 三级标题
                "is_announcement",         # 业务标记：是否为公告
            ], 
            num_results=top_k,             # 限制返回的结果条数
        )

        # 执行查询：在 Redis 的 HNSW 索引空间中进行快速最近邻向量搜索
        vec_results = kb_index.query(vec_query)
        
        # 2. 文本词频匹配 (BM25) 查询 (字面搜索)
        txt_results = []
        try:
            from redisvl.query import FilterQuery
            from redisvl.query.filter import Text
            
            # 使用 Text("%") 算子进行全文词频检索（支持 RediSearch 分词截断搜索）
            text_filter = Text("content") % query
            txt_query = FilterQuery(
                return_fields=[
                    "content", 
                    "vector_distance", 
                    "header_1", 
                    "header_2", 
                    "header_3", 
                    "is_announcement"
                ],
                filter_expression=text_filter,
                num_results=top_k,
            )
            txt_results = kb_index.query(txt_query)
        except Exception as e:
            logger.warning(f"⚠️ BM25 文本检索调度异常: {e}，回退使用单一向量匹配...")
        
        # 3. 倒排多路召回合并 (RRF 混合搜索的简明去重合并)
        seen_content = set()
        merged_results = []
        # 这里采用“交叉拉链 + 去重”的轻量合并，而不是更重的二次排序：
        # 目标是给后续 LLM 一组覆盖面稳定的上下文，而不是在工具层追求复杂排序最优。
        for i in range(max(len(vec_results), len(txt_results))):
            # 先装载语义命中的核心内容
            if i < len(vec_results):
                cnt = vec_results[i].get("content")
                if cnt not in seen_content:
                    merged_results.append(vec_results[i])
                    seen_content.add(cnt)
            # 再装载纯字面精确匹配的内容 (如专有名词)
            if i < len(txt_results):
                cnt = txt_results[i].get("content")
                if cnt not in seen_content:
                    # 对于纯文本命中的项，为了规避距离没有返回，我们可以赋一个初始中等置信度的 dummy 虚拟距离
                    if "vector_distance" not in txt_results[i]:
                        txt_results[i]["vector_distance"] = 0.25 
                    merged_results.append(txt_results[i])
                    seen_content.add(cnt)
        
        # 截断符合目标数量
        results = merged_results[:top_k]

        # --- 步骤 3: 结果后处理 ---
        # 如果搜索不到任何结果，返回友好提示给 Agent
        if not results:
            return f"注意：知识库中暂无与 '{query}' 相关的记录。"

        # 将 Redis 返回的原始数据（字典列表）转换为易于 LLM 阅读的结构化文本。
        # 这里输出的是“供 research prompt 消费的上下文材料”，不是直接展示给用户的最终答案。
        formatted_results = []
        for i, result in enumerate(results, 1):
            # 距离转相关度：距离越小表示越相似。公式：1.0 - 距离
            relevance = 1.0 - float(result.get("vector_distance", 0.0))
            
            # 处理标题路径：将 # ## ### 标题合并为 标题1 > 标题2 这种路径格式
            header_parts = [
                str(result.get("header_1", "") or "").strip(),
                str(result.get("header_2", "") or "").strip(),
                str(result.get("header_3", "") or "").strip(),
            ]
            header_path = " > ".join([h for h in header_parts if h])
            
            # 识别公告标记
            is_announcement = str(result.get("is_announcement", "false")).lower() == "true"

            # 构建当前块的元数据摘要
            prefix_lines = []
            if header_path:
                prefix_lines.append(f"文档章节: {header_path}")
            if is_announcement:
                prefix_lines.append("特殊标记: 【最新公告/通知】")
            prefix_block = "\n".join(prefix_lines)

            # 构建最终展示给 LLM 的单条结果块
            body = f"搜索结果 {i} (相关度: {relevance:.3f}):\n{result['content']}"
            
            # 如果有标题信息则拼接到正文前，增强 LLM 对上下文的理解
            formatted_results.append(
                f"{prefix_block}\n{body}" if prefix_block else body
            )

        # 将所有匹配到的块用双换行符连接成一个大的上下文字符串
        return "\n\n" + "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        # 异常捕获：如 Redis 断连或模型计算错误
        logger.error(f"❌ 检索执行失败: {e}")
        return f"知识库检索系统发生内部错误，请稍后再试。错误详情：{str(e)}"