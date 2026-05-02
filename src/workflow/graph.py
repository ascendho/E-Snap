import logging
from langgraph.graph import StateGraph, END  # StateGraph 用于定义有状态的图，END 是终结节点的标识

# 导入节点函数：工作流中的每一个步骤（Action）
from workflow.nodes import (
    WorkflowState,            # 定义了全局状态的结构（Schema）
    initialize_nodes,         # 初始化节点所需的依赖
    pre_check_node,           # 节点0：前置拦截器
    check_cache_node,         # 节点1：检查语义缓存
    rerank_cache_node,        # 节点1.5：LLM Reranker，判定缓存答案能否复用
    research_node,            # 节点2：执行搜索/研究
    research_supplement_node, # 节点2.5：仅补充缓存未覆盖的部分
    synthesize_response_node, # 节点3：整合资料并生成回答
)
# 导入边缘路由函数：决定流程走向的逻辑（Decision）
from workflow.edges import (
    RouteTarget,
    cache_rerank_router,
    cache_router,
    pre_check_router,
)
# 导入工具初始化函数：主要用于初始化向量数据库检索工具
from workflow.tools import initialize_tools

def create_agent_graph(sys_cache=None, kb_index=None, embeddings=None) -> StateGraph:
    """
    初始化并构建 LangGraph 计算图。
    
    该图定义了智能体如何处理问题：
    1. 先做零层拦截（时间/库存/型号类问题直接兜底）
    2. 再查缓存，优先走 L1 fast path
    3. 命中候选但不能直接复用时，进入 Reranker 做语义裁判
    4. 未命中或裁判拒绝则进入 research / supplement research
    5. 最后统一进入 synthesize_response 收口输出与缓存写回

    这个函数是“全项目工作流拓扑”的单一真源：
    - `nodes.py` 负责“每个节点内部做什么”
    - `edges.py` 负责“节点做完后下一跳去哪里”
    - 本函数负责把二者组装成可执行状态机
    """
    
    # --- 基础组件初始化 ---
    # 将语义缓存实例注入节点逻辑中。
    # 这样 nodes.py 内部就只关心“读写缓存”，不需要知道 cache 实例来自哪里。
    initialize_nodes(sys_cache)
    # 检索工具也在这里统一注入，避免 tool 模块在 import 时抢先初始化重型依赖。
    if kb_index and embeddings:
        initialize_tools(kb_index, embeddings)

    # --- 构建状态机图 ---
    # 使用 WorkflowState 作为底层状态模式，所有节点共享并修改这个状态
    workflow = StateGraph(WorkflowState)

    # 1. 添加节点 (Nodes)
    workflow.add_node(RouteTarget.PRE_CHECK, pre_check_node)
    workflow.add_node(RouteTarget.CHECK_CACHE, check_cache_node)
    workflow.add_node(RouteTarget.RERANK_CACHE, rerank_cache_node)
    workflow.add_node(RouteTarget.RESEARCH, research_node)
    workflow.add_node(RouteTarget.RESEARCH_SUPPLEMENT, research_supplement_node)
    workflow.add_node(RouteTarget.SYNTHESIZE_RESPONSE, synthesize_response_node)

    # 2. 设置入口点 (Entry Point)
    workflow.set_entry_point(RouteTarget.PRE_CHECK)

    # 2.5 配置前置检查到缓存的条件边缘。
    # pre_check 只负责写出 `intercepted=True/False`，真正决定下一跳的是 router。
    workflow.add_conditional_edges(
        RouteTarget.PRE_CHECK,
        pre_check_router,
        {
            RouteTarget.SYNTHESIZE_RESPONSE: RouteTarget.SYNTHESIZE_RESPONSE,
            RouteTarget.CHECK_CACHE: RouteTarget.CHECK_CACHE,
        },
    )

    # check_cache 之后：
    # - exact / near_exact / edit_distance 直接进入 synthesize_response
    # - deterministic subquery partial 直接进入 research_supplement
    # - semantic / subquery candidate 进入 rerank_cache
    # - 完全未命中则进入 research
    workflow.add_conditional_edges(
        RouteTarget.CHECK_CACHE,
        cache_router,
        {
            RouteTarget.SYNTHESIZE_RESPONSE: RouteTarget.SYNTHESIZE_RESPONSE,
            RouteTarget.RESEARCH_SUPPLEMENT: RouteTarget.RESEARCH_SUPPLEMENT,
            RouteTarget.RERANK_CACHE: RouteTarget.RERANK_CACHE,
            RouteTarget.RESEARCH: RouteTarget.RESEARCH,
        },
    )

    # rerank_cache 之后：
    # - full_reuse 说明缓存答案已足够，直接收口
    # - partial_reuse 只研究缺口，不重复做整题 RAG
    # - reject 则完全放弃缓存答案，重新研究
    workflow.add_conditional_edges(
        RouteTarget.RERANK_CACHE,
        cache_rerank_router,
        {
            RouteTarget.SYNTHESIZE_RESPONSE: RouteTarget.SYNTHESIZE_RESPONSE,
            RouteTarget.RESEARCH_SUPPLEMENT: RouteTarget.RESEARCH_SUPPLEMENT,
            RouteTarget.RESEARCH: RouteTarget.RESEARCH,
        },
    )

    # 4. 配置普通边缘 (Normal Edges)
    # research 运行完后，直接进入 synthesize_response 生成最终回答
    workflow.add_edge(RouteTarget.RESEARCH, RouteTarget.SYNTHESIZE_RESPONSE)
    workflow.add_edge(RouteTarget.RESEARCH_SUPPLEMENT, RouteTarget.SYNTHESIZE_RESPONSE)

    # 5. 设置终点
    # synthesize_response 运行完后，流程结束
    workflow.add_edge(RouteTarget.SYNTHESIZE_RESPONSE, END)

    # --- 日志记录与编译 ---
    logger = logging.getLogger("agentic-workflow")
    logger.info("LangGraph 计算图构建完成，逻辑包含：快速缓存/子问题候选 -> 三态Reranker -> 补充研究/单轮RAG -> 动态路由")

    # 编译计算图，返回一个可执行的 app 对象
    return workflow.compile()