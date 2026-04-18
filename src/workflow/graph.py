import logging
from langgraph.graph import StateGraph, END  # StateGraph 用于定义有状态的图，END 是终结节点的标识

# 导入节点函数：工作流中的每一个步骤（Action）
from workflow.nodes import (
    WorkflowState,            # 定义了全局状态的结构（Schema）
    initialize_nodes,         # 初始化节点所需的依赖
    pre_check_node,           # 节点0：前置拦截器
    check_cache_node,         # 节点1：检查语义缓存
    research_node,            # 节点2：执行搜索/研究
    evaluate_quality_node,    # 节点3：评估研究结果质量
    synthesize_response_node, # 节点4：整合资料并生成回答
)
# 导入边缘路由函数：决定流程走向的逻辑（Decision）
from workflow.edges import cache_router, research_quality_router
# 导入工具初始化函数：主要用于初始化向量数据库检索工具
from workflow.tools import initialize_tools

def create_agent_graph(sys_cache=None, kb_index=None, embeddings=None) -> StateGraph:
    """
    初始化并构建 LangGraph 计算图。
    
    该图定义了智能体如何处理问题：
    1. 先查缓存 
    2. 没命中则研究 
    3. 研究完评估质量 
    4. 质量不行重搜，质量行了就出报告。
    """
    
    # --- 基础组件初始化 ---
    # 将语义缓存实例注入节点逻辑中
    initialize_nodes(sys_cache)
    # 如果提供了知识库索引和向量模型，则初始化相关的搜索工具
    if kb_index and embeddings:
        initialize_tools(kb_index, embeddings)

    # --- 构建状态机图 ---
    # 使用 WorkflowState 作为底层状态模式，所有节点共享并修改这个状态
    workflow = StateGraph(WorkflowState)

    # 1. 添加节点 (Nodes)
    workflow.add_node("pre_check", pre_check_node)               # 前置拦截器节点
    workflow.add_node("check_cache", check_cache_node)           # 缓存检查节点
    workflow.add_node("research", research_node)                 # 知识检索/研究节点
    workflow.add_node("evaluate_quality", evaluate_quality_node) # 质量打分节点
    workflow.add_node("synthesize_response", synthesize_response_node) # 最终响应合成节点

    # 2. 设置入口点 (Entry Point)
    workflow.set_entry_point("pre_check")

    # 2.5 配置前置检查到缓存的条件边缘
    workflow.add_conditional_edges(
        "pre_check",
        lambda state: "synthesize_response" if state.get("intercepted", False) else "check_cache",
        {
            "synthesize_response": "synthesize_response",
            "check_cache": "check_cache"
        }
    )

    # 3. 配置条件边缘 (Conditional Edges)
    workflow.add_conditional_edges(
        "check_cache",
        cache_router, # 路由逻辑：命中则去 synthesize_response，未命中则去 research
        {
            "synthesize_response": "synthesize_response",
            "research": "research"
        }
    )

    # 4. 配置普通边缘 (Normal Edges)
    # research 运行完后，无条件进入 evaluate_quality 进行打分
    workflow.add_edge("research", "evaluate_quality")

    # 5. 配置带反馈循环的条件边缘 (Feedback Loop)
    # evaluate_quality 运行完后，由 research_quality_router 决定
    workflow.add_conditional_edges(
        "evaluate_quality",
        research_quality_router, # 路由逻辑：质量达标去合成，不达标则退回 research 重新搜
        {
            "synthesize_response": "synthesize_response",
            "research": "research"
        }
    )

    # 6. 设置终点
    # synthesize_response 运行完后，流程结束
    workflow.add_edge("synthesize_response", END)

    # --- 日志记录与编译 ---
    logger = logging.getLogger("agentic-workflow")
    logger.info("LangGraph 计算图构建完成，逻辑包含：语义缓存 -> RAG循环 -> 动态路由")

    # 编译计算图，返回一个可执行的 app 对象
    return workflow.compile()