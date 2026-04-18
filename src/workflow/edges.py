import logging  # 导入日志模块，用于追踪智能体的决策路径
from typing import Literal  # 用于类型提示，限制函数返回值的具体范围

# 获取名为 "agentic-workflow" 的日志记录器
logger = logging.getLogger("agentic-workflow")

def cache_router(state) -> Literal["synthesize_response", "research"]:
    """
    语义缓存路由决策器。
    
    逻辑：
    1. 检查 state 中标记的 cache_hit（缓存是否命中）。
    2. 如果命中了（即 FAQ 中有现成答案），直接跳转到响应合成阶段，省去耗时的搜索过程。
    3. 如果没命中，则指示智能体进入“研究/搜索”阶段。
    """
    query = state["query"]  # 获取用户的原始提问
    cache_hit = state.get("cache_hit", False)  # 从状态中提取缓存命中标志，默认为 False
    
    if cache_hit:
        # 命中缓存的情况：快速路径，直接生成最终回答
        # :20 是什么意思？为了日志输出的简洁性，我们只显示查询的前 20 个字符，后面用省略号表示。这有助于在日志中快速识别问题的主题，同时避免输出过长的文本导致日志混乱。
        logger.info(f"👉 路由: 缓存命中，跳过研究节点 -> '{query[:20]}...'")
        return "synthesize_response"
    else:
        # 未命中缓存：慢速路径，需要调用搜索工具进行研究
        logger.info(f"👉 路由: 未命中缓存，开始研究 -> '{query[:20]}...'")
        return "research"

def research_quality_router(state) -> Literal["synthesize_response", "research"]:
    """
    研究质量评估路由决策器（通常用于 RAG 循环逻辑）。
    
    逻辑：
    1. 评估当前已搜集到的资料质量得分。
    2. 如果质量达标（>= 0.7），进入响应合成。
    3. 如果质量不达标，但还没超过最大尝试次数，则退回研究节点重新搜索。
    4. 如果超过了最大尝试次数，为防止死循环，强制进入响应合成（尽可能基于现有资料回答）。
    """
    query = state["query"]                                   # 获取原始提问
    score = state.get("research_quality_score", 0.0)         # 获取当前的资料质量得分 (0.0-1.0)
    iterations = state.get("research_iterations", 1)         # 当前是第几次研究循环
    max_iterations = state.get("max_research_iterations", 1) # 允许的最大研究循环次数
    
    if score >= 0.7:
        # 情况 A：研究结果非常精准，可以直接开始写答案了
        logger.info(f"👉 路由: 质量达标，进入内容合成 ({score:.2f}) -> '{query[:20]}...'")
        return "synthesize_response"
    elif iterations >= max_iterations:
        # 情况 B：虽然资料质量一般，但已经尝试了多次，为保证响应速度，停止搜索
        logger.info(f"👉 路由: 达到最大研究次数 ({iterations})，不论质量是否达标，强行合成 -> '{query[:20]}...'")
        return "synthesize_response"
    else:
        # 情况 C：资料不够好（比如没搜到关键信息），且还有尝试机会，指示智能体重新调整策略去搜索
        logger.info(f"👉 路由: 研究质量不达标，进入新一轮研究 ({iterations}/{max_iterations}) -> '{query[:20]}...'")
        return "research"