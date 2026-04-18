# 运行指令：python main.py
# 本脚本是系统的总入口，负责组装所有模块并启动演示场景。

import warnings  # 导入告警管理模块
from common.env import set_ark_key, to_bool_env  # 导入环境变量设置与类型转换工具
from workflow.graph import create_agent_graph  # 导入工作流图构建函数
from knowledge.builder import init_app_knowledge_base  # 导入知识库构建工具
from testing.runner import run_workflow_scenarios  # 导入测试运行器
from cache.auto_heater import setup_cache  # 导入语义缓存预热逻辑
from common.logger import setup_logging  # 导入日志配置工具

def main():
    """
    程序主入口：
    按照 引导 -> 初始化 -> 构建 -> 执行 的顺序完成 AI 智能体系统的启动。
    """
    # 忽略不影响程序运行的库告警（如 LangChain 或 RedisVL 的弃用提示），保持终端整洁
    warnings.simplefilter("ignore")
    
    # 从配置文件或系统变量中读取并设置火山引擎（Ark）的 API 密钥
    set_ark_key()
    
    # 初始化全局日志系统，后续所有模块的 logger 都会按照此配置输出
    logger = setup_logging()

    # ---------------------------------------------------------
    # 步骤 1) 初始化 RAG 知识库
    # ---------------------------------------------------------
    # 将原始 Markdown 文本进行切块、向量化并写入 Redis 向量索引。
    logger.info("初始化知识库 ......")

    # kb_index: Redis 向量索引对象（SearchIndex 实例）。
    #           它像是一个“带索引的图书馆”，负责执行底层的向量相似度检索。
    # embeddings: 向量化模型对象（HFTextVectorizer 实例）。
    #           它是“翻译官”，负责将用户的中文提问转换成计算机能理解的 1024 维向量。
    kb_index, embeddings = init_app_knowledge_base()

    # ---------------------------------------------------------
    # 步骤 2) 初始化语义缓存 (Semantic Cache)
    # ---------------------------------------------------------
    # 语义缓存用于拦截意思相近的重复提问，直接返回答案，无需经过大模型推理。
    logger.info("初始化语义缓存 ......")
    
    # cache 实例包含了 Redis 连接和预加载的 FAQ 种子数据。
    cache = setup_cache()

    # ---------------------------------------------------------
    # 步骤 3) 构建 LangGraph 智能体计算图
    # ---------------------------------------------------------
    # 将业务节点（缓存检查、研究、评估、合成）和路由逻辑编排成一个状态机。
    logger.info("构建 LangGraph 工作流计算图...")
    
    # 注入前面初始化的缓存和知识库组件，生成可执行的 Workflow 对象
    workflow_app = create_agent_graph(cache, kb_index, embeddings)

    # ---------------------------------------------------------
    # 步骤 4) 运行场景化测试
    # ---------------------------------------------------------
    # 从环境变量读取配置：是否在控制台打印详细的 JSON 运行结果（默认不打印）
    show_console_results = to_bool_env("SHOW_CONSOLE_RESULTS", False)
    
    # 执行预设的测试场景：
    run_workflow_scenarios(
        workflow_app=workflow_app,
        logger=logger,
        show_console_results=show_console_results,
    )

if __name__ == "__main__":
    # 只有当本脚本作为主程序运行时，才调用 main 函数
    main()