import logging  

def setup_logging():
    """
    初始化全局日志配置，统一输出格式。
    
    该函数执行以下操作：
    1. 配置全局日志的显示级别。
    2. 定义日志记录的字符串格式（时间、级别、内容）。
    3. 返回一个特定名称的日志对象供后续调用。
    
    返回：
        logging.Logger: 配置好的日志记录器实例。
    """
    # `basicConfig` 是最简单的日志配置入口
    logging.basicConfig(
        # 设置日志级别为 INFO。
        # 级别顺序：DEBUG < INFO < WARNING < ERROR < CRITICAL
        # 设置为 INFO 意味着 DEBUG 级别的日志将被忽略，显示 INFO 及以上级别的日志。
        level=logging.INFO,
        
        # 定义输出格式：
        # %(asctime)s   : 打印日志的时间
        # %(levelname)s : 打印日志级别名称（如 INFO, ERROR）
        # %(message)s   : 打印具体的日志消息内容
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # 获取（或创建）一个名为 "agentic-workflow" 的日志记录器。
    # 使用具名日志记录器而不是根记录器（root），是为了在大型应用中更好地标识日志来源。
    return logging.getLogger("agentic-workflow")

# --- 使用示例 ---
# 如果把本文件当作脚本直接运行：
# if __name__ == "__main__":
#     logger = setup_logging()
#     logger.info("日志系统已启动")  # 输出: 2023-10-27 10:00:00,000 | INFO | 日志系统已启动