from cache.faq_data_container import FAQDataContainer
from cache.engine import SemanticCacheWrapper

def setup_cache():
    """
    初始化语义缓存系统，并使用预定义的 FAQ 数据进行索引预热。

    这是“本项目是否有预加载缓存数据”的直接答案入口：
    - main.py 启动离线流程时会调这里
    - api/server.py 启动 Web 服务时也会调这里

    这里做的不是懒加载，而是启动期预热：
    先把 FAQ 种子问答读出来，再一次性写进缓存，
    这样高频标准问题在第一位用户到来前就已经具备 exact / near_exact / semantic 命中能力。
    """
    # 直接实例化，配置全部从 env 获取
    cache = SemanticCacheWrapper()
    data = FAQDataContainer()

    # FAQDataContainer 只负责把 CSV 读成 DataFrame；
    # 这里再转成 cache.store_batch() 能直接消费的问答字典列表。
    qa_pairs = data.faq_df.to_dict(orient="records")
    
    # clear=True 表示每次启动都先清空旧缓存，再用最新 FAQ 种子做一次完整预热。
    # 这让 FAQ 标准答案始终成为缓存里的基线事实，避免旧数据残留。
    cache.store_batch(qa_pairs, clear=True)
    
    return cache
