from semantic_cache.faq_data_container import FAQDataContainer
from semantic_cache.engine import SemanticCacheWrapper

def setup_semantic_cache():
    """
    初始化语义缓存系统，并使用预定义的 FAQ 数据进行索引预热。
    """
    # 直接实例化，配置全部从 env 获取
    cache = SemanticCacheWrapper()
    data = FAQDataContainer()

    # 业务数据转化逻辑保留在业务测处理
    qa_pairs = data.faq_df.to_dict(orient="records")
    
    # 彻底清空并进行预热
    cache.store_batch(qa_pairs, clear=True)
    
    return cache
