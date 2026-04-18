import pandas as pd  # 导入 pandas 库，用于处理结构化的表格数据 (DataFrame)

class FAQDataContainer:
    """
    FAQ 数据容器类。
    
    主要职责：
    负责从本地磁盘加载预定义的常见问题（FAQ）种子数据。
    这些数据通常是业务专家整理的标准问答，用于在系统启动时对“语义缓存”进行预热，
    从而确保高频问题能够被快速拦截并直接返回标准答案。
    """
    
    def __init__(self):
        """
        初始化数据容器。
        在对象创建时，自动执行数据加载逻辑。
        """
        
        # 加载位于 data/ 目录下的 faq_seed.csv 文件。
        # 预期该 CSV 文件包含核心字段：
        # - question: 标准提问文本
        # - answer: 对应的官方答复
        # - (可选) id: 唯一标识符
        import os
        faq_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "faq_seed.csv")
        self.faq_df = pd.read_csv(faq_path)
        
        # 在控制台输出加载成功的反馈信息，便于开发者确认数据规模和路径是否正确
        print(f"✅ FAQ 种子数据加载成功：共计 {len(self.faq_df)} 条问答对")