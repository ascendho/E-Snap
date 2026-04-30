import json
import os

"""测试场景模块。"""

# 加载独立的测试数据文件
_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_scenarios.json")
try:
    with open(_data_path, "r", encoding="utf-8") as f:
        _test_data = json.load(f)
except FileNotFoundError:
    print(f"⚠️ 测试数据文件缺失: {_data_path}")
    _test_data = {}

SCENARIO_RUNS = []

# 动态构建所有的测试场景
def _load_scenario(key_name, title_prefix):
    queries = _test_data.get(key_name, [])
    for i, q in enumerate(queries, 1):
        SCENARIO_RUNS.append({"title": f"{title_prefix}-问题{i}", "query": q})

_load_scenario("SCENARIO_1_QUERIES", "场景1")
_load_scenario("SCENARIO_2_QUERIES", "场景2")
_load_scenario("SCENARIO_3_QUERIES", "场景3")
_load_scenario("SCENARIO_4_FUZZY_QUERIES", "场景4(模糊测试)")
_load_scenario("SCENARIO_5_INTERCEPT_QUERIES", "场景5(拦截测试)")
_load_scenario("SCENARIO_6_PARTIAL_REUSE_QUERIES", "场景6(部分复用测试)")
