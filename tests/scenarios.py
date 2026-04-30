import json
import os

"""测试场景模块。"""

# 加载独立的测试数据文件
_profile = os.getenv("TEST_SCENARIO_PROFILE", "debug").strip().lower()
_data_filename = "test_scenarios_full.json" if _profile == "full" else "test_scenarios.json"
_data_path = os.path.join(os.path.dirname(__file__), "..", "data", _data_filename)
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
_load_scenario("SCENARIO_6_PARTIAL_REUSE_SUCCESS_QUERIES", "场景6(部分复用成功)")
_load_scenario("SCENARIO_7_PARTIAL_REUSE_REJECT_QUERIES", "场景7(部分复用拒绝)")
