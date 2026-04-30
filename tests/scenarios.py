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

def _normalize_query_item(item):
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        query = item.get("query", "")
        if isinstance(query, str):
            return query.strip()
    return ""


def _iter_queries(data):
    if isinstance(data, list):
        for item in data:
            query = _normalize_query_item(item)
            if query:
                yield query
        return

    if isinstance(data, dict):
        for queries in data.values():
            if not isinstance(queries, list):
                continue
            for item in queries:
                query = _normalize_query_item(item)
                if query:
                    yield query


for index, query in enumerate(_iter_queries(_test_data), 1):
    SCENARIO_RUNS.append({"title": f"测试{index:02d}", "query": query})
