"""结果路径分类器：测试报表与日志共用同一套语义。"""

from typing import Dict, List


def execution_path(result: Dict) -> List[str]:
    return result.get("execution_path", []) or []


def is_exact_bypass(result: Dict) -> bool:
    return result.get("cache_hit", False) and result.get("cache_match_type") == "exact"


def is_near_exact_bypass(result: Dict) -> bool:
    return result.get("cache_hit", False) and result.get("cache_match_type") == "near_exact"


def is_edit_distance_bypass(result: Dict) -> bool:
    return result.get("cache_hit", False) and result.get("cache_match_type") == "edit_distance"


def is_rerank_candidate(result: Dict) -> bool:
    return (
        bool(result.get("cache_matched_question"))
        and result.get("cache_match_type") not in {"exact", "near_exact", "none"}
        and result.get("cache_rerank_attempt") not in {"skipped", "none"}
    )


def is_reranked_full_reuse(result: Dict) -> bool:
    return is_rerank_candidate(result) and result.get("cache_reuse_mode") == "full_reuse"


def is_dual_subquery_reuse(result: Dict) -> bool:
    return result.get("cache_reuse_mode") == "dual_subquery"


def is_partial_reuse(result: Dict) -> bool:
    if result.get("cache_reuse_mode") == "partial_reuse":
        return True
    if is_dual_subquery_reuse(result):
        return False
    return "supplement_researched" in execution_path(result)


def is_reranker_exception(result: Dict) -> bool:
    return str(result.get("cache_rerank_reason", "")).startswith("rerank_exception:")


def classify_path(result: Dict) -> str:
    """对一条工作流结果做单一来源的路径分类。"""
    if result.get("intercepted", False):
        return "拦截"
    if is_exact_bypass(result):
        return "精确缓存直出"
    if is_near_exact_bypass(result):
        return "近精确缓存直出"
    if is_edit_distance_bypass(result):
        return "编辑距离缓存直出"
    if is_dual_subquery_reuse(result):
        return "Dual Subquery Cache Hit"
    if is_reranked_full_reuse(result):
        return "Reranker完整复用"
    if is_partial_reuse(result):
        return "部分复用+补充研究"
    if is_rerank_candidate(result):
        return "Reranker拒绝后研究"
    return "完整研究"
