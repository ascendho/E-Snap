import asyncio
import json
import time
import warnings
import os
import sys

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from common.env import set_ark_key
from workflow.graph import create_agent_graph
from workflow.edges import RouteTarget, cache_router, cache_rerank_router
from workflow.nodes import (
    build_initial_state,
    check_cache_node,
    get_research_llm,
    pre_check_node,
    prepare_research_messages,
    research_supplement_node,
    rerank_cache_node,
    synthesize_response_node,
    wait_for_background_tasks,
)
from workflow.state import _record_llm_usage, initialize_metrics, update_metrics
from knowledge.builder import init_app_knowledge_base
from cache.auto_heater import setup_cache
from common.logger import setup_logging
from redis import Redis

warnings.simplefilter("ignore")

app = FastAPI(title="E-Snap API", version="1.0.0")

# Setup CORS to allow your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to point to the Netlify/GitHub branch URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for security
ACCESS_CODE = os.environ.get("ACCESS_CODE", "HIRE_ME_2026")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

logger = setup_logging()
workflow_app = None
redis_client = None
system_status = {
    "ready": False,
    "stage": "not_started",
    "message": "Backend is starting up.",
    "last_error": None,
    "cache_entry_count": 0,
    "cache_has_contact_prompt": False,
}


def update_system_status(stage: str, message: str, *, ready: bool = False, last_error: str | None = None):
    system_status["ready"] = ready
    system_status["stage"] = stage
    system_status["message"] = message
    system_status["last_error"] = last_error

def init_system():
    global workflow_app, redis_client
    update_system_status("loading_env", "Loading environment configuration...")
    logger.info(f"Python executable: {sys.executable}")
    set_ark_key()
    logger.info("Initializing Agent System for API...")

    update_system_status("building_knowledge_base", "Building knowledge base...")
    kb_index, embeddings = init_app_knowledge_base()

    update_system_status("warming_cache", "Warming FAQ cache...")
    cache = setup_cache()
    cache_entry_count = len(getattr(cache, "_answer_by_question", {}) or {})
    cache_has_contact_prompt = bool(getattr(cache, "contains_prompt_variant", lambda prompt: False)("怎么联系人工？"))
    system_status["cache_entry_count"] = cache_entry_count
    system_status["cache_has_contact_prompt"] = cache_has_contact_prompt
    logger.info(
        "FAQ cache warmed: entries=%s, contains '怎么联系人工？'=%s",
        cache_entry_count,
        cache_has_contact_prompt,
    )

    update_system_status("creating_workflow", "Creating agent workflow...")
    workflow_app = create_agent_graph(cache, kb_index, embeddings)

    update_system_status("connecting_redis", "Connecting API Redis client...")
    redis_client = Redis.from_url(REDIS_URL, decode_responses=True)

    update_system_status("ready", "Backend ready. You can start chatting now.", ready=True)
    logger.info("Agent System Ready!")

@app.on_event("startup")
async def startup_event():
    try:
        init_system()
    except Exception as exc:
        update_system_status(
            "error",
            "Backend startup failed. Check the terminal traceback before retrying.",
            last_error=str(exc),
        )
        logger.exception("Application startup failed")
        raise

# Simple dependency injection to verify access code
async def verify_access_code(authorization: str = None):
    # Depending on how the frontend sends it, we can look at the header `X-Access-Code` or `Authorization`
    pass

class ChatRequest(BaseModel):
    query: str
    access_code: str

class ChatResponse(BaseModel):
    answer: str
    latency_ms: float
    cache_hit: bool
    intercepted: bool
    cache_match_type: str
    cache_reuse_mode: str
    label_key: str
    label_text: str
    cache_written_prompts: list[str]

class ValidateRequest(BaseModel):
    access_code: str


@app.get("/health")
async def health_check():
    return {
        "ready": system_status["ready"],
        "stage": system_status["stage"],
        "message": system_status["message"],
        "error": system_status["last_error"],
        "cache_entry_count": system_status["cache_entry_count"],
        "cache_has_contact_prompt": system_status["cache_has_contact_prompt"],
        "has_workflow": workflow_app is not None,
        "has_redis_client": redis_client is not None,
        "python_executable": sys.executable,
        "process_id": os.getpid(),
    }

# Simple IP rate restrictor helper
def check_rate_limit(ip: str):
    if not redis_client:
        return
    # Allow 10 requests per minute per IP
    key = f"rate_limit:{ip}"
    current = redis_client.get(key)
    if current and int(current) > 10:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    redis_client.incr(key)
    redis_client.expire(key, 60)

def get_client_ip(request: Request) -> str:
    return request.client.host if request.client and request.client.host else "127.0.0.1"

def validate_chat_request(payload: ChatRequest, client_ip: str):
    if payload.access_code != ACCESS_CODE:
        logger.warning(f"Failed authorization attempt from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Access Code. Please use the code provided in the resume."
        )

    if not system_status["ready"] or workflow_app is None:
        if system_status["stage"] == "error":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Backend startup failed. Check the terminal traceback and retry after fixing it."
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend is still initializing. Wait until the terminal shows 'Application startup complete.' and retry."
        )

    check_rate_limit(client_ip)

# Single-source-of-truth label rules. Both /chat (sync) and /chat/stream (final event)
# resolve their UI badge through this table; the frontend fetches it once via /labels so
# button text never drifts between server and client.
# Order matters: the first matching rule wins.
# 因此规则顺序本身就是业务语义的一部分：
# 更具体、更强约束的标签需要排在更宽泛的标签前面，避免被后者“抢先命中”。
LABEL_RULES: list[tuple[str, str, callable]] = [
    ("zero_intercept",      "Zero-Layer Intercept",       lambda s: s.get("intercepted", False)),
    ("cache_exact",         "Exact Cache Hit",            lambda s: s.get("cache_hit", False) and s.get("cache_match_type") == "exact"),
    ("cache_near_exact",    "Near-Exact Cache Hit",       lambda s: s.get("cache_hit", False) and s.get("cache_match_type") == "near_exact"),
    ("cache_edit_distance", "Edit-Distance Cache Hit",    lambda s: s.get("cache_hit", False) and s.get("cache_match_type") == "edit_distance"),
    ("cache_semantic_reuse","Reranked Cache Reuse",       lambda s: s.get("cache_reuse_mode") == "full_reuse"),
    ("cache_dual_subquery", "Dual Subquery Cache Hit",    lambda s: s.get("cache_reuse_mode") == "dual_subquery"),
    ("cache_partial_reuse", "Partial Cache Reuse + RAG",  lambda s: s.get("cache_reuse_mode") == "partial_reuse"),
]
DEFAULT_LABEL = ("rag_full_research", "LLM Analysis / Full RAG")


def resolve_label(final_state: dict) -> tuple[str, str]:
    for key, text, predicate in LABEL_RULES:
        if predicate(final_state):
            return key, text
    return DEFAULT_LABEL


def build_label_metadata(final_state: dict) -> dict:
    # 后端只输出“这次回答属于哪种执行路径”的语义元数据，
    # 不负责图标、颜色、CSS class 这类前端呈现细节。
    cache_hit = final_state.get("cache_hit", False)
    cache_match_type = final_state.get("cache_match_type", "none")
    cache_reuse_mode = final_state.get("cache_reuse_mode", "none")
    label_key, label_text = resolve_label(final_state)
    return {
        "cache_hit": cache_hit,
        "intercepted": final_state.get("intercepted", False),
        "cache_match_type": cache_match_type,
        "cache_reuse_mode": cache_reuse_mode,
        "label_key": label_key,
        "label_text": label_text,
    }

def build_chat_response(final_state: dict, latency_ms: float) -> ChatResponse:
    answer = final_state.get("final_response", "系统遇到了未知错误，请稍后重试。")
    metadata = build_label_metadata(final_state)
    metadata["cache_written_prompts"] = final_state.get("cache_written_prompts", []) or []
    return ChatResponse(answer=answer, latency_ms=latency_ms, **metadata)

def build_stream_final_event(final_state: dict, latency_ms: float, answer: str) -> dict:
    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "cache_written_prompts": final_state.get("cache_written_prompts", []) or [],
        **build_label_metadata(final_state),
    }


def _compute_elapsed_latency(start_time: float) -> float:
    return round((time.time() - start_time) * 1000, 2)


def _build_stream_ready_final_event(final_state: dict, start_time: float, answer: str) -> dict:
    """Build the stream completion payload after synchronizing background metrics.

    `/chat/stream` now matches `/chat` and the offline runner: the final event is emitted
    only after background cache writeback threads have completed, so `total_latency`
    carries one consistent definition across all entry points.
    """
    return build_stream_final_event(final_state, _finalize_total_latency(final_state, start_time), answer)


def _chunk_has_usage_metadata(chunk) -> bool:
    usage_metadata = getattr(chunk, "usage_metadata", None) or {}
    if usage_metadata:
        return True
    response_metadata = getattr(chunk, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}
    return bool(token_usage)


async def _request_disconnected(request: Request | None) -> bool:
    if request is None:
        return False
    checker = getattr(request, "is_disconnected", None)
    if not callable(checker):
        return False
    try:
        return bool(await checker())
    except Exception:
        return True


def _finalize_total_latency(state: dict, start_time: float) -> float:
    """等待后台任务完成、计算响应总耗时并同步到 state['metrics']['total_latency']。

    该 helper 主要服务于 `/chat`、`/chat/stream` 与离线 runner：
    后台子问题缓存写回的 token / 成本会被纳入本次响应统计。

    这样三个入口对 `total_latency` 的含义保持一致：
    都表示“回答完成且后台写回也同步完成”的总墙钟时间。
    """
    if isinstance(state, dict):
        wait_for_background_tasks(state)
    latency = _compute_elapsed_latency(start_time)
    if isinstance(state, dict):
        metrics = state.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            state["metrics"] = metrics
        metrics["total_latency"] = latency
    return latency

# Stream event type discriminators. Keep these in sync with frontend/app.js.
STREAM_EVENT_STATUS = "status"
STREAM_EVENT_TOKEN = "token"
STREAM_EVENT_FINAL = "final"
STREAM_EVENT_ERROR = "error"


def stream_event(event_type: str, **payload) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"

def iter_text_chunks(text: str, chunk_size: int = 24):
    for start in range(0, len(text), chunk_size):
        yield text[start:start + chunk_size]

def extract_chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")

@app.post("/validate")
async def validate_code(request: ValidateRequest):
    if request.access_code != ACCESS_CODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Access Code"
        )
    return {"status": "ok", "message": "Access code is valid"}


@app.get("/labels")
async def list_labels():
    """Expose the label key→text mapping so the frontend never duplicates the rule text.

    The frontend fetches this once on load and uses it to populate badge captions; CSS
    classes remain client-side because they are presentation concerns.
    """
    labels = {key: text for key, text, _ in LABEL_RULES}
    labels[DEFAULT_LABEL[0]] = DEFAULT_LABEL[1]
    return {"labels": labels, "default": DEFAULT_LABEL[0]}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request):
    client_ip = get_client_ip(request)
    validate_chat_request(payload, client_ip)

    logger.info(f"Processing API Query: {payload.query}")
    start_time = time.time()
    
    try:
        # `/chat` 只是 workflow 的一个同步封装层：
        # 校验请求 -> 构造 state -> 调工作流 -> 封装响应。
        # 业务逻辑故意不写在这里，避免 API 层与 workflow 层重复维护同一套规则。
        initial_state = build_initial_state(payload.query)
        
        # Invoke the workflow graph
        final_state = workflow_app.invoke(initial_state)
        
        latency = _finalize_total_latency(final_state, start_time)
        response = build_chat_response(final_state, latency)
        
        logger.info(f"Answer generated in {latency}ms (Cache Hit: {response.cache_hit}, Intercepted: {response.intercepted})")
        return response

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")

@app.post("/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest, request: Request):
    client_ip = get_client_ip(request)
    validate_chat_request(payload, client_ip)

    async def event_generator():
        # `/chat/stream` 没有直接走 `workflow_app.invoke()`，
        # 而是手动复用同一批 node / router，以便在关键阶段向前端发送 status / token / final 事件。
        # 因而它的关键维护原则是：节点逻辑必须和 LangGraph 主路径保持一致，
        # 否则就会出现“同步接口正确、流式接口漂移”的问题。
        async def client_disconnected() -> bool:
            disconnected = await _request_disconnected(request)
            if disconnected:
                logger.info("Streaming client disconnected; aborting remaining work.")
            return disconnected

        start_time = time.time()
        try:
            logger.info(f"Processing streaming API Query: {payload.query}")
            if await client_disconnected():
                return
            state = build_initial_state(payload.query)

            yield stream_event(STREAM_EVENT_STATUS, stage=RouteTarget.PRE_CHECK, message="正在进行前置校验...")
            await asyncio.sleep(0)
            if await client_disconnected():
                return
            state = pre_check_node(state)

            if state.get("intercepted", False):
                if await client_disconnected():
                    return
                state = synthesize_response_node(state)
                answer = state.get("final_response", "")
                for chunk in iter_text_chunks(answer):
                    if await client_disconnected():
                        return
                    yield stream_event(STREAM_EVENT_TOKEN, content=chunk)
                    await asyncio.sleep(0)
                if await client_disconnected():
                    return
                yield stream_event(STREAM_EVENT_FINAL, **_build_stream_ready_final_event(state, start_time, answer))
                return

            yield stream_event(STREAM_EVENT_STATUS, stage=RouteTarget.CHECK_CACHE, message="正在检查缓存...")
            await asyncio.sleep(0)
            if await client_disconnected():
                return
            state = check_cache_node(state)
            # 与 graph.py 中的 `check_cache -> cache_router` 保持同构，
            # 这样流式模式和同步模式才能共享同一套路由语义。
            route = cache_router(state)

            if route == RouteTarget.SYNTHESIZE_RESPONSE:
                if await client_disconnected():
                    return
                state = synthesize_response_node(state)
                answer = state.get("final_response", "")
                for chunk in iter_text_chunks(answer):
                    if await client_disconnected():
                        return
                    yield stream_event(STREAM_EVENT_TOKEN, content=chunk)
                    await asyncio.sleep(0)
                if await client_disconnected():
                    return
                yield stream_event(STREAM_EVENT_FINAL, **_build_stream_ready_final_event(state, start_time, answer))
                return

            if route == RouteTarget.RESEARCH_SUPPLEMENT:
                yield stream_event(STREAM_EVENT_STATUS, stage=RouteTarget.RESEARCH_SUPPLEMENT, message="正在补充缓存未覆盖的部分...")
                await asyncio.sleep(0)
                if await client_disconnected():
                    return
                state = research_supplement_node(state)
                if await client_disconnected():
                    return
                state = synthesize_response_node(state)
                answer = state.get("final_response", "")
                for chunk in iter_text_chunks(answer):
                    if await client_disconnected():
                        return
                    yield stream_event(STREAM_EVENT_TOKEN, content=chunk)
                    await asyncio.sleep(0)
                if await client_disconnected():
                    return
                yield stream_event(STREAM_EVENT_FINAL, **_build_stream_ready_final_event(state, start_time, answer))
                return

            if route == RouteTarget.RERANK_CACHE:
                yield stream_event(STREAM_EVENT_STATUS, stage=RouteTarget.RERANK_CACHE, message="正在进行缓存语义裁判...")
                await asyncio.sleep(0)
                if await client_disconnected():
                    return
                state = rerank_cache_node(state)
                # 第二次路由同理：reranker 节点只负责写 state，
                # 如何把 state 变成响应事件仍由这里控制。
                route = cache_rerank_router(state)
                if route == RouteTarget.SYNTHESIZE_RESPONSE:
                    if await client_disconnected():
                        return
                    state = synthesize_response_node(state)
                    answer = state.get("final_response", "")
                    for chunk in iter_text_chunks(answer):
                        if await client_disconnected():
                            return
                        yield stream_event(STREAM_EVENT_TOKEN, content=chunk)
                        await asyncio.sleep(0)
                    if await client_disconnected():
                        return
                    yield stream_event(STREAM_EVENT_FINAL, **_build_stream_ready_final_event(state, start_time, answer))
                    return
                if route == RouteTarget.RESEARCH_SUPPLEMENT:
                    yield stream_event(STREAM_EVENT_STATUS, stage=RouteTarget.RESEARCH_SUPPLEMENT, message="正在补充缓存未覆盖的部分...")
                    await asyncio.sleep(0)
                    if await client_disconnected():
                        return
                    state = research_supplement_node(state)
                    if await client_disconnected():
                        return
                    state = synthesize_response_node(state)
                    answer = state.get("final_response", "")
                    for chunk in iter_text_chunks(answer):
                        if await client_disconnected():
                            return
                        yield stream_event(STREAM_EVENT_TOKEN, content=chunk)
                        await asyncio.sleep(0)
                    if await client_disconnected():
                        return
                    yield stream_event(STREAM_EVENT_FINAL, **_build_stream_ready_final_event(state, start_time, answer))
                    return

            yield stream_event(STREAM_EVENT_STATUS, stage=RouteTarget.RESEARCH, message="正在检索知识库并生成回答...")
            await asyncio.sleep(0)
            if await client_disconnected():
                return
            research_start = time.perf_counter()
            # 这里复用与同步 research 相同的 message 构造逻辑，
            # 区别只在最后的输出协议：同步模式一次性拿完整答案，
            # 流式模式则边生成边把 token 发给前端。
            messages, research_llm_invocations, needs_final_generation = prepare_research_messages(
                payload.query,
                llm_usage=state.get("llm_usage"),
                usage_lock=state.get("llm_usage_lock"),
            )

            final_generation_calls = 1 if needs_final_generation else 0
            if needs_final_generation:
                assembled_answer = ""
                usage_chunk = None
                for chunk in get_research_llm().stream(messages):
                    if await client_disconnected():
                        return
                    if _chunk_has_usage_metadata(chunk):
                        usage_chunk = chunk
                    piece = extract_chunk_text(chunk)
                    if not piece:
                        continue
                    assembled_answer += piece
                    yield stream_event(STREAM_EVENT_TOKEN, content=piece)
                    await asyncio.sleep(0)
                if usage_chunk is not None:
                    _record_llm_usage(
                        state.get("llm_usage"),
                        "research",
                        usage_chunk,
                        usage_lock=state.get("llm_usage_lock"),
                    )
            else:
                final_message = messages[-1]
                assembled_answer = getattr(final_message, "content", "") or str(final_message)
                for chunk in iter_text_chunks(assembled_answer):
                    if await client_disconnected():
                        return
                    yield stream_event(STREAM_EVENT_TOKEN, content=chunk)
                    await asyncio.sleep(0)

            research_time = (time.perf_counter() - research_start) * 1000
            metrics = update_metrics(
                state.get("metrics", initialize_metrics()),
                research_latency=research_time,
                total_research_iterations=1,
            )
            state = {
                **state,
                "answer": assembled_answer,
                "research_iterations": state.get("research_iterations", 0) + 1,
                "execution_path": state.get("execution_path", []) + ["researched"],
                "llm_calls": {
                    **state.get("llm_calls", {}),
                    "research_llm": state.get("llm_calls", {}).get("research_llm", 0) + research_llm_invocations + final_generation_calls,
                },
                "metrics": metrics,
            }

            if await client_disconnected():
                return
            state = synthesize_response_node(state)
            if await client_disconnected():
                return
            yield stream_event(
                STREAM_EVENT_FINAL,
                **_build_stream_ready_final_event(state, start_time, state.get("final_response", assembled_answer)),
            )
        except Exception as exc:
            logger.exception("Error processing streaming query")
            yield stream_event(STREAM_EVENT_ERROR, message=str(exc) or "An internal error occurred while processing your request.")

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_touch_icon() -> Response:
    return Response(status_code=204)

# --- Mount Frontend Static Files ---
# Important: This must be mounted AFTER API routes, otherwise it will intercept API calls.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
