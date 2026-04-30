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
from workflow.edges import cache_router, cache_rerank_router
from workflow.nodes import (
    build_initial_state,
    check_cache_node,
    get_research_llm,
    pre_check_node,
    prepare_research_messages,
    research_supplement_node,
    rerank_cache_node,
    synthesize_response_node,
)
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

class ValidateRequest(BaseModel):
    access_code: str


@app.get("/health")
async def health_check():
    return {
        "ready": system_status["ready"],
        "stage": system_status["stage"],
        "message": system_status["message"],
        "error": system_status["last_error"],
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

def build_label_metadata(final_state: dict) -> dict:
    intercepted = final_state.get("intercepted", False)
    cache_hit = final_state.get("cache_hit", False)
    cache_match_type = final_state.get("cache_match_type", "none")
    cache_reuse_mode = final_state.get("cache_reuse_mode", "none")

    if intercepted:
        label_key = "zero_intercept"
        label_text = "Zero-Layer Intercept"
    elif cache_hit and cache_match_type == "exact":
        label_key = "cache_exact"
        label_text = "Exact Cache Hit"
    elif cache_hit and cache_match_type == "near_exact":
        label_key = "cache_near_exact"
        label_text = "Near-Exact Cache Hit"
    elif cache_reuse_mode == "full_reuse":
        label_key = "cache_semantic_reuse"
        label_text = "Reranked Cache Reuse"
    elif cache_reuse_mode == "partial_reuse":
        label_key = "cache_partial_reuse"
        label_text = "Partial Cache Reuse + RAG"
    else:
        label_key = "rag_full_research"
        label_text = "LLM Analysis / Full RAG"

    return {
        "cache_hit": cache_hit,
        "intercepted": intercepted,
        "cache_match_type": cache_match_type,
        "cache_reuse_mode": cache_reuse_mode,
        "label_key": label_key,
        "label_text": label_text,
    }

def build_chat_response(final_state: dict, latency_ms: float) -> ChatResponse:
    answer = final_state.get("final_response", "系统遇到了未知错误，请稍后重试。")
    metadata = build_label_metadata(final_state)
    return ChatResponse(answer=answer, latency_ms=latency_ms, **metadata)

def build_stream_final_event(final_state: dict, latency_ms: float, answer: str) -> dict:
    return {
        "answer": answer,
        "latency_ms": latency_ms,
        **build_label_metadata(final_state),
    }

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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request):
    client_ip = get_client_ip(request)
    validate_chat_request(payload, client_ip)

    logger.info(f"Processing API Query: {payload.query}")
    start_time = time.time()
    
    try:
        initial_state = build_initial_state(payload.query)
        
        # Invoke the workflow graph
        final_state = workflow_app.invoke(initial_state)
        
        latency = round((time.time() - start_time) * 1000, 2)
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
        start_time = time.time()
        try:
            logger.info(f"Processing streaming API Query: {payload.query}")
            state = build_initial_state(payload.query)

            yield stream_event("status", stage="pre_check", message="正在进行前置校验...")
            await asyncio.sleep(0)
            state = pre_check_node(state)

            if state.get("intercepted", False):
                state = synthesize_response_node(state)
                answer = state.get("final_response", "")
                for chunk in iter_text_chunks(answer):
                    yield stream_event("token", content=chunk)
                    await asyncio.sleep(0)
                latency = round((time.time() - start_time) * 1000, 2)
                yield stream_event("final", **build_stream_final_event(state, latency, answer))
                return

            yield stream_event("status", stage="check_cache", message="正在检查缓存...")
            await asyncio.sleep(0)
            state = check_cache_node(state)
            route = cache_router(state)

            if route == "synthesize_response":
                state = synthesize_response_node(state)
                answer = state.get("final_response", "")
                for chunk in iter_text_chunks(answer):
                    yield stream_event("token", content=chunk)
                    await asyncio.sleep(0)
                latency = round((time.time() - start_time) * 1000, 2)
                yield stream_event("final", **build_stream_final_event(state, latency, answer))
                return

            if route == "rerank_cache":
                yield stream_event("status", stage="rerank_cache", message="正在进行缓存语义裁判...")
                await asyncio.sleep(0)
                state = rerank_cache_node(state)
                route = cache_rerank_router(state)
                if route == "synthesize_response":
                    state = synthesize_response_node(state)
                    answer = state.get("final_response", "")
                    for chunk in iter_text_chunks(answer):
                        yield stream_event("token", content=chunk)
                        await asyncio.sleep(0)
                    latency = round((time.time() - start_time) * 1000, 2)
                    yield stream_event("final", **build_stream_final_event(state, latency, answer))
                    return
                if route == "research_supplement":
                    yield stream_event("status", stage="research_supplement", message="正在补充缓存未覆盖的部分...")
                    await asyncio.sleep(0)
                    state = research_supplement_node(state)
                    state = synthesize_response_node(state)
                    answer = state.get("final_response", "")
                    for chunk in iter_text_chunks(answer):
                        yield stream_event("token", content=chunk)
                        await asyncio.sleep(0)
                    latency = round((time.time() - start_time) * 1000, 2)
                    yield stream_event("final", **build_stream_final_event(state, latency, answer))
                    return

            yield stream_event("status", stage="research", message="正在检索知识库并生成回答...")
            await asyncio.sleep(0)
            messages, research_llm_invocations, needs_final_generation = prepare_research_messages(payload.query)

            if needs_final_generation:
                assembled_answer = ""
                for chunk in get_research_llm().stream(messages):
                    piece = extract_chunk_text(chunk)
                    if not piece:
                        continue
                    assembled_answer += piece
                    yield stream_event("token", content=piece)
                    await asyncio.sleep(0)
                state = {
                    **state,
                    "answer": assembled_answer,
                    "research_iterations": state.get("research_iterations", 0) + 1,
                    "execution_path": state.get("execution_path", []) + ["researched"],
                    "llm_calls": {
                        **state.get("llm_calls", {}),
                        "research_llm": state.get("llm_calls", {}).get("research_llm", 0) + research_llm_invocations + 1,
                    },
                }
            else:
                final_message = messages[-1]
                assembled_answer = getattr(final_message, "content", "") or str(final_message)
                for chunk in iter_text_chunks(assembled_answer):
                    yield stream_event("token", content=chunk)
                    await asyncio.sleep(0)
                state = {
                    **state,
                    "answer": assembled_answer,
                    "research_iterations": state.get("research_iterations", 0) + 1,
                    "execution_path": state.get("execution_path", []) + ["researched"],
                    "llm_calls": {
                        **state.get("llm_calls", {}),
                        "research_llm": state.get("llm_calls", {}).get("research_llm", 0) + research_llm_invocations,
                    },
                }

            state = synthesize_response_node(state)
            latency = round((time.time() - start_time) * 1000, 2)
            yield stream_event("final", **build_stream_final_event(state, latency, state.get("final_response", assembled_answer)))
        except Exception as exc:
            logger.exception("Error processing streaming query")
            yield stream_event("error", message=str(exc) or "An internal error occurred while processing your request.")

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
