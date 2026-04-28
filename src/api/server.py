from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
import warnings
import os
import sys
from common.env import set_ark_key
from workflow.graph import create_agent_graph
from workflow.nodes import build_initial_state
from knowledge.builder import init_app_knowledge_base
from cache.auto_heater import setup_cache
from common.logger import setup_logging
import time
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

@app.post("/validate")
async def validate_code(request: ValidateRequest):
    if request.access_code != ACCESS_CODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Access Code"
        )
    return {"status": "ok", "message": "Access code is valid"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, client_ip: str = "127.0.0.1"):
    # 1. Security Check: Validate Access Code
    if request.access_code != ACCESS_CODE:
        logger.warning(f"Failed authorization attempt from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Access Code. Please use the code provided in the resume."
        )

    # 2. Runtime Readiness Check
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
    
    # 3. Security Check: Rate Limiting
    check_rate_limit(client_ip)
    
    # 4. Process the query using LangGraph
    logger.info(f"Processing API Query: {request.query}")
    start_time = time.time()
    
    try:
        initial_state = build_initial_state(request.query)
        
        # Invoke the workflow graph
        final_state = workflow_app.invoke(initial_state)
        
        latency = round((time.time() - start_time) * 1000, 2)
        answer = final_state.get("final_response", "系统遇到了未知错误，请稍后重试。")
        cache_hit = final_state.get("cache_hit", False)
        intercepted = final_state.get("intercepted", False)
        
        logger.info(f"Answer generated in {latency}ms (Cache Hit: {cache_hit}, Intercepted: {intercepted})")
        return ChatResponse(answer=answer, latency_ms=latency, cache_hit=cache_hit, intercepted=intercepted)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")


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
