import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel, Field

app = FastAPI(title="News Agent Server", version="1.0")

def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


RATE_LIMIT_WINDOW_SEC = _get_int_env("RATE_LIMIT_WINDOW_SEC", 98)
RATE_LIMIT_MAX_REQUESTS = _get_int_env("RATE_LIMIT_MAX_REQUESTS", 15)
JOB_TTL_SEC = _get_int_env("JOB_TTL_SEC", 3600)
JOB_CLEANUP_INTERVAL_SEC = _get_int_env("JOB_CLEANUP_INTERVAL_SEC", 300)
CACHE_TTL_SEC = _get_int_env("CACHE_TTL_SEC", 600)
_RATE_LIMIT_STATE: Dict[str, Dict[str, float]] = {}
_RATE_LIMIT_LOCK = asyncio.Lock()
_RATE_LIMIT_EXCLUDE = {"/health"}

_JOB_STORE: Dict[str, Dict] = {}
_CLEANUP_TASK: Optional[asyncio.Task] = None
_CACHE_STORE: Dict[str, Dict] = {}

SCHEMA_PATH = Path(__file__).resolve().parent / "news_agent.schema.json"
_SCHEMA_ERROR = None
_SCHEMA_VALIDATOR = None
try:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _SCHEMA_VALIDATOR = Draft202012Validator(schema)
except Exception as exc:  # pragma: no cover - schema load failure should surface in response
    _SCHEMA_ERROR = exc


class NewsAgentRequest(BaseModel):
    symbol: Optional[str] = Field(default=None, description="Symbol, e.g., BBCA.JK")
    query: Optional[str] = Field(default=None, description="Search query override")
    max_articles: int = Field(default=20, ge=1, le=200)
    detail_limit: int = Field(default=3, ge=0, le=50)
    headless: bool = Field(default=True)


class TokenBucketRateLimiter(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.capacity = RATE_LIMIT_MAX_REQUESTS
        self.refill_rate = RATE_LIMIT_MAX_REQUESTS / RATE_LIMIT_WINDOW_SEC

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _RATE_LIMIT_EXCLUDE:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()

        async with _RATE_LIMIT_LOCK:
            state = _RATE_LIMIT_STATE.get(client, {"tokens": self.capacity, "ts": now})
            elapsed = now - state["ts"]
            tokens = min(self.capacity, state["tokens"] + elapsed * self.refill_rate)
            if tokens < 1:
                return JSONResponse(
                    status_code=429,
                    content={"error": "rate_limit_exceeded", "message": "Rate limit exceeded"},
                )
            state["tokens"] = tokens - 1
            state["ts"] = now
            _RATE_LIMIT_STATE[client] = state

        return await call_next(request)


app.add_middleware(TokenBucketRateLimiter)


def _cleanup_jobs(now: Optional[float] = None) -> None:
    if JOB_TTL_SEC <= 0:
        return
    now = now or time.time()
    expired = [
        job_id
        for job_id, job in _JOB_STORE.items()
        if now - job.get("updated_at", now) > JOB_TTL_SEC
    ]
    for job_id in expired:
        _JOB_STORE.pop(job_id, None)


def _cleanup_cache(now: Optional[float] = None) -> None:
    if CACHE_TTL_SEC <= 0:
        return
    now = now or time.time()
    expired = [
        key
        for key, entry in _CACHE_STORE.items()
        if now - entry.get("ts", now) > CACHE_TTL_SEC
    ]
    for key in expired:
        _CACHE_STORE.pop(key, None)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SEC)
        _cleanup_jobs()
        _cleanup_cache()


@app.on_event("startup")
async def _startup() -> None:
    global _CLEANUP_TASK
    if JOB_CLEANUP_INTERVAL_SEC > 0:
        _CLEANUP_TASK = asyncio.create_task(_cleanup_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _CLEANUP_TASK
    if _CLEANUP_TASK:
        _CLEANUP_TASK.cancel()
        _CLEANUP_TASK = None


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "schema_loaded": _SCHEMA_VALIDATOR is not None,
        "rate_limit": {
            "window_sec": RATE_LIMIT_WINDOW_SEC,
            "max_requests": RATE_LIMIT_MAX_REQUESTS,
            "mode": "token_bucket",
        },
        "jobs": {
            "ttl_sec": JOB_TTL_SEC,
            "cleanup_interval_sec": JOB_CLEANUP_INTERVAL_SEC,
            "count": len(_JOB_STORE),
        },
        "cache": {
            "ttl_sec": CACHE_TTL_SEC,
            "count": len(_CACHE_STORE),
        },
    }


async def _execute_agent(payload: NewsAgentRequest) -> Dict:
    if not payload.symbol and not payload.query:
        raise HTTPException(status_code=400, detail="Either symbol or query is required")

    agent_path = Path(__file__).resolve().parent / "news_agent.py"
    if not agent_path.exists():
        raise HTTPException(status_code=500, detail="news_agent.py not found")

    cache_key = json.dumps(payload.dict(), sort_keys=True)
    _cleanup_cache()
    cached = _CACHE_STORE.get(cache_key)
    if cached and (time.time() - cached.get("ts", 0) <= CACHE_TTL_SEC):
        return cached["result"]

    cmd = [
        sys.executable,
        str(agent_path),
        "--max-articles",
        str(payload.max_articles),
        "--detail-limit",
        str(payload.detail_limit),
        "--headless",
        "true" if payload.headless else "false",
    ]

    if payload.symbol:
        cmd.extend(["--symbol", payload.symbol])
    if payload.query:
        cmd.extend(["--query", payload.query])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=240)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent execution timed out")

    if proc.returncode != 0:
        detail = (stderr or b"").decode(errors="ignore").strip() or "Agent execution failed"
        raise HTTPException(status_code=500, detail=detail)

    try:
        data = json.loads((stdout or b"").decode(errors="ignore"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Agent output is not valid JSON")

    if _SCHEMA_VALIDATOR is None:
        raise HTTPException(status_code=500, detail=f"Schema validator unavailable: {_SCHEMA_ERROR}")
    try:
        _SCHEMA_VALIDATOR.validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "schema_validation_failed",
                "message": exc.message,
                "path": [str(p) for p in exc.path],
                "validator": exc.validator,
            },
        )

    if CACHE_TTL_SEC > 0:
        _CACHE_STORE[cache_key] = {"ts": time.time(), "result": data}

    return data


async def _run_job(job_id: str, payload: NewsAgentRequest) -> None:
    _JOB_STORE[job_id]["status"] = "running"
    _JOB_STORE[job_id]["updated_at"] = time.time()
    try:
        result = await _execute_agent(payload)
        _JOB_STORE[job_id]["status"] = "done"
        _JOB_STORE[job_id]["result"] = result
    except HTTPException as exc:
        _JOB_STORE[job_id]["status"] = "error"
        _JOB_STORE[job_id]["error"] = exc.detail
        _JOB_STORE[job_id]["status_code"] = exc.status_code
    except Exception as exc:  # pragma: no cover - unexpected error
        _JOB_STORE[job_id]["status"] = "error"
        _JOB_STORE[job_id]["error"] = str(exc)
        _JOB_STORE[job_id]["status_code"] = 500
    finally:
        _JOB_STORE[job_id]["updated_at"] = time.time()


@app.post("/news-agent/run")
async def run_news_agent(payload: NewsAgentRequest):
    try:
        return await _execute_agent(payload)
    except HTTPException as exc:
        if exc.status_code == 422 and isinstance(exc.detail, dict):
            return JSONResponse(status_code=422, content=exc.detail)
        raise


@app.post("/news-agent/queue")
async def queue_news_agent(payload: NewsAgentRequest):
    _cleanup_jobs()
    _cleanup_cache()
    job_id = uuid.uuid4().hex
    _JOB_STORE[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    asyncio.create_task(_run_job(job_id, payload))
    return {"job_id": job_id, "status": "queued"}


@app.get("/news-agent/jobs/{job_id}")
def get_job_status(job_id: str):
    _cleanup_jobs()
    _cleanup_cache()
    job = _JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.post("/news-agent/admin/clear-jobs")
def clear_jobs():
    cleared = len(_JOB_STORE)
    _JOB_STORE.clear()
    return {"cleared": cleared}
