from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from . import metrics as prometheus_metrics
from .backends import MockBackend, TransformersBackend
from .cache import KVCache
from .config import Settings
from .dashboard import get_dashboard_html
from .errors import AdmissionError, NoHealthyWorkers
from .log import configure_logging, get_logger
from .metrics import LATENCY, REQUESTS, metrics_response
from .router import Router, Worker
from .scheduler import Scheduler

logger = get_logger("api")


class CompletionRequest(BaseModel):
    model: str = "mock"
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=16, ge=1, le=4096)
    stream: bool = False
    priority: int = Field(default=0, ge=-10, le=10)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "mock"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=16, ge=1, le=4096)
    stream: bool = False
    priority: int = Field(default=0, ge=-10, le=10)

    @model_validator(mode="after")
    def has_content(self):
        if not any(message.content for message in self.messages):
            raise ValueError("messages must contain content")
        return self


def _error(message: str, error_type: str = "invalid_request_error", status: int = 400, headers: dict[str, str] | None = None):
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type}}, headers=headers)


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings.from_env()
    workers: list[Worker] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting inference engine", extra={"model": settings.model})
        if settings.model == "mock":
            backend_factory = lambda: MockBackend()
        else:
            shared_backend = TransformersBackend(settings.model, settings.device)
            backend_factory = lambda: shared_backend
        for index in range(2):
            scheduler = Scheduler(backend_factory(), KVCache(settings.cache_blocks, settings.cache_block_tokens), settings.max_batch_size, settings.batch_window_ms, settings.max_queue_size)
            await scheduler.start()
            workers.append(Worker(f"worker-{index + 1}", scheduler))
            logger.info("worker started", extra={"worker": f"worker-{index + 1}"})
        app.state.router = Router(workers, settings.routing_policy, settings.heartbeat_timeout_s)
        yield
        for worker in workers:
            await worker.scheduler.stop()
            logger.info("worker stopped", extra={"worker": worker.worker_id})
        workers.clear()
        logger.info("inference engine stopped")

    app = FastAPI(title="Mini-Together", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_request_timing(request: Request, call_next):
        """Record end-to-end timing for every HTTP response.

        Wrapping the response body iterator ensures streaming requests are
        timed until the stream actually finishes, rather than only until the
        ``StreamingResponse`` object is created.
        """

        started = time.perf_counter()

        def completed(status: int) -> None:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            LATENCY.observe(duration_ms / 1000)
            logger.info(
                "request completed",
                extra={
                    "endpoint": request.url.path,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )

        try:
            response = await call_next(request)
        except Exception:
            completed(500)
            raise

        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            completed(response.status_code)
            return response

        async def timed_body():
            try:
                async for chunk in body_iterator:
                    yield chunk
            except Exception:
                completed(500)
                raise
            else:
                completed(response.status_code)

        response.body_iterator = timed_body()
        return response

    async def generate(request: Request, prompt: str, model: str, max_tokens: int, stream: bool, priority: int, endpoint: str):
        started = time.monotonic()
        if model in ("mock", "default") and settings.model != "mock":
            model = settings.model
        elif model != settings.model and not (settings.model == "mock" and model == "mock"):
            logger.warning("request rejected: model unavailable", extra={"endpoint": endpoint, "status": 404})
            return _error(f"model '{model}' is not available", status=404)
        if max_tokens > settings.max_tokens:
            logger.warning("request rejected: token limit exceeded", extra={"endpoint": endpoint, "status": 400})
            return _error(f"max_tokens exceeds configured limit of {settings.max_tokens}", status=400)
        try:
            worker = app.state.router.choose()
        except NoHealthyWorkers as exc:
            logger.error("request rejected: no healthy workers", extra={"endpoint": endpoint, "status": 503})
            return _error(str(exc), "service_unavailable", 503)

        prefix = "chatcmpl-" if endpoint == "chat" else "cmpl-"
        request_id = prefix + uuid.uuid4().hex
        logger.info("request accepted", extra={"endpoint": endpoint, "request_id": request_id, "worker": worker.worker_id})
        try:
            iterator = await worker.scheduler.submit(prompt, max_tokens, priority)
        except AdmissionError as exc:
            logger.warning("request rejected: scheduler queue full", extra={"endpoint": endpoint, "status": 429})
            return _error(str(exc), "rate_limit_exceeded", 429, headers={"Retry-After": "1"})

        if not stream:
            chunks = []
            async for token in iterator:
                if isinstance(token, Exception):
                    logger.error("request failed during generation", extra={"endpoint": endpoint, "request_id": request_id, "status": 500})
                    return _error(str(token), "server_error", 500)
                chunks.append(token)
            REQUESTS.labels(endpoint, "ok").inc()
            logger.info("generation completed", extra={"endpoint": endpoint, "request_id": request_id, "worker": worker.worker_id, "status": 200, "duration_ms": round((time.monotonic() - started) * 1000, 2)})
            prompt_tokens = len(prompt.split())
            completion_tokens = len(chunks)
            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"text": "".join(chunks), "index": 0, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        async def events():
            async for token in iterator:
                if await request.is_disconnected():
                    break
                if isinstance(token, Exception):
                    yield f"data: {json.dumps({'error': {'message': str(token), 'type': 'server_error'}})}\n\n"
                    return
                if endpoint == "chat":
                    payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}],
                    }
                else:
                    payload = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"text": token, "index": 0, "finish_reason": None}],
                    }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            REQUESTS.labels(endpoint, "ok").inc()
            logger.info("generation stream completed", extra={"endpoint": endpoint, "request_id": request_id, "worker": worker.worker_id, "status": 200, "duration_ms": round((time.monotonic() - started) * 1000, 2)})

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        return HTMLResponse(get_dashboard_html())

    @app.get("/api/status")
    async def api_status():
        router = getattr(app.state, "router", None)
        workers_data = []
        if router:
            router.refresh_health()
            for w in router.workers:
                cache = getattr(w.scheduler, "cache", None)
                w_info = {
                    "id": w.worker_id,
                    "healthy": w.healthy,
                    "active": w.active,
                    "queue_depth": w.queue_depth,
                    "latency_ms": round(w.latency * 1000, 2),
                }
                if cache:
                    w_info["cache_utilization"] = round(cache.utilization, 3)
                    w_info["cache_fragmentation"] = round(cache.fragmentation, 3)
                    w_info["cache_free_blocks"] = len(cache.free)
                    w_info["cache_total_blocks"] = cache.capacity
                workers_data.append(w_info)

        total_tokens = 0
        try:
            total_tokens = int(prometheus_metrics.TOKENS._value.get())
        except (AttributeError, TypeError) as exc:
            logger.debug("unable to read token metric", exc_info=exc)

        total_requests = 0
        try:
            for metric in prometheus_metrics.REQUESTS.collect():
                for sample in metric.samples:
                    if sample.name == "mini_requests_total":
                        total_requests += int(sample.value)
        except (AttributeError, IndexError) as exc:
            logger.debug("unable to read request metric", exc_info=exc)

        live_tps = round(prometheus_metrics.get_tokens_per_second(), 1)

        return {
            "model": settings.model,
            "device": settings.device,
            "policy": settings.routing_policy,
            "workers": workers_data,
            "metrics": {
                "tokens": total_tokens,
                "requests": total_requests,
                "tokens_per_sec": live_tps,
            },
        }

    @app.get("/health")
    async def health():
        router = getattr(app.state, "router", None)
        if not router:
            return {"status": "starting"}
        router.refresh_health()
        healthy = sum(worker.healthy for worker in router.workers)
        return {"status": "ok" if healthy else "unhealthy", "workers": [{"id": w.worker_id, "healthy": w.healthy} for w in router.workers]}

    @app.get("/metrics")
    async def metrics():
        body, content_type = metrics_response()
        return Response(body, media_type=content_type.split(";")[0], headers={"Content-Type": content_type})

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mini-together",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(payload: CompletionRequest, request: Request):
        return await generate(request, payload.prompt, payload.model, payload.max_tokens, payload.stream, payload.priority, "completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatRequest, request: Request):
        prompt = "\n".join(f"{message.role}: {message.content}" for message in payload.messages) + "\nassistant:"
        response = await generate(request, prompt, payload.model, payload.max_tokens, payload.stream, payload.priority, "chat")
        if payload.stream or not isinstance(response, dict):
            return response
        response["object"] = "chat.completion"
        response["choices"][0]["message"] = {"role": "assistant", "content": response["choices"][0].pop("text")}
        return response

    return app


app = create_app()
