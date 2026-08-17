from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from .backends import MockBackend, TransformersBackend
from .cache import KVCache
from .config import Settings
from .metrics import metrics_response, REQUESTS, LATENCY
from .router import Router, Worker
from .scheduler import Scheduler
from .log import configure_logging, get_logger


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


def _error(message: str, error_type: str = "invalid_request_error", status: int = 400):
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type}})


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings.from_env()
    workers: list[Worker] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting inference engine", extra={"model": settings.model})
        backend_factory = lambda: (MockBackend() if settings.model == "mock" else TransformersBackend(settings.model, settings.device))
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
        if model != settings.model and not (settings.model == "mock" and model == "mock"):
            logger.warning("request rejected: model unavailable", extra={"endpoint": endpoint, "status": 404})
            return _error(f"model '{model}' is not available", status=404)
        if max_tokens > settings.max_tokens:
            logger.warning("request rejected: token limit exceeded", extra={"endpoint": endpoint, "status": 400})
            return _error(f"max_tokens exceeds configured limit of {settings.max_tokens}", status=400)
        try:
            worker = app.state.router.choose()
        except RuntimeError as exc:
            logger.error("request rejected: no healthy workers", extra={"endpoint": endpoint, "status": 503})
            return _error(str(exc), "service_unavailable", 503)
        worker.active += 1
        request_id = "cmpl-" + str(int(started * 1000000))
        logger.info("request accepted", extra={"endpoint": endpoint, "request_id": request_id, "worker": worker.worker_id})
        try:
            iterator = await worker.scheduler.submit(prompt, max_tokens, priority)
            if not stream:
                chunks = []
                async for token in iterator:
                    if isinstance(token, Exception):
                        logger.error("request failed during generation", extra={"endpoint": endpoint, "request_id": request_id, "status": 500})
                        return _error(str(token), "server_error", 500)
                    chunks.append(token)
                REQUESTS.labels(endpoint, "ok").inc()
                logger.info("generation completed", extra={"endpoint": endpoint, "request_id": request_id, "worker": worker.worker_id, "status": 200, "duration_ms": round((time.monotonic() - started) * 1000, 2)})
                return {"id": request_id, "object": "text_completion", "model": model, "choices": [{"text": "".join(chunks), "index": 0, "finish_reason": "stop"}]}

            async def events():
                try:
                    async for token in iterator:
                        if await request.is_disconnected():
                            break
                        if isinstance(token, Exception):
                            yield f"data: {json.dumps({'error': {'message': str(token), 'type': 'server_error'}})}\n\n"
                            return
                        payload = {"id": request_id, "object": "text_completion.chunk", "model": model, "choices": [{"text": token, "index": 0, "finish_reason": None}]}
                        yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    REQUESTS.labels(endpoint, "ok").inc()
                    logger.info("generation stream completed", extra={"endpoint": endpoint, "request_id": request_id, "worker": worker.worker_id, "status": 200, "duration_ms": round((time.monotonic() - started) * 1000, 2)})
                finally:
                    worker.active = max(0, worker.active - 1)

            return StreamingResponse(events(), media_type="text/event-stream")
        finally:
            if not stream:
                worker.active = max(0, worker.active - 1)

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
