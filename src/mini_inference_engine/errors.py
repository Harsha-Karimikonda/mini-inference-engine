from __future__ import annotations

from typing import Any

__all__ = [
    "AdmissionError",
    "BackendError",
    "CachePressure",
    "GenerationError",
    "InferenceEngineError",
    "InvalidRequestError",
    "MiniInferenceError",
    "ModelNotFoundError",
    "NoHealthyWorkers",
    "RequestCancelledError",
    "TokenLimitExceededError",
    "WorkerUnavailableError",
    "error_response",
]


class InferenceEngineError(RuntimeError):
    """Base exception for all mini-inference-engine errors."""

    status_code: int = 500
    error_type: str = "server_error"
    headers: dict[str, str] | None = None

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_type: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type
        if headers is not None:
            self.headers = headers

    def to_dict(self) -> dict[str, Any]:
        """Convert the error to an OpenAI-compatible dictionary."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
            }
        }

    def to_response(self) -> Any:
        """Convert the error to a FastAPI JSONResponse."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=self.status_code,
            content=self.to_dict(),
            headers=self.headers,
        )


# Alias for InferenceEngineError
MiniInferenceError = InferenceEngineError


class AdmissionError(InferenceEngineError):
    """Raised when the scheduler queue is full and cannot admit new jobs."""

    status_code: int = 429
    error_type: str = "rate_limit_exceeded"

    def __init__(
        self,
        message: str = "scheduler admission limit reached",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            error_type="rate_limit_exceeded",
            headers=headers if headers is not None else {"Retry-After": "1"},
        )


class NoHealthyWorkers(InferenceEngineError):
    """Raised when no healthy workers are available to route requests to."""

    status_code: int = 503
    error_type: str = "service_unavailable"

    def __init__(
        self,
        message: str = "no healthy workers",
        status_code: int = 503,
        error_type: str = "service_unavailable",
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
        )


class CachePressure(InferenceEngineError):
    """Raised when KV cache blocks cannot be allocated due to memory pressure."""

    status_code: int = 503
    error_type: str = "server_error"

    def __init__(
        self,
        message: str = "cache capacity exceeded",
        status_code: int = 503,
        error_type: str = "server_error",
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
        )


class ModelNotFoundError(InferenceEngineError):
    """Raised when the requested model is not available."""

    status_code: int = 404
    error_type: str = "invalid_request_error"

    def __init__(
        self,
        model: str,
        message: str | None = None,
    ) -> None:
        self.model = model
        msg = message or f"model '{model}' is not available"
        super().__init__(
            message=msg,
            status_code=404,
            error_type="invalid_request_error",
        )


class TokenLimitExceededError(InferenceEngineError):
    """Raised when max_tokens exceeds configured limits."""

    status_code: int = 400
    error_type: str = "invalid_request_error"

    def __init__(
        self,
        message: str | None = None,
        max_tokens: int | None = None,
        limit: int | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.limit = limit
        if message is None:
            if limit is not None:
                message = f"max_tokens exceeds configured limit of {limit}"
            else:
                message = "max_tokens exceeds configured limit"
        super().__init__(
            message=message,
            status_code=400,
            error_type="invalid_request_error",
        )


class InvalidRequestError(InferenceEngineError):
    """Raised when request payload or parameters are invalid."""

    status_code: int = 400
    error_type: str = "invalid_request_error"

    def __init__(
        self,
        message: str = "invalid request",
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
        )


class BackendError(InferenceEngineError):
    """Raised when the underlying model backend fails during execution."""

    status_code: int = 500
    error_type: str = "server_error"

    def __init__(
        self,
        message: str = "backend generation failed",
        status_code: int = 500,
        error_type: str = "server_error",
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
        )


class GenerationError(BackendError):
    """Raised when token generation fails during inference."""

    def __init__(
        self,
        message: str = "generation failed",
        status_code: int = 500,
        error_type: str = "server_error",
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
        )


class RequestCancelledError(InferenceEngineError):
    """Raised when a request is cancelled by the client or scheduler."""

    status_code: int = 499
    error_type: str = "client_cancelled"

    def __init__(
        self,
        message: str = "request was cancelled",
        status_code: int = 499,
        error_type: str = "client_cancelled",
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
        )


class WorkerUnavailableError(InferenceEngineError):
    """Raised when a specific worker is unreachable, dead, or draining."""

    status_code: int = 503
    error_type: str = "service_unavailable"

    def __init__(
        self,
        message: str = "worker unavailable",
        worker_id: str | None = None,
        status_code: int = 503,
        error_type: str = "service_unavailable",
    ) -> None:
        self.worker_id = worker_id
        if worker_id and message == "worker unavailable":
            message = f"worker '{worker_id}' is unavailable"
        super().__init__(
            message=message,
            status_code=status_code,
            error_type=error_type,
        )


def error_response(
    error: InferenceEngineError | Exception | str,
    status_code: int | None = None,
    error_type: str | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Create a standardized FastAPI JSONResponse from an error."""
    from fastapi.responses import JSONResponse

    if isinstance(error, InferenceEngineError):
        code = status_code if status_code is not None else error.status_code
        err_type = error_type if error_type is not None else error.error_type
        hdrs = headers if headers is not None else error.headers
        msg = error.message
    elif isinstance(error, Exception):
        code = status_code or 500
        err_type = error_type or "server_error"
        hdrs = headers
        msg = str(error)
    else:
        code = status_code or 400
        err_type = error_type or "invalid_request_error"
        hdrs = headers
        msg = str(error)

    return JSONResponse(
        status_code=code,
        content={"error": {"message": msg, "type": err_type}},
        headers=hdrs,
    )
