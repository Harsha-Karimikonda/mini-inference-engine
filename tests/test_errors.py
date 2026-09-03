import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from mini_inference_engine.api import create_app
from mini_inference_engine.errors import (
    AdmissionError,
    BackendError,
    CachePressure,
    GenerationError,
    InferenceEngineError,
    InvalidRequestError,
    MiniInferenceError,
    ModelNotFoundError,
    NoHealthyWorkers,
    RequestCancelledError,
    TokenLimitExceededError,
    WorkerUnavailableError,
    error_response,
)


def test_queue_full_returns_429():
    app = create_app()
    with TestClient(app) as client:
        # Simulate worker queue saturation by having submit raise AdmissionError
        for worker in app.state.router.workers:
            worker.scheduler.submit = AsyncMock(side_effect=AdmissionError("scheduler admission limit reached"))

        response = client.post("/v1/completions", json={"model": "mock", "prompt": "hello", "max_tokens": 16})
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "1"
        data = response.json()
        assert data["error"]["type"] == "rate_limit_exceeded"
        assert "scheduler admission limit reached" in data["error"]["message"]


def test_no_healthy_workers_returns_503():
    app = create_app()
    with TestClient(app) as client:
        # Mark all workers unhealthy
        for worker in app.state.router.workers:
            worker.healthy = False
            worker.last_heartbeat = 0
            worker.scheduler.running = False
        response = client.post("/v1/completions", json={"model": "mock", "prompt": "hello", "max_tokens": 2})
        assert response.status_code == 503
        data = response.json()
        assert data["error"]["type"] == "service_unavailable"


def test_error_hierarchy_and_attributes():
    assert MiniInferenceError is InferenceEngineError
    err = InferenceEngineError("custom error", status_code=418, error_type="teapot_error")
    assert isinstance(err, RuntimeError)
    assert err.status_code == 418
    assert err.error_type == "teapot_error"
    assert err.to_dict() == {"error": {"message": "custom error", "type": "teapot_error"}}

    res = err.to_response()
    assert res.status_code == 418
    assert json.loads(res.body.decode()) == {"error": {"message": "custom error", "type": "teapot_error"}}


def test_all_custom_exceptions():
    adm = AdmissionError()
    assert adm.status_code == 429
    assert adm.error_type == "rate_limit_exceeded"
    assert adm.headers == {"Retry-After": "1"}
    assert isinstance(adm, InferenceEngineError)

    nhw = NoHealthyWorkers()
    assert nhw.status_code == 503
    assert nhw.error_type == "service_unavailable"
    assert isinstance(nhw, InferenceEngineError)

    cp = CachePressure()
    assert cp.status_code == 503
    assert isinstance(cp, InferenceEngineError)

    mnf = ModelNotFoundError("gpt-4")
    assert mnf.status_code == 404
    assert mnf.error_type == "invalid_request_error"
    assert "gpt-4" in mnf.message

    tle = TokenLimitExceededError(limit=4096, max_tokens=5000)
    assert tle.status_code == 400
    assert tle.limit == 4096
    assert tle.max_tokens == 5000
    assert "4096" in tle.message

    ire = InvalidRequestError("bad param")
    assert ire.status_code == 400
    assert ire.error_type == "invalid_request_error"

    be = BackendError("gpu failure")
    assert be.status_code == 500
    assert be.error_type == "server_error"

    ge = GenerationError("oom")
    assert isinstance(ge, BackendError)
    assert ge.status_code == 500

    rce = RequestCancelledError()
    assert rce.status_code == 499
    assert rce.error_type == "client_cancelled"

    wue = WorkerUnavailableError(worker_id="w-1")
    assert wue.status_code == 503
    assert "w-1" in wue.message


def test_error_response_helper():
    # From InferenceEngineError
    resp1 = error_response(AdmissionError())
    assert resp1.status_code == 429
    assert json.loads(resp1.body.decode())["error"]["type"] == "rate_limit_exceeded"

    # From generic Exception
    resp2 = error_response(ValueError("invalid value"))
    assert resp2.status_code == 500
    assert json.loads(resp2.body.decode())["error"]["message"] == "invalid value"

    # From string
    resp3 = error_response("not found", status_code=404)
    assert resp3.status_code == 404
    assert json.loads(resp3.body.decode())["error"]["message"] == "not found"
