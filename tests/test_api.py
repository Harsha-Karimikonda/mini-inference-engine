from fastapi.testclient import TestClient

from mini_inference_engine.api import create_app


def test_openai_completion_and_streaming():
    with TestClient(create_app()) as client:
        response = client.post("/v1/completions", json={"model": "mock", "prompt": "hello world", "max_tokens": 2})
        assert response.status_code == 200
        data = response.json()
        assert data["id"].startswith("cmpl-")
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 2
        assert data["usage"]["completion_tokens"] == 2
        assert data["usage"]["total_tokens"] == 4

        streamed = client.post("/v1/completions", json={"model": "mock", "prompt": "hello", "max_tokens": 2, "stream": True})
        assert streamed.status_code == 200
        assert "text_completion.chunk" in streamed.text
        assert "data: [DONE]" in streamed.text


def test_validation_and_chat():
    with TestClient(create_app()) as client:
        assert client.post("/v1/completions", json={"model": "mock", "prompt": "", "max_tokens": 2}).status_code == 422

        # Non-streaming chat
        response = client.post("/v1/chat/completions", json={"model": "mock", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data

        # Streaming chat
        streamed = client.post("/v1/chat/completions", json={"model": "mock", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2, "stream": True})
        assert streamed.status_code == 200
        assert "chat.completion.chunk" in streamed.text
        assert '"delta":' in streamed.text
        assert "data: [DONE]" in streamed.text


def test_models_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "mock"
        assert data["data"][0]["owned_by"] == "mini-together"


def test_dashboard_and_status_endpoints():
    with TestClient(create_app()) as client:
        # Check /dashboard and / return HTML
        dash = client.get("/dashboard")
        assert dash.status_code == 200
        assert "<title>Mini-Together Inference Engine</title>" in dash.text

        root = client.get("/")
        assert root.status_code == 200
        assert "<title>Mini-Together Inference Engine</title>" in root.text

        # Check /api/status returns json with workers and metrics
        status = client.get("/api/status")
        assert status.status_code == 200
        data = status.json()
        assert data["model"] == "mock"
        assert len(data["workers"]) == 2
        assert "latency_ms" in data["workers"][0]
        assert "cache_utilization" in data["workers"][0]
        assert "metrics" in data
