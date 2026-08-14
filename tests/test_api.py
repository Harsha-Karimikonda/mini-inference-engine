from fastapi.testclient import TestClient
from mini_inference_engine.api import create_app


def test_openai_completion_and_streaming():
    with TestClient(create_app()) as client:
        response = client.post("/v1/completions", json={"model": "mock", "prompt": "hello", "max_tokens": 2})
        assert response.status_code == 200
        assert response.json()["choices"][0]["finish_reason"] == "stop"
        streamed = client.post("/v1/completions", json={"model": "mock", "prompt": "hello", "max_tokens": 2, "stream": True})
        assert "data: [DONE]" in streamed.text


def test_validation_and_chat():
    with TestClient(create_app()) as client:
        assert client.post("/v1/completions", json={"model": "mock", "prompt": "", "max_tokens": 2}).status_code == 422
        response = client.post("/v1/chat/completions", json={"model": "mock", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1})
        assert response.json()["object"] == "chat.completion"
