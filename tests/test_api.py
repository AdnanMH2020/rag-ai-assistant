from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_returns_response():
    response = client.post("/ask", json={
        "question": "What is this document about?"
    })

    assert response.status_code == 200

    data = response.json()

    assert "question" in data
    assert "answer" in data
    assert "latency_seconds" in data


def test_answer_is_not_empty():
    response = client.post("/ask", json={
        "question": "Explain the main topic"
    })

    data = response.json()

    assert len(data["answer"]) > 0