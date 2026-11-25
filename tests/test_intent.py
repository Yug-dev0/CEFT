"""Tests for multimodal query intent classifier and routing."""

from fastapi.testclient import TestClient
import pytest

from app.services.intent import classify_intent
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)


def test_classify_visual_intent():
    res = classify_intent("Show me the architecture diagram and latency graph")
    assert res.primary_intent == "visual_inspection"
    assert "image" in res.suggested_modalities
    assert res.confidence >= 0.7


def test_classify_speaker_intent():
    res = classify_intent("What did Alex say during the incident postmortem meeting?")
    assert res.primary_intent == "speaker_inquiry"
    assert "video" in res.suggested_modalities
    assert "audio" in res.suggested_modalities
    assert "Alex" in res.detected_speakers


def test_classify_document_intent():
    res = classify_intent("What does the PDF report say on page 3 about retry policies?")
    assert res.primary_intent == "document_lookup"
    assert "pdf" in res.suggested_modalities


def test_classify_log_intent():
    res = classify_intent("Check the JSON logs for 504 error code telemetry")
    assert res.primary_intent == "log_investigation"
    assert "json" in res.suggested_modalities


def test_classify_temporal_intent():
    res = classify_intent("What happened first before the deployment finished?")
    assert res.temporal_focus is True


def test_api_query_intent_endpoint(client):
    resp = client.post("/query/intent", json={"query": "Where is the Grafana dashboard screenshot?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_intent"] == "visual_inspection"
    assert "image" in data["suggested_modalities"]
