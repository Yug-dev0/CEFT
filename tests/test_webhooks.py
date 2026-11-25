"""Tests for background ingestion job webhooks and event signing."""

import hashlib
import hmac
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.schemas.knowledge import MediaModality
from app.schemas.records import JobRecord
from app.services.webhooks import dispatch_job_event, sign_payload
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)


def test_hmac_event_signing():
    secret = "my-webhook-secret"
    payload = b'{"event": "job.completed", "job_id": "test-123"}'
    sig = sign_payload(payload, secret)

    assert sig.startswith("sha256=")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    assert sig == f"sha256={expected}"


def test_dispatch_job_event_with_callback_url():
    job = JobRecord(
        id="job_test_webhook",
        source_id="src_1",
        filename="meeting.mp4",
        modality=MediaModality.VIDEO,
        file_path="/tmp/meeting.mp4",
        source_attributes={"callback_url": "https://example.com/webhook"},
    )

    with patch("app.services.webhooks._post_webhook", return_value=True) as mock_post:
        success = dispatch_job_event(job, "job.completed", result={"segments": 5}, sync=True)
        assert success is True
        assert mock_post.called
        call_args = mock_post.call_args[0]
        assert call_args[0] == "https://example.com/webhook"
        body = json.loads(call_args[1].decode("utf-8"))
        assert body["event"] == "job.completed"
        assert body["job_id"] == "job_test_webhook"
        assert body["result"]["segments"] == 5


def test_dispatch_without_callback_returns_false():
    job = JobRecord(
        id="job_no_hook",
        source_id="src_2",
        filename="notes.pdf",
        modality=MediaModality.PDF,
        file_path="/tmp/notes.pdf",
    )
    success = dispatch_job_event(job, "job.completed", sync=True)
    assert success is False


def test_api_webhook_test_endpoint(client):
    with patch("app.services.webhooks._post_webhook", return_value=True):
        resp = client.post("/jobs/webhook/test", json={"url": "https://example.com/hook", "secret": "sec123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
