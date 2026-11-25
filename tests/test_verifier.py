"""Tests for citation verification and hallucination detection."""

import pytest
from fastapi.testclient import TestClient

from app.services.verifier import verify_answer
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)


def test_verify_supported_answer():
    answer = "The team resolved the timeout by configuring exponential backoff [1]."
    sources = [
        {
            "citation": 1,
            "source": "incident_meeting.mp4",
            "snippet": "We decided to configure exponential backoff to fix the timeout issue.",
        }
    ]

    res = verify_answer(answer, sources)
    assert res.is_verified is True
    assert res.faithfulness_score >= 0.6
    assert res.citation_precision == 1.0
    assert res.total_claims == 1
    assert res.claims[0].status == "supported"
    assert res.hallucination_warnings == []


def test_verify_hallucinated_citation_index():
    answer = "The database load spiked [1], and the memory leaked on checkout [5]."
    sources = [
        {
            "citation": 1,
            "source": "report.pdf",
            "snippet": "Database load spiked due to retries.",
        }
    ]

    res = verify_answer(answer, sources)
    assert res.is_verified is False
    assert res.citation_precision == 0.5  # 1 of 2 citations is valid
    assert any("Citation [5]" in w for w in res.hallucination_warnings)


def test_verify_uncited_claims():
    answer = "Everything was working fine. The team went for lunch."
    sources = []

    res = verify_answer(answer, sources)
    assert res.total_claims == 2
    assert all(c.status == "uncited" for c in res.claims)


def test_api_verify_claim_endpoint(client):
    payload = {
        "answer": "Switching to exponential backoff resolved checkout 504 errors [1].",
        "sources": [
            {
                "citation": 1,
                "source": "postmortem.pdf",
                "snippet": "Exponential backoff resolved the checkout 504 errors completely.",
            }
        ],
    }
    resp = client.post("/query/verify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_verified"] is True
    assert data["faithfulness_score"] >= 0.6
    assert data["citation_precision"] == 1.0
    assert len(data["claims"]) == 1
