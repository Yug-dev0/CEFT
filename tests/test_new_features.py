"""Tests for new features: Investigation Dossier, Chat Sessions, and Timeline."""

from fastapi.testclient import TestClient
import pytest

from app.services.chat_session import ChatSessionManager
from app.services.dossier import generate_html_dossier, generate_markdown_dossier
from app.services.timeline import get_timeline_events
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)


def test_markdown_dossier_generation():
    query = "How did the team resolve checkout timeout?"
    answer = "The team resolved it using exponential backoff [1]."
    sources = [
        {
            "citation": 1,
            "source": "incident.mp4",
            "modality": "video",
            "locator": "00:10 - 00:20",
            "confidence": 0.95,
            "snippet": "We switched to exponential backoff.",
        }
    ]
    hits = [
        {
            "source": "incident.mp4",
            "timestamp": "00:10 - 00:20",
            "score": 0.92,
            "transcript": "We switched to exponential backoff.",
            "visual_summary": "Architecture slide",
            "entities": ["Checkout Service"],
            "speakers": ["Alex"],
            "metadata": {"modality": "video"},
        }
    ]

    md = generate_markdown_dossier(query, answer, sources, hits, grounded=True, method="llm")
    assert "# Investigation Dossier:" in md
    assert "checkout timeout" in md
    assert "exponential backoff [1]" in md
    assert "| **[1]** | `incident.mp4` | `video` | 00:10 - 00:20 |" in md
    assert "Checkout Service" in md
    assert "Alex" in md


def test_html_dossier_generation():
    query = "Why did the DB load spike?"
    answer = "Excessive retries without backoff [1]."
    sources = [{"citation": 1, "source": "postmortem.pdf", "modality": "pdf", "locator": "Page 3"}]
    hits = [{"source": "postmortem.pdf", "timestamp": "Page 3", "transcript": "Retries spiked the DB"}]

    html = generate_html_dossier(query, answer, sources, hits, grounded=True)
    assert "<!DOCTYPE html>" in html
    assert "Investigation Dossier:" in html
    assert "postmortem.pdf" in html
    assert "Page 3" in html


def test_chat_session_manager():
    mgr = ChatSessionManager()
    session = mgr.get_or_create("test-sess-1")
    assert session.session_id == "test-sess-1"

    # Turn 1
    t1 = mgr.add_turn(
        session_id="test-sess-1",
        query="What happened to the checkout service?",
        reformulated_query="What happened to the checkout service?",
        answer="It suffered high latency due to connection exhaustion [1].",
        grounded=True,
        sources=[{"citation": 1, "source": "report.pdf"}],
    )
    assert t1.turn_id is not None
    assert len(session.turns) == 1

    # Turn 2: Follow-up question with pronoun "it"
    ref_q = mgr.reformulate_query(session, "How did they fix it?")
    assert "checkout service" in ref_q or "happened" in ref_q

    # List sessions
    sessions = mgr.list_sessions()
    assert any(s["session_id"] == "test-sess-1" for s in sessions)

    # Delete session
    assert mgr.delete_session("test-sess-1") is True
    assert mgr.get_session("test-sess-1") is None


def test_timeline_events(repo):
    events = get_timeline_events(repo, limit=20)
    assert isinstance(events, list)


def test_dossier_api_endpoint(client):
    resp = client.post("/query/dossier", json={"query": "checkout timeout", "format": "markdown"})
    assert resp.status_code == 200
    assert "Investigation Dossier:" in resp.text

    resp_html = client.post("/query/dossier", json={"query": "checkout timeout", "format": "html"})
    assert resp_html.status_code == 200
    assert "<!DOCTYPE html>" in resp_html.text


def test_chat_api_endpoints(client):
    # Post first message
    resp = client.post("/query/chat", json={"query": "What caused the timeout issue?"})
    assert resp.status_code == 200
    data = resp.json()
    sess_id = data["session_id"]
    assert len(data["turns"]) == 1

    # Post follow up
    resp2 = client.post("/query/chat", json={"session_id": sess_id, "query": "Who fixed it?"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["turns"]) == 2

    # Get sessions
    list_resp = client.get("/query/chat/sessions")
    assert list_resp.status_code == 200
    assert any(s["session_id"] == sess_id for s in list_resp.json())

    # Get single session
    sess_resp = client.get(f"/query/chat/{sess_id}")
    assert sess_resp.status_code == 200
    assert len(sess_resp.json()["turns"]) == 2

    # Delete session
    del_resp = client.delete(f"/query/chat/{sess_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True


def test_timeline_api_endpoint(client):
    resp = client.get("/knowledge/timeline?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
