"""Tests for sliding-window rate limiter and scoped API keys."""

from fastapi.testclient import TestClient
import pytest

from app.api.security import SlidingWindowRateLimiter, _rate_limiter
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _rate_limiter.reset()
    return TestClient(app)


def test_sliding_window_rate_limiter():
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=10.0)

    # 3 allowed
    ok1, rem1, _ = limiter.check("client_a")
    ok2, rem2, _ = limiter.check("client_a")
    ok3, rem3, _ = limiter.check("client_a")
    assert ok1 and ok2 and ok3
    assert rem1 == 2
    assert rem2 == 1
    assert rem3 == 0

    # 4th is rejected
    ok4, rem4, reset_sec = limiter.check("client_a")
    assert not ok4
    assert rem4 == 0
    assert reset_sec > 0

    # Different client is still allowed
    ok_b, _, _ = limiter.check("client_b")
    assert ok_b


def test_rate_limiting_middleware(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    _rate_limiter.reset()

    # Request 1: OK
    r1 = client.get("/health")
    assert r1.status_code == 200
    assert r1.headers.get("X-RateLimit-Limit") == "2"

    # Request 2: OK
    r2 = client.get("/health")
    assert r2.status_code == 200

    # Request 3: 429 Too Many Requests
    r3 = client.get("/health")
    assert r3.status_code == 429
    assert "Rate limit exceeded" in r3.json()["detail"]
    assert "Retry-After" in r3.headers


def test_scoped_api_keys(client, monkeypatch):
    monkeypatch.setenv("CEFT_API_KEY", "admin-secret-key")
    monkeypatch.setenv("CEFT_READONLY_API_KEY", "reader-secret-key")
    _rate_limiter.reset()

    # 1. Unauthenticated write fails with 401
    r_unauth = client.post("/query", json={"query": "test"})
    assert r_unauth.status_code == 401

    # 2. Read-only key allows /query
    r_ro_query = client.post("/query", json={"query": "test"}, headers={"X-API-Key": "reader-secret-key"})
    assert r_ro_query.status_code == 200

    # 3. Read-only key is forbidden for mutations like /sources/abc delete
    r_ro_del = client.delete("/sources/abc", headers={"X-API-Key": "reader-secret-key"})
    assert r_ro_del.status_code == 403

    # 4. Admin key allows both
    r_admin_del = client.delete("/sources/nonexistent", headers={"X-API-Key": "admin-secret-key"})
    assert r_admin_del.status_code == 404  # passed auth, reached route
