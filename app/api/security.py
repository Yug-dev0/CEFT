"""HTTP hardening: request IDs and logging, security headers, optional API key.

* Every response carries ``X-Request-ID`` (the caller's, or a new one) and a
  log line with method, path, status and duration. An unhandled exception
  becomes a JSON 500 with the request ID instead of a stack trace.
* Security headers on every response. The website gets a strict Content
  Security Policy (everything from this origin, no inline scripts); the
  interactive API docs, which load Swagger UI from a CDN, are left out.
* When ``CEFT_API_KEY`` (or ``WEFT_API_KEY``) is set, every request that changes data or calls a
  model (all ``POST``, ``PUT``, ``PATCH`` and ``DELETE``) must send it as
  ``X-API-Key`` or ``Authorization: Bearer``. Reads stay open so media can be
  played by the browser; put the server behind your own auth if the data
  itself is private.
"""

from __future__ import annotations

from collections import defaultdict
import hmac
import logging
import threading
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings

_log = logging.getLogger("ceft.http")

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class SlidingWindowRateLimiter:
    """Sliding-window in-memory request rate limiter."""

    def __init__(self, limit: int = 120, window_seconds: float = 60.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self._history: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        now = time.time()
        with self._lock:
            history = self._history[key]
            cutoff = now - self.window_seconds
            self._history[key] = [t for t in history if t > cutoff]
            history = self._history[key]

            remaining = max(0, self.limit - len(history))
            reset_seconds = max(1.0, (history[0] + self.window_seconds - now)) if history else self.window_seconds

            if len(history) >= self.limit:
                return False, 0, reset_seconds

            history.append(now)
            return True, remaining - 1, reset_seconds

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


_rate_limiter = SlidingWindowRateLimiter()

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "media-src 'self' blob:",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


def presented_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def key_is_valid(expected: str, presented: str | None) -> bool:
    return bool(presented) and hmac.compare_digest(expected.encode(), presented.encode())


class HardeningMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()

        settings = get_settings()

        # Rate limiting check
        rate_headers: dict[str, str] = {}
        if settings.rate_limit_per_minute > 0:
            client_ip = request.client.host if request.client else "127.0.0.1"
            _rate_limiter.limit = settings.rate_limit_per_minute
            allowed, remaining, reset_sec = _rate_limiter.check(client_ip)
            rate_headers = {
                "X-RateLimit-Limit": str(settings.rate_limit_per_minute),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(reset_sec)),
            }
            if not allowed:
                blocked_resp = JSONResponse(
                    {"detail": "Rate limit exceeded. Please slow down."},
                    status_code=429,
                    headers={"Retry-After": str(int(reset_sec)), **rate_headers},
                )
                blocked_resp.headers["X-Request-ID"] = request_id
                return blocked_resp

        # API key authentication with scoped role check
        presented = presented_key(request)
        is_write = request.method in WRITE_METHODS

        if settings.api_key and is_write:
            admin_valid = key_is_valid(settings.api_key, presented)
            readonly_valid = bool(settings.readonly_api_key and key_is_valid(settings.readonly_api_key, presented))

            if readonly_valid:
                # Read-only keys can query endpoints but not perform mutations/uploads/deletions
                if request.url.path.startswith("/query"):
                    admin_valid = True
                else:
                    forbidden_resp = JSONResponse(
                        {"detail": "Read-only API key cannot perform mutation or upload operations."},
                        status_code=403,
                    )
                    forbidden_resp.headers["X-Request-ID"] = request_id
                    return forbidden_resp

            if not admin_valid:
                response: Response = JSONResponse(
                    {"detail": "This server requires an API key for this request. Send it as X-API-Key."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                try:
                    response = await call_next(request)
                except Exception:
                    _log.exception("Unhandled error on %s %s [%s]", request.method, request.url.path, request_id)
                    response = JSONResponse(
                        {"detail": "Internal server error.", "request_id": request_id}, status_code=500
                    )
        else:
            try:
                response = await call_next(request)
            except Exception:  # anything a route did not turn into an HTTP error
                _log.exception("Unhandled error on %s %s [%s]", request.method, request.url.path, request_id)
                response = JSONResponse(
                    {"detail": "Internal server error.", "request_id": request_id}, status_code=500
                )

        response.headers["X-Request-ID"] = request_id
        for name, value in rate_headers.items():
            response.headers[name] = value
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if not request.url.path.startswith(_DOCS_PATHS):
            response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        if request.url.path in ("/", "/app") or request.url.path.startswith("/static/"):
            # Revalidate the frontend on every load so a changed file is never served stale.
            response.headers.setdefault("Cache-Control", "no-cache")

        elapsed = (time.perf_counter() - started) * 1000
        if not request.url.path.startswith(("/static/", "/uploads/", "/derived/", "/frames/")):
            _log.info("%s %s %d %.0fms [%s]", request.method, request.url.path, response.status_code, elapsed, request_id)
        return response
