"""Webhook and event notification system for background ingestion jobs.

Dispatches HMAC-SHA256 signed event payloads to registered callback URLs
or environment-configured webhook endpoints on job completion or failure.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
import threading
import urllib.request
from typing import Any

from app.schemas.records import JobRecord

_log = logging.getLogger("ceft.webhooks")


def sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Generate HMAC-SHA256 hex signature for event validation."""
    mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def _post_webhook(url: str, payload_bytes: bytes, signature: str | None = None) -> bool:
    headers = {"Content-Type": "application/json", "User-Agent": "Ceft-Webhook/0.3.0"}
    if signature:
        headers["X-CEFT-Signature"] = signature

    req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        _log.warning("Webhook dispatch to %s failed: %s", url, exc)
        return False


def dispatch_job_event(
    job: JobRecord,
    event: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    sync: bool = False,
) -> bool:
    """Send job lifecycle event to webhook URL (asynchronously by default)."""
    callback_url = job.source_attributes.get("callback_url") or os.getenv("CEFT_WEBHOOK_URL")
    if not callback_url:
        return False

    secret = os.getenv("CEFT_WEBHOOK_SECRET")
    modality_val = job.modality.value if hasattr(job.modality, "value") else str(job.modality)
    payload = {
        "event": event,
        "job_id": job.id,
        "source_id": job.source_id,
        "filename": job.filename,
        "modality": modality_val,
        "result": result or {},
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload_bytes = json.dumps(payload, default=str).encode("utf-8")
    sig = sign_payload(payload_bytes, secret) if secret else None

    if sync:
        return _post_webhook(callback_url, payload_bytes, sig)

    # Dispatch in daemon thread so background worker is not stalled
    t = threading.Thread(
        target=_post_webhook,
        args=(callback_url, payload_bytes, sig),
        name=f"webhook-{job.id}",
        daemon=True,
    )
    t.start()
    return True
