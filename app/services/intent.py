"""Multimodal query intent classifier and modality routing.

Detects whether a user prompt seeks visual artifacts (diagrams, dashboards),
spoken dialogue (speaker attribution, quotes), documentary evidence (PDFs),
or structured telemetry (JSON logs), extracting targeted entities and speakers.
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas.knowledge import MediaModality, QueryIntentResponse

_VISUAL_CUES = {
    "diagram", "chart", "graph", "dashboard", "screenshot", "slide", "visual",
    "image", "picture", "frame", "ui", "screen", "architecture", "mockup",
}
_SPEECH_CUES = {
    "say", "said", "spoke", "spoken", "discuss", "discussed", "mention", "mentioned",
    "talk", "talked", "meeting", "call", "voice", "audio", "recording", "quote",
}
_DOC_CUES = {
    "report", "postmortem", "document", "doc", "pdf", "page", "section", "written",
    "guide", "manual", "policy", "specification", "spec",
}
_STRUCTURED_CUES = {
    "json", "log", "logs", "telemetry", "metric", "ticket", "event", "error code",
    "http", "payload", "record", "database row",
}
_TEMPORAL_CUES = {
    "first", "after", "before", "then", "timeline", "chronological", "sequence",
    "order", "when", "start", "end", "finally", "subsequently",
}

_WORD_RE = re.compile(r"[a-z0-9_\-\.:]+")
_CAP_WORDS_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_KNOWN_SPEAKERS = {"Alex", "Sarah", "Yug", "David", "Elena", "Michael", "Chris", "Jordan"}


def classify_intent(query: str) -> QueryIntentResponse:
    """Classify the user's information need and detect target modalities and entities."""
    q_lower = query.lower()
    words = set(_WORD_RE.findall(q_lower))

    visual_score = sum(1 for w in _VISUAL_CUES if w in words or w in q_lower)
    speech_score = sum(1 for w in _SPEECH_CUES if w in words or w in q_lower)
    doc_score = sum(1 for w in _DOC_CUES if w in words or w in q_lower)
    struct_score = sum(1 for w in _STRUCTURED_CUES if w in words or w in q_lower)
    temporal_score = sum(1 for w in _TEMPORAL_CUES if w in words or w in q_lower)

    # Detect speakers
    detected_speakers = []
    for spk in _KNOWN_SPEAKERS:
        if re.search(r"\b" + re.escape(spk) + r"\b", query, re.IGNORECASE):
            detected_speakers.append(spk)

    # Detect potential entity proper nouns
    caps = _CAP_WORDS_RE.findall(query)
    detected_entities = [c for c in caps if c not in _KNOWN_SPEAKERS and len(c) > 3][:5]

    suggested_modalities: list[str] = []
    primary_intent = "general_multimodal"
    confidence = 0.5

    scores = {
        "visual_inspection": visual_score,
        "speaker_inquiry": speech_score + (2 if detected_speakers else 0),
        "document_lookup": doc_score,
        "log_investigation": struct_score,
    }
    best_intent, best_val = max(scores.items(), key=lambda item: item[1])

    if best_val > 0:
        primary_intent = best_intent
        confidence = min(0.95, 0.6 + best_val * 0.1)

        if best_intent == "visual_inspection":
            suggested_modalities = [MediaModality.IMAGE.value, MediaModality.VIDEO.value, MediaModality.PDF.value]
        elif best_intent == "speaker_inquiry":
            suggested_modalities = [MediaModality.VIDEO.value, MediaModality.AUDIO.value]
        elif best_intent == "document_lookup":
            suggested_modalities = [MediaModality.PDF.value]
        elif best_intent == "log_investigation":
            suggested_modalities = [MediaModality.JSON.value]
    else:
        if temporal_score > 0:
            primary_intent = "chronological_sequence"
            confidence = 0.75
        suggested_modalities = [
            MediaModality.VIDEO.value,
            MediaModality.PDF.value,
            MediaModality.IMAGE.value,
            MediaModality.AUDIO.value,
            MediaModality.JSON.value,
        ]

    return QueryIntentResponse(
        query=query,
        primary_intent=primary_intent,
        suggested_modalities=suggested_modalities,
        detected_speakers=detected_speakers,
        detected_entities=detected_entities,
        temporal_focus=bool(temporal_score > 0),
        confidence=round(confidence, 2),
    )
