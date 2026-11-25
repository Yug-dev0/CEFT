"""Cross-modal relevance re-ranking for multimodal evidence candidates.

Combines lexical overlap, exact identifier preservation, entity alignment,
provenance precision, and cross-modal corroboration to produce calibrated,
explainable relevance scores.
"""

from __future__ import annotations

import math
import re
from typing import Any

from app.db.repository import normalize_entity_name

_WORD = re.compile(r"[a-z0-9_\-\.:]+")
_EXACT_IDENTIFIER = re.compile(r"^[a-z0-9_\-\.:]{3,}$")
_STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "during", "each", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "know", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "show", "shown", "said", "tell", "explain",
}


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


def compute_lexical_overlap(query_tokens: list[str], document_text: str) -> float:
    """Compute normalized token coverage and frequency density."""
    if not query_tokens or not document_text:
        return 0.0
    doc_tokens = _tokenize(document_text)
    if not doc_tokens:
        return 0.0
    doc_token_set = set(doc_tokens)
    matched = sum(1 for qt in set(query_tokens) if qt in doc_token_set)
    coverage = matched / len(set(query_tokens))

    # Frequency bonus for repeated relevant terms
    freq_hits = sum(min(doc_tokens.count(qt), 3) for qt in set(query_tokens))
    density = min(1.0, freq_hits / (math.sqrt(len(doc_tokens)) + 1.0))
    return float(0.6 * coverage + 0.4 * density)


def compute_identifier_match(query_tokens: list[str], document_text: str) -> float:
    """Reward preservation of technical tokens like numbers, error codes, service names."""
    doc_lower = document_text.lower()
    tech_tokens = [
        t for t in query_tokens
        if (any(c.isdigit() for c in t) or "_" in t or "-" in t or "." in t or len(t) >= 6)
    ]
    if not tech_tokens:
        return 0.5  # Neutral when query has no strict technical identifiers
    matched_tech = sum(1 for t in tech_tokens if t in doc_lower)
    return float(matched_tech / len(tech_tokens))


def compute_entity_alignment(query_tokens: list[str], hit_entities: list[str]) -> float:
    """Measure alignment between query keywords and segment entity mentions."""
    if not hit_entities:
        return 0.0
    query_set = set(query_tokens)
    matched = 0
    for entity in hit_entities:
        norm = normalize_entity_name(str(entity))
        entity_tokens = set(_tokenize(norm))
        if entity_tokens & query_set:
            matched += 1
    return min(1.0, matched / max(1, len(hit_entities)))


def compute_provenance_richness(hit: dict[str, Any]) -> float:
    """Score the verifiable specificity of the evidence locator."""
    score = 0.0
    metadata = hit.get("metadata") or {}
    locator = str(hit.get("timestamp") or metadata.get("timestamp") or "")

    # Temporal range or exact page
    if ":" in locator or "page" in locator.lower():
        score += 0.5
    elif locator:
        score += 0.2

    # Visual bounding box or frame path
    if hit.get("frame_path") or metadata.get("frame_path"):
        score += 0.3
    if hit.get("matched_regions"):
        score += 0.2

    return min(1.0, score)


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    subqueries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rerank candidate hits using multimodal lexical, entity, and provenance cross-scoring.

    Each hit is enriched with:
    - ``rerank_score``: Calibrated score in [0.0, 1.0].
    - ``rerank_breakdown``: Sub-scores for auditability and inspection.
    """
    if not hits:
        return []

    combined_query_text = f"{query} " + " ".join(subqueries or [])
    query_tokens = _tokenize(combined_query_text)

    scored_hits = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        text_corpus = " ".join(
            filter(
                None,
                [
                    str(hit.get("transcript") or metadata.get("transcript") or ""),
                    str(hit.get("visual_summary") or metadata.get("visual_summary") or ""),
                    str(metadata.get("content") or hit.get("document") or ""),
                    " ".join(str(e) for e in (hit.get("entities") or metadata.get("entities") or [])),
                    " ".join(str(s) for s in (hit.get("speakers") or metadata.get("speakers") or [])),
                ],
            )
        )

        base_score = float(hit.get("score") or hit.get("similarity_score") or 0.5)
        lexical_score = compute_lexical_overlap(query_tokens, text_corpus)
        id_score = compute_identifier_match(query_tokens, text_corpus)
        entity_score = compute_entity_alignment(
            query_tokens, list(hit.get("entities") or metadata.get("entities") or [])
        )
        prov_score = compute_provenance_richness(hit)
        graph_bonus = 0.1 if hit.get("via") or (hit.get("signals") or {}).get("graph_boost") else 0.0

        # Fused reranking formula
        final_score = (
            0.35 * base_score
            + 0.25 * lexical_score
            + 0.15 * id_score
            + 0.15 * entity_score
            + 0.10 * prov_score
            + graph_bonus
        )
        final_score = min(1.0, max(0.0, final_score))

        hit_copy = dict(hit)
        hit_copy["rerank_score"] = round(final_score, 4)
        hit_copy["rerank_breakdown"] = {
            "base_score": round(base_score, 4),
            "lexical_overlap": round(lexical_score, 4),
            "identifier_match": round(id_score, 4),
            "entity_alignment": round(entity_score, 4),
            "provenance_richness": round(prov_score, 4),
            "graph_bonus": round(graph_bonus, 4),
        }
        scored_hits.append(hit_copy)

    # Sort descending by re-ranked score
    scored_hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return scored_hits
