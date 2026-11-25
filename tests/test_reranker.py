"""Tests for cross-modal relevance re-ranking."""

from app.services.reranker import (
    compute_entity_alignment,
    compute_identifier_match,
    compute_lexical_overlap,
    compute_provenance_richness,
    rerank_hits,
)


def test_lexical_overlap_scoring():
    query_tokens = ["checkout", "latency", "spike"]
    text1 = "The checkout service observed a latency spike after deployment."
    text2 = "Meeting discussing general roadmap and vacation plans."

    score1 = compute_lexical_overlap(query_tokens, text1)
    score2 = compute_lexical_overlap(query_tokens, text2)

    assert score1 > score2
    assert score1 > 0.5
    assert score2 == 0.0


def test_identifier_preservation():
    query_tokens = ["timeout=30s", "error_code_504"]
    text_with_id = "Configured timeout=30s to mitigate error_code_504 on gateway."
    text_without_id = "We saw some timeouts and errors."

    score_match = compute_identifier_match(query_tokens, text_with_id)
    score_no_match = compute_identifier_match(query_tokens, text_without_id)

    assert score_match > score_no_match
    assert score_match == 1.0


def test_entity_alignment():
    query_tokens = ["checkout", "payment"]
    entities = ["Checkout Service", "Payment Gateway"]
    score = compute_entity_alignment(query_tokens, entities)
    assert score > 0.5


def test_provenance_richness():
    hit_with_rich_provenance = {
        "timestamp": "01:23 - 01:45",
        "frame_path": "/frames/scene1.jpg",
        "matched_regions": [{"text": "Graph", "box": {"ymin": 0.1}}],
    }
    hit_bare = {"timestamp": None}

    score_rich = compute_provenance_richness(hit_with_rich_provenance)
    score_bare = compute_provenance_richness(hit_bare)

    assert score_rich > score_bare
    assert score_rich >= 0.8


def test_rerank_hits_ordering():
    query = "How did the team resolve checkout timeout 504?"
    hit_relevant = {
        "id": "seg_1",
        "score": 0.6,
        "transcript": "Resolved checkout timeout 504 using exponential backoff.",
        "timestamp": "02:15 - 02:30",
        "entities": ["Checkout Service"],
        "metadata": {"modality": "video"},
    }
    hit_generic = {
        "id": "seg_2",
        "score": 0.65,
        "transcript": "Meeting started at 10 AM with general announcements.",
        "timestamp": None,
        "entities": [],
        "metadata": {"modality": "audio"},
    }

    reranked = rerank_hits(query, [hit_generic, hit_relevant])
    assert len(reranked) == 2
    # The relevant hit should be boosted to rank 1 despite lower initial vector score
    assert reranked[0]["id"] == "seg_1"
    assert "rerank_score" in reranked[0]
    assert "rerank_breakdown" in reranked[0]
