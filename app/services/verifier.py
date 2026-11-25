"""Evidence verification and hallucination detection for synthesized answers.

Audits whether citations [1], [2] in an answer map to valid evidence sources,
and computes claim-level lexical entailment to detect ungrounded statements.
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas.knowledge import AnswerSource, VerificationResult, VerifiedClaim

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD_RE = re.compile(r"[a-z0-9_\-\.:]+")
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
    "you", "your",
}


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


def _extract_citation_indices(text: str) -> list[int]:
    return [int(m) for m in _CITATION_RE.findall(text)]


def _source_evidence_text(src: dict[str, Any] | AnswerSource) -> str:
    if isinstance(src, AnswerSource):
        return src.source or ""
    # Check snippet, text, transcript, content
    parts = [
        str(src.get("snippet") or ""),
        str(src.get("transcript") or ""),
        str(src.get("visual_summary") or ""),
        str(src.get("content") or ""),
        str(src.get("source") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def verify_answer(
    answer: str,
    sources: list[dict[str, Any]] | list[AnswerSource],
    hits: list[dict[str, Any]] | None = None,
) -> VerificationResult:
    """Audit citation mapping and factual grounding of claims against sources."""
    if not answer.strip():
        return VerificationResult(
            is_verified=False,
            faithfulness_score=0.0,
            citation_precision=0.0,
            total_claims=0,
            supported_claims=0,
            claims=[],
            hallucination_warnings=["Answer is empty"],
        )

    # Build source index lookup (1-indexed matching citation markers [1], [2])
    source_map: dict[int, str] = {}
    for i, s in enumerate(sources, start=1):
        citation_num = getattr(s, "evidence_index", None) if isinstance(s, AnswerSource) else s.get("citation")
        key = citation_num if citation_num is not None else i
        source_map[int(key)] = _source_evidence_text(s)

    # Augment with hits text if available
    if hits:
        for i, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            hit_text = " ".join(
                filter(
                    None,
                    [
                        str(hit.get("transcript") or metadata.get("transcript") or ""),
                        str(hit.get("visual_summary") or metadata.get("visual_summary") or ""),
                        str(metadata.get("content") or hit.get("document") or ""),
                    ],
                )
            )
            if i in source_map and hit_text:
                source_map[i] = f"{source_map[i]} {hit_text}".strip()
            elif i not in source_map and hit_text:
                source_map[i] = hit_text

    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    if not raw_sentences:
        raw_sentences = [answer.strip()]

    verified_claims: list[VerifiedClaim] = []
    hallucination_warnings: list[str] = []
    total_citations = 0
    valid_citations = 0

    for sent in raw_sentences:
        c_indices = _extract_citation_indices(sent)
        total_citations += len(c_indices)

        if not c_indices:
            # Factual sentence without citations
            verified_claims.append(
                VerifiedClaim(
                    sentence=sent,
                    citations=[],
                    status="uncited",
                    support_score=0.0,
                    evidence_snippet=None,
                )
            )
            continue

        sent_tokens = _tokenize(sent)
        matched_sources_text: list[str] = []
        for idx in c_indices:
            if idx in source_map:
                valid_citations += 1
                matched_sources_text.append(source_map[idx])
            else:
                hallucination_warnings.append(
                    f"Citation [{idx}] does not correspond to any retrieved evidence source."
                )

        combined_evidence = " ".join(matched_sources_text)
        if not combined_evidence:
            verified_claims.append(
                VerifiedClaim(
                    sentence=sent,
                    citations=c_indices,
                    status="unsupported",
                    support_score=0.0,
                    evidence_snippet=None,
                )
            )
            continue

        evidence_tokens = set(_tokenize(combined_evidence))
        if not sent_tokens:
            support_score = 1.0
        else:
            overlap = sum(1 for t in sent_tokens if t in evidence_tokens)
            support_score = float(overlap / len(sent_tokens))

        # Categorize grounding status
        if support_score >= 0.45:
            status = "supported"
        elif support_score >= 0.20:
            status = "partially_supported"
        else:
            status = "unsupported"
            hallucination_warnings.append(
                f"Low factual support ({round(support_score, 2)}) for claim: '{sent[:80]}...'"
            )

        snippet = combined_evidence[:200] + "..." if len(combined_evidence) > 200 else combined_evidence
        verified_claims.append(
            VerifiedClaim(
                sentence=sent,
                citations=c_indices,
                status=status,
                support_score=round(support_score, 4),
                evidence_snippet=snippet,
            )
        )

    supported_count = sum(1 for c in verified_claims if c.status in {"supported", "partially_supported"})
    total_claims = len(verified_claims)
    faithfulness = round(supported_count / total_claims, 4) if total_claims > 0 else 1.0
    precision = round(valid_citations / total_citations, 4) if total_citations > 0 else 1.0
    is_verified = bool(faithfulness >= 0.60 and precision == 1.0)

    return VerificationResult(
        is_verified=is_verified,
        faithfulness_score=faithfulness,
        citation_precision=precision,
        total_claims=total_claims,
        supported_claims=supported_count,
        claims=verified_claims,
        hallucination_warnings=hallucination_warnings,
    )
