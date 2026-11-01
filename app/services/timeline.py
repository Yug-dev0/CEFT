"""Unified Chronological Timeline Service for Ceft.

Aggregates spatiotemporal evidence across videos, audio recordings,
PDFs, screenshots, and JSON logs into an interactive chronological stream.
"""

from __future__ import annotations

from typing import Any

from app.db.repository import KnowledgeRepository, get_repository


def get_timeline_events(
    repository: KnowledgeRepository | None = None,
    *,
    limit: int = 100,
    source_id: str | None = None,
    modality: str | None = None,
) -> list[dict[str, Any]]:
    """Build a sorted timeline of multimodal evidence across the knowledge base."""
    repo = repository or get_repository()
    sources = {s.source.id: s.source for s in repo.list_sources()}

    if source_id and source_id in sources:
        target_sources = [sources[source_id]]
    else:
        target_sources = list(sources.values())

    events: list[dict[str, Any]] = []

    for src in target_sources:
        if modality and src.modality.value != modality:
            continue

        segments = repo.list_segments(src.id)
        for seg in segments:
            loc = seg.locator
            has_time = (
                loc.start_seconds is not None
                or loc.page_number is not None
                or loc.label is not None
            )

            # Format human-friendly time badge
            time_display = None
            sort_key = 0.0

            if loc.start_seconds is not None:
                start_m, start_s = divmod(int(loc.start_seconds), 60)
                if loc.end_seconds is not None:
                    end_m, end_s = divmod(int(loc.end_seconds), 60)
                    time_display = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
                else:
                    time_display = f"{start_m:02d}:{start_s:02d}"
                sort_key = loc.start_seconds
            elif loc.page_number is not None:
                time_display = f"Page {loc.page_number}"
                sort_key = 10000.0 + loc.page_number
            elif loc.label is not None:
                time_display = loc.label
                sort_key = 20000.0 + seg.ordinal
            else:
                time_display = f"Segment #{seg.ordinal + 1}"
                sort_key = 30000.0 + seg.ordinal

            # Associated date if specified on source or segment
            recorded_at = (
                seg.attributes.get("timestamp")
                or seg.attributes.get("created_at")
                or src.attributes.get("recorded_at")
                or src.ingested_at.isoformat()
            )

            snippet = seg.text or seg.visual_summary or ""
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."

            # Get linked relations
            relations = [
                {
                    "relation": r.relation.value,
                    "direction": d,
                    "other_id": r.dst_id if d == "out" else r.src_id,
                }
                for r, d in repo.relations_for(seg.id)
            ]

            events.append({
                "id": seg.id,
                "source_id": src.id,
                "source": src.filename,
                "modality": seg.modality.value,
                "time_display": time_display,
                "sort_key": sort_key,
                "recorded_at": recorded_at,
                "ordinal": seg.ordinal,
                "snippet": snippet,
                "visual_summary": seg.visual_summary,
                "frame_path": seg.frame_path,
                "confidence": seg.confidence,
                "entities": [
                    e.name for e in repo.all_entities()
                    if e.id in {m[1] for m in repo.mentions_of() if m[0] == seg.id}
                ] if False else [],  # fast path below
                "speakers": seg.attributes.get("speakers") or [],
                "relation_count": len(relations),
                "relations": relations[:3],
            })

    # Hydrate entities efficiently
    if events:
        seg_ids = [e["id"] for e in events]
        mentions = repo.mentions_of()
        ent_map = {e.id: e.name for e in repo.all_entities()}
        seg_entities: dict[str, list[str]] = {}
        for sid, eid in mentions:
            if sid in set(seg_ids) and eid in ent_map:
                seg_entities.setdefault(sid, []).append(ent_map[eid])
        for e in events:
            e["entities"] = seg_entities.get(e["id"], [])

    # Sort events by source and time
    events.sort(key=lambda x: (x["source"], x["sort_key"]))
    return events[:limit]

