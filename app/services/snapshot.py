"""Knowledge base snapshot export and import archive engine.

Exports structured SQLite records (sources, segments, entities, relations)
into a portable JSON archive, and imports snapshots with vector index rebuilding.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from app.db.repository import KnowledgeRepository
from app.schemas.records import (
    EntityRecord,
    RelationRecord,
    RelationType,
    SegmentRecord,
    SourceRecord,
)


def export_knowledge_snapshot(repo: KnowledgeRepository) -> dict[str, Any]:
    """Export complete system-of-record knowledge into a portable JSON structure."""
    sources = repo.list_sources()
    segments = [seg for batch in repo.iter_all_segments() for seg in batch]
    entities = repo.all_entities()
    relations = repo.list_relations(list(RelationType))

    return {
        "version": "0.3.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "sources": len(sources),
            "segments": len(segments),
            "entities": len(entities),
            "relations": len(relations),
        },
        "sources": [s.source.model_dump(mode="json") for s in sources],
        "segments": [s.model_dump(mode="json") for s in segments],
        "entities": [e.model_dump(mode="json") for e in entities],
        "relations": [r.model_dump(mode="json") for r in relations],
    }


def import_knowledge_snapshot(
    snapshot: dict[str, Any],
    repo: KnowledgeRepository,
    *,
    rebuild_vector_index: bool = True,
) -> dict[str, int]:
    """Import snapshot data into SQLite and optionally rebuild Chroma embeddings."""
    raw_sources = snapshot.get("sources") or []
    raw_segments = snapshot.get("segments") or []
    raw_entities = snapshot.get("entities") or []
    raw_relations = snapshot.get("relations") or []

    # 1. Upsert sources
    imported_sources = 0
    for s_dict in raw_sources:
        try:
            repo.upsert_source(SourceRecord(**s_dict))
            imported_sources += 1
        except Exception:
            pass

    # 2. Add segments
    imported_segments = 0
    segment_objs = []
    for seg_dict in raw_segments:
        try:
            segment_objs.append(SegmentRecord(**seg_dict))
        except Exception:
            pass
    if segment_objs:
        repo.add_segments(segment_objs)
        imported_segments = len(segment_objs)

    # 3. Upsert entities
    imported_entities = 0
    for e_dict in raw_entities:
        try:
            repo.upsert_entity(e_dict.get("name", ""), e_dict.get("entity_type", "concept"))
            imported_entities += 1
        except Exception:
            pass

    # 4. Add relations
    imported_relations = 0
    relation_objs = []
    for r_dict in raw_relations:
        try:
            relation_objs.append(RelationRecord(**r_dict))
        except Exception:
            pass
    if relation_objs:
        repo.add_relations(relation_objs)
        imported_relations = len(relation_objs)

    # 5. Rebuild vector index if requested
    if rebuild_vector_index and imported_segments > 0:
        from app.services.ingestion import rebuild_index

        try:
            rebuild_index(repository=repo)
        except Exception:
            pass

    return {
        "sources": imported_sources,
        "segments": imported_segments,
        "entities": imported_entities,
        "relations": imported_relations,
    }
