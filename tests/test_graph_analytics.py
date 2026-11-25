"""Tests for graph analytics, PageRank centrality, and cross-modal bridge detection."""

from fastapi.testclient import TestClient
import pytest

from app.schemas.knowledge import MediaModality
from app.schemas.records import (
    EntityRecord,
    NodeKind,
    RelationRecord,
    RelationType,
    SegmentKind,
    SegmentRecord,
    SourceRecord,
)
from app.services.graph import compute_graph_analytics
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)


def test_empty_graph_analytics(repo):
    analytics = compute_graph_analytics(repo)
    assert analytics.total_nodes == 0
    assert analytics.total_edges == 0
    assert analytics.density == 0.0
    assert analytics.top_entities_by_pagerank == []
    assert analytics.cross_modal_bridges == []


def test_graph_analytics_with_cross_modal_entities(repo):
    # Create two sources with different modalities
    s1 = SourceRecord(id="src_video", filename="meeting.mp4", modality=MediaModality.VIDEO)
    s2 = SourceRecord(id="src_pdf", filename="report.pdf", modality=MediaModality.PDF)
    repo.upsert_source(s1)
    repo.upsert_source(s2)

    # Segments
    seg1 = SegmentRecord(
        id="seg_1",
        source_id="src_video",
        modality=MediaModality.VIDEO,
        kind=SegmentKind.AV_WINDOW,
        ordinal=0,
        confidence=1.0,
        extractor="test",
    )
    seg2 = SegmentRecord(
        id="seg_2",
        source_id="src_pdf",
        modality=MediaModality.PDF,
        kind=SegmentKind.PAGE,
        ordinal=0,
        confidence=1.0,
        extractor="test",
    )
    repo.add_segments([seg1, seg2])

    # Entity mentioned across both modalities
    inserted_ent = repo.upsert_entity("Checkout Service", "service")
    assert inserted_ent is not None
    ent_id = inserted_ent.id

    # Relations: seg1 -> ent_checkout, seg2 -> ent_checkout, seg1 -(depicts)-> seg2
    r1 = RelationRecord(
        id="r1", src_kind=NodeKind.SEGMENT, src_id="seg_1", dst_kind=NodeKind.ENTITY, dst_id=ent_id, relation=RelationType.MENTIONS
    )
    r2 = RelationRecord(
        id="r2", src_kind=NodeKind.SEGMENT, src_id="seg_2", dst_kind=NodeKind.ENTITY, dst_id=ent_id, relation=RelationType.MENTIONS
    )
    r3 = RelationRecord(
        id="r3", src_kind=NodeKind.SEGMENT, src_id="seg_1", dst_kind=NodeKind.SEGMENT, dst_id="seg_2", relation=RelationType.DEPICTS
    )
    repo.add_relations([r1, r2, r3])

    analytics = compute_graph_analytics(repo)
    assert analytics.total_nodes == 3
    assert analytics.total_edges == 3
    assert analytics.density > 0.0
    assert analytics.modality_distribution.get("video") == 1
    assert analytics.modality_distribution.get("pdf") == 1

    # Check PageRank top entities
    assert len(analytics.top_entities_by_pagerank) > 0
    top = analytics.top_entities_by_pagerank[0]
    assert top.entity_id == ent_id
    assert top.name == "Checkout Service"

    # Check cross modal bridges
    assert len(analytics.cross_modal_bridges) == 1
    bridge = analytics.cross_modal_bridges[0]
    assert bridge.id == ent_id
    assert set(bridge.modalities) == {"video", "pdf"}
    assert bridge.source_count == 2


def test_api_graph_analytics_endpoint(client):
    resp = client.get("/graph/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_nodes" in data
    assert "total_edges" in data
    assert "density" in data
    assert "top_entities_by_pagerank" in data
    assert "cross_modal_bridges" in data
