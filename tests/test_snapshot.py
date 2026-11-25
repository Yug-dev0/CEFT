"""Tests for knowledge base snapshot export and import archive engine."""

from fastapi.testclient import TestClient
import pytest

from app.schemas.knowledge import MediaModality
from app.schemas.records import Locator, SegmentKind, SegmentRecord, SourceRecord
from app.services.snapshot import export_knowledge_snapshot, import_knowledge_snapshot
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)


def test_export_and_import_snapshot(repo):
    # Setup some test data
    source = SourceRecord(id="src_snap_test", filename="snapshot_doc.pdf", modality=MediaModality.PDF)
    repo.upsert_source(source)
    seg = SegmentRecord(
        id="seg_snap_1",
        source_id="src_snap_test",
        modality=MediaModality.PDF,
        kind=SegmentKind.PAGE,
        locator=Locator(page_number=1),
        text="Snapshot test page content.",
        ordinal=0,
        confidence=1.0,
        extractor="test",
    )
    repo.add_segments([seg])
    repo.upsert_entity("Snapshot Entity", "concept")

    # Export
    snapshot = export_knowledge_snapshot(repo)
    assert snapshot["version"] == "0.3.0"
    assert snapshot["stats"]["sources"] >= 1
    assert snapshot["stats"]["segments"] >= 1
    assert snapshot["stats"]["entities"] >= 1

    # Clear or test re-import
    res = import_knowledge_snapshot(snapshot, repo, rebuild_vector_index=False)
    assert res["sources"] >= 1
    assert res["segments"] >= 1


def test_api_snapshot_endpoints(client):
    export_resp = client.get("/knowledge/export")
    assert export_resp.status_code == 200
    data = export_resp.json()
    assert "version" in data
    assert "stats" in data
    assert "sources" in data

    import_resp = client.post("/knowledge/import", json=data)
    assert import_resp.status_code == 200
    res_data = import_resp.json()
    assert "sources" in res_data
    assert "segments" in res_data
