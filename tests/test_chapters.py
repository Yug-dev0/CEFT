"""Tests for automated video and audio incident chaptering."""

from fastapi.testclient import TestClient
import pytest

from app.schemas.knowledge import MediaModality
from app.schemas.records import Locator, SegmentKind, SegmentRecord, SourceRecord
from app.services.chapters import generate_source_chapters
from main import app


@pytest.fixture()
def client(repo, store, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ROOT", str(tmp_path / "media"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return TestClient(app)


def test_generate_chapters_for_temporal_video(repo):
    source = SourceRecord(id="src_meeting_video", filename="meeting.mp4", modality=MediaModality.VIDEO)
    repo.upsert_source(source)

    # Add 4 segments: 2 in first chapter (0-30s), 2 after gap in second chapter (100-130s)
    seg1 = SegmentRecord(
        id="s1",
        source_id="src_meeting_video",
        modality=MediaModality.VIDEO,
        kind=SegmentKind.AV_WINDOW,
        locator=Locator(start_seconds=0.0, end_seconds=15.0),
        text="Alex introduced the checkout incident overview.",
        frame_path="/frames/s1.jpg",
        attributes={"speakers": [{"name": "Alex"}]},
        ordinal=0,
        confidence=0.95,
        extractor="test",
    )
    seg2 = SegmentRecord(
        id="s2",
        source_id="src_meeting_video",
        modality=MediaModality.VIDEO,
        kind=SegmentKind.AV_WINDOW,
        locator=Locator(start_seconds=15.0, end_seconds=30.0),
        text="Alex walked through latency graph spikes.",
        frame_path="/frames/s2.jpg",
        attributes={"speakers": [{"name": "Alex"}]},
        ordinal=1,
        confidence=0.95,
        extractor="test",
    )
    # Gap > 60s -> starts new chapter
    seg3 = SegmentRecord(
        id="s3",
        source_id="src_meeting_video",
        modality=MediaModality.VIDEO,
        kind=SegmentKind.AV_WINDOW,
        locator=Locator(start_seconds=120.0, end_seconds=135.0),
        text="Sarah explained database connection pool limits.",
        frame_path="/frames/s3.jpg",
        attributes={"speakers": [{"name": "Sarah"}]},
        ordinal=2,
        confidence=0.95,
        extractor="test",
    )
    seg4 = SegmentRecord(
        id="s4",
        source_id="src_meeting_video",
        modality=MediaModality.VIDEO,
        kind=SegmentKind.AV_WINDOW,
        locator=Locator(start_seconds=135.0, end_seconds=150.0),
        text="Sarah proposed exponential backoff retry rollout.",
        frame_path="/frames/s4.jpg",
        attributes={"speakers": [{"name": "Sarah"}]},
        ordinal=3,
        confidence=0.95,
        extractor="test",
    )
    repo.add_segments([seg1, seg2, seg3, seg4])

    resp = generate_source_chapters(repo, "src_meeting_video")
    assert resp is not None
    assert resp.total_chapters == 2
    assert resp.chapters[0].chapter_index == 1
    assert resp.chapters[0].start_seconds == 0.0
    assert resp.chapters[0].end_seconds == 30.0
    assert "Alex" in resp.chapters[0].speakers
    assert resp.chapters[0].keyframe_path == "/frames/s1.jpg"

    assert resp.chapters[1].chapter_index == 2
    assert resp.chapters[1].start_seconds == 120.0
    assert resp.chapters[1].end_seconds == 150.0
    assert "Sarah" in resp.chapters[1].speakers


def test_api_chapters_endpoint(client, repo):
    source = SourceRecord(id="src_audio_memo", filename="memo.mp3", modality=MediaModality.AUDIO)
    repo.upsert_source(source)
    seg = SegmentRecord(
        id="s_memo",
        source_id="src_audio_memo",
        modality=MediaModality.AUDIO,
        kind=SegmentKind.SPEECH,
        locator=Locator(start_seconds=0.0, end_seconds=10.0),
        text="Status update memo.",
        ordinal=0,
        confidence=1.0,
        extractor="test",
    )
    repo.add_segments([seg])

    resp = client.get("/sources/src_audio_memo/chapters")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_id"] == "src_audio_memo"
    assert data["total_chapters"] == 1
    assert len(data["chapters"]) == 1
