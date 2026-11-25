"""Semantic chapter generation for temporal (video, audio) and document sources."""

from __future__ import annotations

from typing import Any

from app.db.repository import KnowledgeRepository
from app.schemas.records import (
    SegmentRecord,
    SourceChapter,
    SourceChaptersResponse,
    SourceRecord,
    format_seconds,
)


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "00:00"
    return format_seconds(seconds)


def generate_source_chapters(
    repo: KnowledgeRepository,
    source_id: str,
    max_chapter_duration: float = 300.0,
) -> SourceChaptersResponse | None:
    source = repo.get_source(source_id)
    if source is None:
        return None

    segments = repo.list_segments(source_id)
    if not segments:
        return SourceChaptersResponse(
            source_id=source.id,
            filename=source.filename,
            modality=source.modality.value,
            total_chapters=0,
            chapters=[],
        )

    # Sort segments chronologically or by ordinal
    sorted_segs = sorted(
        segments,
        key=lambda s: (
            s.locator.start_seconds if s.locator.start_seconds is not None else float(s.ordinal),
            s.ordinal,
        ),
    )

    # Group segments into chapters
    chapter_groups: list[list[SegmentRecord]] = []
    current_group: list[SegmentRecord] = []
    group_start_time = sorted_segs[0].locator.start_seconds or 0.0

    for seg in sorted_segs:
        curr_start = seg.locator.start_seconds or 0.0
        if not current_group:
            current_group.append(seg)
            group_start_time = curr_start
            continue

        duration = curr_start - group_start_time
        prev_end = current_group[-1].locator.end_seconds or current_group[-1].locator.start_seconds or 0.0
        gap = curr_start - prev_end

        if (duration >= max_chapter_duration) or (gap >= 60.0 and len(current_group) >= 2):
            chapter_groups.append(current_group)
            current_group = [seg]
            group_start_time = curr_start
        else:
            current_group.append(seg)

    if current_group:
        chapter_groups.append(current_group)

    # Fetch entity mentions for all segments
    all_seg_ids = [s.id for s in sorted_segs]
    mentions_map = repo.entities_for_segments(all_seg_ids)

    chapters: list[SourceChapter] = []
    for idx, group in enumerate(chapter_groups, start=1):
        seg_ids = [s.id for s in group]
        first_seg = group[0]
        last_seg = group[-1]

        start_sec = first_seg.locator.start_seconds or 0.0
        end_sec = last_seg.locator.end_seconds or (start_sec + 5.0)
        duration = max(0.0, end_sec - start_sec)

        group_entities = set()
        group_speakers = set()
        keyframe_path = None

        texts = []
        for s in group:
            if s.text:
                texts.append(s.text.strip())
            elif s.visual_summary:
                texts.append(s.visual_summary.strip())
            if not keyframe_path and s.frame_path:
                keyframe_path = s.frame_path
            for ent, _ in mentions_map.get(s.id, []):
                group_entities.add(ent.name)
            for spk in s.attributes.get("speakers") or []:
                name = spk.get("name") or spk.get("label")
                if name:
                    group_speakers.add(name)

        combined_summary = " ".join(texts)
        if len(combined_summary) > 280:
            summary = combined_summary[:277] + "..."
        else:
            summary = combined_summary or f"Chapter {idx} evidence segment."

        # Generate descriptive title
        if group_entities:
            top_ents = sorted(group_entities)[:2]
            title = f"Discussion of {', '.join(top_ents)}"
        elif group_speakers:
            title = f"Remarks by {', '.join(sorted(group_speakers)[:2])}"
        elif first_seg.visual_summary:
            title = first_seg.visual_summary[:50]
        elif first_seg.locator.page_number:
            title = f"Pages {first_seg.locator.page_number}-{last_seg.locator.page_number or first_seg.locator.page_number}"
        else:
            title = f"Chapter {idx}: {_format_time(start_sec)} - {_format_time(end_sec)}"

        chapters.append(
            SourceChapter(
                chapter_index=idx,
                title=title,
                start_seconds=round(start_sec, 2),
                end_seconds=round(end_sec, 2),
                start_time_str=_format_time(start_sec),
                end_time_str=_format_time(end_sec),
                duration_seconds=round(duration, 2),
                summary=summary,
                keyframe_path=keyframe_path,
                entities=sorted(group_entities),
                speakers=sorted(group_speakers),
                segment_ids=seg_ids,
            )
        )

    return SourceChaptersResponse(
        source_id=source.id,
        filename=source.filename,
        modality=source.modality.value,
        total_chapters=len(chapters),
        chapters=chapters,
    )
