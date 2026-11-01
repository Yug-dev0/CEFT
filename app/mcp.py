"""Model Context Protocol (MCP) stdio server for Ceft.

Exposes Ceft's multimodal knowledge base to AI agents (Claude Desktop, Cursor,
Antigravity, and other MCP clients) over standard JSON-RPC 2.0 stdio transport.

Supported tools:
- ``ceft_search``: Hybrid and graph-aware search over multimodal evidence.
- ``ceft_ask``: Synthesizes an answer with verifiable, byte-exact citations.
- ``ceft_get_evidence``: Full details of a specific segment (locator, text, OCR).
- ``ceft_get_entity``: Entity info, aliases, and cross-modal timeline.
- ``ceft_list_sources``: Lists indexed files and modalities.
- ``ceft_get_timeline``: Chronological event timeline across all files.

Usage:
    python -m app.mcp
    python -m app.cli mcp
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from app.db.repository import get_repository
from app.services import retrieval
from app.services.answer_synthesizer import synthesize_answer

_log = logging.getLogger("ceft.mcp")

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ceft"
SERVER_VERSION = "0.3.0"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "ceft_search",
        "description": (
            "Search Ceft's multimodal knowledge base (video transcripts, keyframe visuals, "
            "PDF pages, screenshots, and JSON logs). Returns evidence ranked by hybrid fusion "
            "with exact locators (timestamps, page numbers, bounding boxes)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or search query.",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of evidence segments to return (1-20).",
                },
                "modalities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional filter by modality: 'video', 'audio', 'image', 'pdf', 'json'.",
                },
                "hybrid": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to fuse dense embedding search with BM25 keyword matching and entity matching.",
                },
                "expand": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to expand along the knowledge graph to pull in corroborating evidence.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ceft_ask",
        "description": (
            "Ask a question and receive a grounded, synthesized answer with numbered citations "
            "pointing to the exact second in video, page in PDF, or region of an image."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question to ask the knowledge base.",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of evidence segments to retrieve for synthesis.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ceft_get_evidence",
        "description": (
            "Retrieve full details of a specific piece of evidence by segment ID: "
            "transcript, visual summary, OCR text blocks, bounding boxes, and graph relations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "The unique segment ID.",
                },
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "ceft_get_entity",
        "description": (
            "Lookup an entity (person, system, component, metric) by name to inspect "
            "its aliases, mentions across files, and cross-modal timeline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The entity name or alias.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "ceft_list_sources",
        "description": (
            "List all ingested source files in Ceft's knowledge repository, including "
            "filename, modality, segment count, and status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ceft_get_timeline",
        "description": (
            "Get a chronological event timeline across all indexed files (video/audio timestamps, "
            "JSON log timestamps, and document dates)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum number of timeline entries.",
                },
            },
        },
    },
]


def _handle_ceft_search(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"error": "query cannot be empty"})
    limit = max(1, min(20, int(args.get("limit", 5))))
    modalities = args.get("modalities")
    hybrid = bool(args.get("hybrid", True))
    expand = bool(args.get("expand", True))

    res = retrieval.retrieve(
        query=query,
        limit=limit,
        hybrid=hybrid,
        expand=expand,
        modalities=modalities,
    )
    formatted = []
    for rank, h in enumerate(res.hits, start=1):
        formatted.append({
            "rank": rank,
            "segment_id": h.get("id"),
            "source": h.get("source"),
            "modality": (h.get("metadata") or {}).get("modality"),
            "locator": h.get("timestamp"),
            "score": h.get("score"),
            "transcript": h.get("transcript"),
            "visual_summary": h.get("visual_summary"),
            "entities": h.get("entities", []),
            "speakers": h.get("speakers", []),
            "via": h.get("via"),
        })
    return json.dumps({
        "query": res.query,
        "subqueries": res.subqueries,
        "results_count": len(formatted),
        "hits": formatted,
    }, indent=2)


def _handle_ceft_ask(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"error": "query cannot be empty"})
    limit = max(1, min(15, int(args.get("limit", 5))))

    res = retrieval.retrieve(query=query, limit=limit)
    answer = synthesize_answer(query, res.hits)
    return json.dumps({
        "query": query,
        "answer": answer.answer,
        "grounded": answer.grounded,
        "method": answer.method,
        "sources": answer.sources,
    }, indent=2)


def _handle_ceft_get_evidence(args: dict[str, Any]) -> str:
    segment_id = str(args.get("segment_id", "")).strip()
    if not segment_id:
        return json.dumps({"error": "segment_id cannot be empty"})
    repo = get_repository()
    seg = repo.get_segment(segment_id)
    if seg is None:
        return json.dumps({"error": f"Segment not found: {segment_id}"})
    src = repo.get_source(seg.source_id)
    relations = [
        {
            "relation": r.relation.value,
            "direction": d,
            "other_id": r.dst_id if d == "out" else r.src_id,
            "confidence": r.confidence,
        }
        for r, d in repo.relations_for(segment_id)
    ]
    return json.dumps({
        "segment_id": seg.id,
        "source": src.filename if src else "unknown",
        "modality": seg.modality.value,
        "locator": seg.locator.model_dump(),
        "text": seg.text,
        "visual_summary": seg.visual_summary,
        "confidence": seg.confidence,
        "frame_path": seg.frame_path,
        "attributes": seg.attributes,
        "relations": relations,
    }, indent=2)


def _handle_ceft_get_entity(args: dict[str, Any]) -> str:
    name = str(args.get("name", "")).strip()
    if not name:
        return json.dumps({"error": "name cannot be empty"})
    repo = get_repository()
    normalized = name.lower()
    entities = [e for e in repo.all_entities() if e.normalized_name == normalized or normalized in e.normalized_name]
    if not entities:
        return json.dumps({"error": f"Entity not found: {name}"})
    ent = entities[0]
    timeline = repo.entity_timeline(ent.id)
    return json.dumps({
        "entity_id": ent.id,
        "name": ent.name,
        "type": ent.entity_type.value if hasattr(ent.entity_type, "value") else str(ent.entity_type),
        "timeline_mentions_count": len(timeline),
        "timeline": [t.model_dump() for t in timeline],
    }, indent=2)


def _handle_ceft_list_sources(_args: dict[str, Any]) -> str:
    repo = get_repository()
    sources = repo.list_sources()
    return json.dumps([
        {
            "source_id": s.source.id,
            "filename": s.source.filename,
            "modality": s.source.modality.value,
            "segment_count": s.segment_count,
            "status": s.source.status.value,
            "ingested_at": s.source.ingested_at.isoformat(),
        }
        for s in sources
    ], indent=2)


def _handle_ceft_get_timeline(args: dict[str, Any]) -> str:
    limit = max(1, min(200, int(args.get("limit", 50))))
    repo = get_repository()
    sources = {s.source.id: s.source for s in repo.list_sources()}
    events = []

    for src_id, src in sources.items():
        segments = repo.list_segments(src_id)
        for seg in segments:
            loc = seg.locator
            has_time = loc.start_seconds is not None or loc.page_number is not None or loc.label is not None
            if not has_time and not seg.text and not seg.visual_summary:
                continue

            # Format human readable time
            time_display = None
            if loc.start_seconds is not None:
                start_m, start_s = divmod(int(loc.start_seconds), 60)
                if loc.end_seconds is not None:
                    end_m, end_s = divmod(int(loc.end_seconds), 60)
                    time_display = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
                else:
                    time_display = f"{start_m:02d}:{start_s:02d}"
            elif loc.page_number is not None:
                time_display = f"Page {loc.page_number}"
            elif loc.label is not None:
                time_display = loc.label

            # Check if JSON segment contains embedded timestamp attribute
            created_at = seg.attributes.get("timestamp") or seg.attributes.get("created_at") or src.ingested_at.isoformat()

            events.append({
                "segment_id": seg.id,
                "source": src.filename,
                "modality": seg.modality.value,
                "time_display": time_display,
                "start_seconds": loc.start_seconds,
                "created_at": created_at,
                "snippet": (seg.text or seg.visual_summary or "")[:160],
                "speakers": seg.attributes.get("speakers", []),
            })

    # Sort events by time
    events.sort(key=lambda e: (e.get("start_seconds") is None, e.get("start_seconds") or 0.0, e.get("created_at") or ""))
    return json.dumps(events[:limit], indent=2)


TOOL_HANDLERS = {
    "ceft_search": _handle_ceft_search,
    "ceft_ask": _handle_ceft_ask,
    "ceft_get_evidence": _handle_ceft_get_evidence,
    "ceft_get_entity": _handle_ceft_get_entity,
    "ceft_list_sources": _handle_ceft_list_sources,
    "ceft_get_timeline": _handle_ceft_get_timeline,
}


def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 request and return the response."""
    msg_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    # Notifications do not return responses
    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS,
            },
        }

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}],
                    "isError": True,
                },
            }
        try:
            output_text = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": output_text}],
                },
            }
        except Exception as exc:
            _log.exception("Error executing tool %s", tool_name)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Execution error: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }


def run_stdio_server() -> None:
    """Run the JSON-RPC stdio loop."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()

