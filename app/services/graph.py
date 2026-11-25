"""Read-side graph queries: entity timelines and node neighbourhoods."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

from app.db.repository import KnowledgeRepository
from app.schemas.records import (
    CrossModalBridge,
    EntityTimeline,
    GraphAnalytics,
    GraphEdge,
    GraphNode,
    GraphView,
    NodeKind,
    RelationType,
    SegmentRecord,
    SourceRecord,
    TimelineEntry,
    TopEntityMetric,
)


def source_time(source: SourceRecord) -> tuple[datetime, str]:
    """When the source's content happened: ``recorded_at`` if given, else ingestion time."""
    raw = source.attributes.get("recorded_at")
    if isinstance(raw, str) and raw:
        try:
            when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return when, "recorded_at"
        except ValueError:
            pass
    when = source.ingested_at
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when, "ingested_at"


def segment_label(segment: SegmentRecord, filename: str | None = None) -> str:
    where = segment.locator.display() or ("image" if segment.modality.value == "image" else f"#{segment.ordinal + 1}")
    return f"{filename} · {where}" if filename else where


def entity_timeline(repo: KnowledgeRepository, entity_id: str) -> EntityTimeline | None:
    entity = repo.get_entity(entity_id)
    if entity is None:
        return None
    groups = repo.same_as_groups()
    group = groups.get(entity_id, entity_id)
    alias_ids = [e for e, g in groups.items() if g == group and e != entity_id]
    aliases = list(repo.get_entities(alias_ids).values())

    segment_ids = sorted({sid for sid, _ in repo.mentions_of([entity_id, *alias_ids])})
    segments = repo.get_segments(segment_ids)
    sources: dict[str, SourceRecord] = {}
    entries: list[TimelineEntry] = []
    for segment in segments.values():
        source = sources.get(segment.source_id) or repo.get_source(segment.source_id)
        if source is None:
            continue
        sources[source.id] = source
        when, basis = source_time(source)
        entries.append(
            TimelineEntry(
                when=when,
                when_source=basis,
                offset_seconds=segment.locator.start_seconds,
                segment=segment,
                source=source,
            )
        )
    entries.sort(key=lambda e: (e.when, e.source.id, e.segment.ordinal))
    return EntityTimeline(entity=entity, aliases=aliases, entries=entries)


def neighbourhood(
    repo: KnowledgeRepository,
    node_id: str,
    *,
    depth: int = 1,
    limit: int = 120,
    include_next: bool = False,
) -> GraphView | None:
    """Breadth-first neighbourhood of a segment or entity, capped at ``limit`` nodes."""
    start_segment = repo.get_segment(node_id)
    start_entity = None if start_segment else repo.get_entity(node_id)
    if start_segment is None and start_entity is None:
        return None

    seen: set[str] = {node_id}
    edges: dict[str, GraphEdge] = {}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    while queue and len(seen) < limit:
        current, level = queue.popleft()
        if level >= depth:
            continue
        for relation, _direction in repo.relations_for(current):
            if relation.relation is RelationType.NEXT and not include_next:
                continue
            edges[relation.id] = GraphEdge(
                id=relation.id,
                source=relation.src_id,
                target=relation.dst_id,
                relation=relation.relation,
                confidence=relation.confidence,
            )
            for other in (relation.src_id, relation.dst_id):
                if other not in seen and len(seen) < limit:
                    seen.add(other)
                    queue.append((other, level + 1))

    segments = repo.get_segments(list(seen))
    entities = repo.get_entities([n for n in seen if n not in segments])
    filenames: dict[str, str] = {}
    nodes: list[GraphNode] = []
    for seg in segments.values():
        if seg.source_id not in filenames:
            src = repo.get_source(seg.source_id)
            filenames[seg.source_id] = src.filename if src else seg.source_id
        nodes.append(
            GraphNode(
                id=seg.id,
                kind=NodeKind.SEGMENT,
                label=segment_label(seg, filenames[seg.source_id]),
                modality=seg.modality.value,
                source_id=seg.source_id,
            )
        )
    for ent in entities.values():
        nodes.append(GraphNode(id=ent.id, kind=NodeKind.ENTITY, label=ent.name, entity_type=ent.entity_type))
    known = {n.id for n in nodes}
    return GraphView(
        center=node_id,
        nodes=nodes,
        edges=[e for e in edges.values() if e.source in known and e.target in known],
    )


OVERVIEW_RELATIONS = (
    RelationType.DEPICTS,
    RelationType.CORROBORATES,
    RelationType.CO_OCCURS,
    RelationType.SHOWS_SAME,
)


def overview(
    repo: KnowledgeRepository,
    *,
    entities: bool = True,
    min_sources: int = 2,
    limit: int = 400,
) -> GraphView:
    """The whole knowledge base as one graph.

    Every segment that has a cross-modal or temporal link, plus (optionally)
    the entities that tie at least ``min_sources`` different sources
    together, with their ``mentions`` edges. Isolated segments and entities
    seen in a single source are left out: they add nodes, not structure.
    """
    links = repo.list_relations(list(OVERVIEW_RELATIONS))
    segment_ids: set[str] = set()
    for rel in links:
        segment_ids.update((rel.src_id, rel.dst_id))

    entity_edges: list = []
    if entities:
        mentions = repo.list_relations([RelationType.MENTIONS])
        all_segments = {seg.id: seg for batch in repo.iter_all_segments() for seg in batch}
        sources_of: dict[str, set[str]] = defaultdict(set)
        for rel in mentions:
            seg = all_segments.get(rel.src_id)
            if seg is not None:
                sources_of[rel.dst_id].add(seg.source_id)
        keep = {eid for eid, srcs in sources_of.items() if len(srcs) >= min_sources}
        entity_edges = [r for r in mentions if r.dst_id in keep and r.src_id in all_segments]
        segment_ids.update(r.src_id for r in entity_edges)

    segments = repo.get_segments(sorted(segment_ids))
    # Keep the most connected segments when the corpus is larger than the cap.
    degree: dict[str, int] = defaultdict(int)
    for rel in [*links, *entity_edges]:
        degree[rel.src_id] += 1
        degree[rel.dst_id] += 1
    entity_ids = {r.dst_id for r in entity_edges}
    ranked = sorted([*segments, *entity_ids], key=lambda n: degree[n], reverse=True)[:limit]
    kept = set(ranked)

    filenames: dict[str, str] = {}
    nodes: list[GraphNode] = []
    for seg in segments.values():
        if seg.id not in kept:
            continue
        if seg.source_id not in filenames:
            src = repo.get_source(seg.source_id)
            filenames[seg.source_id] = src.filename if src else seg.source_id
        nodes.append(GraphNode(id=seg.id, kind=NodeKind.SEGMENT, label=segment_label(seg, filenames[seg.source_id]),
                               modality=seg.modality.value, source_id=seg.source_id))
    for ent in repo.get_entities(sorted(entity_ids & kept)).values():
        nodes.append(GraphNode(id=ent.id, kind=NodeKind.ENTITY, label=ent.name, entity_type=ent.entity_type))

    known = {n.id for n in nodes}
    edges = [
        GraphEdge(id=r.id, source=r.src_id, target=r.dst_id, relation=r.relation, confidence=r.confidence)
        for r in [*links, *entity_edges]
        if r.src_id in known and r.dst_id in known
    ]
    return GraphView(center=None, nodes=nodes, edges=edges)


def compute_graph_analytics(repo: KnowledgeRepository) -> GraphAnalytics:
    """Compute topology metrics, PageRank centrality, and cross-modal bridge nodes."""
    all_segments = {seg.id: seg for batch in repo.iter_all_segments() for seg in batch}
    all_relations = repo.list_relations(list(RelationType))
    all_entities = {e.id: e for e in repo.all_entities()}

    modality_distribution: dict[str, int] = defaultdict(int)
    for seg in all_segments.values():
        modality_distribution[seg.modality.value] += 1

    # Build adjacency graph
    neighbors: dict[str, set[str]] = defaultdict(set)
    degree: dict[str, int] = defaultdict(int)

    for r in all_relations:
        neighbors[r.src_id].add(r.dst_id)
        neighbors[r.dst_id].add(r.src_id)
        degree[r.src_id] += 1
        degree[r.dst_id] += 1

    all_node_ids = set(all_segments.keys()) | set(all_entities.keys()) | set(degree.keys())
    total_nodes = len(all_node_ids)
    total_edges = len(all_relations)
    density = (2.0 * total_edges) / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0.0

    # PageRank algorithm (power iteration)
    pagerank: dict[str, float] = {}
    if total_nodes > 0:
        d = 0.85
        pr = {nid: 1.0 / total_nodes for nid in all_node_ids}
        for _ in range(20):
            next_pr = {nid: (1.0 - d) / total_nodes for nid in all_node_ids}
            for nid in all_node_ids:
                outs = neighbors[nid]
                if outs:
                    contrib = d * pr[nid] / len(outs)
                    for target in outs:
                        next_pr[target] += contrib
                else:
                    contrib = d * pr[nid] / total_nodes
                    for target in all_node_ids:
                        next_pr[target] += contrib
            pr = next_pr
        pagerank = pr

    # Top entities by PageRank
    top_pr_entities: list[TopEntityMetric] = []
    for eid, ent in all_entities.items():
        score = pagerank.get(eid, 0.0)
        top_pr_entities.append(
            TopEntityMetric(
                entity_id=eid,
                name=ent.name,
                entity_type=ent.entity_type,
                score=round(score, 6),
                degree=degree.get(eid, 0),
            )
        )
    top_pr_entities.sort(key=lambda m: (m.score, m.degree), reverse=True)
    top_pr_entities = top_pr_entities[:15]

    # Top connected entities
    top_connected: list[TopEntityMetric] = []
    for eid, ent in all_entities.items():
        deg = degree.get(eid, 0)
        top_connected.append(
            TopEntityMetric(
                entity_id=eid,
                name=ent.name,
                entity_type=ent.entity_type,
                score=float(deg),
                degree=deg,
            )
        )
    top_connected.sort(key=lambda m: m.degree, reverse=True)
    top_connected = top_connected[:15]

    # Cross-modal bridges
    bridges: list[CrossModalBridge] = []
    mentions = repo.list_relations([RelationType.MENTIONS])
    entity_to_segments: dict[str, set[str]] = defaultdict(set)
    for r in mentions:
        entity_to_segments[r.dst_id].add(r.src_id)

    for eid, seg_ids in entity_to_segments.items():
        ent = all_entities.get(eid) or repo.get_entity(eid)
        if ent is None:
            continue
        seg_objs = [all_segments[sid] for sid in seg_ids if sid in all_segments]
        modalities = sorted({s.modality.value for s in seg_objs})
        sources = {s.source_id for s in seg_objs}
        if len(modalities) >= 2:
            bridge_score = round(len(modalities) * 1.5 + len(sources) * 1.0 + degree.get(eid, 0) * 0.2, 2)
            bridges.append(
                CrossModalBridge(
                    id=eid,
                    name=ent.name,
                    kind=NodeKind.ENTITY,
                    modalities=modalities,
                    source_count=len(sources),
                    bridge_score=bridge_score,
                )
            )
    bridges.sort(key=lambda b: b.bridge_score, reverse=True)

    return GraphAnalytics(
        total_nodes=total_nodes,
        total_edges=total_edges,
        density=round(density, 6),
        modality_distribution=dict(modality_distribution),
        top_entities_by_pagerank=top_pr_entities,
        top_connected_entities=top_connected,
        cross_modal_bridges=bridges[:20],
    )

