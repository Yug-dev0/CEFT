"""Investigation Dossier Generator.

Produces standalone, comprehensive investigation reports (in Markdown and HTML)
summarizing user queries, synthesized answers, byte-exact spatiotemporal citations,
and the cross-modal evidence chain.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
from typing import Any


def generate_markdown_dossier(
    query: str,
    answer_text: str,
    sources: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    *,
    grounded: bool = True,
    method: str = "llm",
) -> str:
    """Generate a GitHub-flavored Markdown dossier for postmortems and issues."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# Investigation Dossier: {query}",
        "",
        f"> **Generated:** {now}  ",
        f"> **Status:** {'🟢 Grounded & Verified' if grounded else '🟡 Partially Grounded / Extractive'}  ",
        f"> **Synthesis Method:** {method.upper()}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Findings",
        "",
        answer_text.strip(),
        "",
        "---",
        "",
        "## 2. Evidence Provenance Matrix",
        "",
        "| Citation | Source File | Modality | Exact Locator | Score / Conf | Evidence Excerpt |",
        "| :---: | :--- | :---: | :--- | :---: | :--- |",
    ]

    for s in sources:
        cite = s.get("citation", "?")
        src = s.get("source", "unknown")
        mod = s.get("modality", "unknown")
        loc = s.get("locator", "—")
        conf = f"{float(s['confidence']):.2f}" if s.get("confidence") is not None else "—"
        snippet = str(s.get("snippet") or s.get("text") or s.get("visual_summary") or "").replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"| **[{cite}]** | `{src}` | `{mod}` | {loc} | {conf} | {snippet} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Multimodal Evidence",
        "",
    ])

    for i, hit in enumerate(hits, start=1):
        src = hit.get("source", "unknown")
        metadata = hit.get("metadata") or {}
        mod = metadata.get("modality") or hit.get("modality", "unknown")
        loc = hit.get("timestamp") or "—"
        score = hit.get("score")
        score_str = f"{float(score):.3f}" if score is not None else "N/A"
        transcript = hit.get("transcript")
        visual = hit.get("visual_summary")
        entities = hit.get("entities") or []
        speakers = hit.get("speakers") or []
        via = hit.get("via")

        lines.append(f"### Evidence #{i}: `{src}` ({mod})")
        lines.append(f"- **Locator:** {loc}")
        lines.append(f"- **Retrieval Score:** {score_str}")
        if speakers:
            lines.append(f"- **Speaker(s):** {', '.join(speakers)}")
        if entities:
            lines.append(f"- **Entities:** {', '.join(entities)}")
        if via:
            lines.append(f"- **Graph Link:** `{via.get('relation')}` ({via.get('direction')}) from segment `{via.get('from_segment_id')}`")
        if transcript:
            lines.append(f"\n**Spoken / Text Content:**\n> {transcript.strip()}\n")
        if visual:
            lines.append(f"\n**Visual Description:**\n> {visual.strip()}\n")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Report automatically compiled by Ceft Multimodal Provenance-Aware RAG Pipeline.*",
    ])
    return "\n".join(lines)


def generate_html_dossier(
    query: str,
    answer_text: str,
    sources: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    *,
    grounded: bool = True,
    method: str = "llm",
) -> str:
    """Generate a self-contained, beautifully styled HTML dossier."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    esc = html.escape

    rows_html = []
    for s in sources:
        cite = esc(str(s.get("citation", "?")))
        src = esc(str(s.get("source", "unknown")))
        mod = esc(str(s.get("modality", "unknown")))
        loc = esc(str(s.get("locator", "—")))
        conf = f"{float(s['confidence']):.2f}" if s.get("confidence") is not None else "—"
        snippet = esc(str(s.get("snippet") or s.get("text") or s.get("visual_summary") or "")[:140])
        rows_html.append(
            f"<tr><td><span class='badge cite'>[{cite}]</span></td>"
            f"<td><code>{src}</code></td>"
            f"<td><span class='badge mod'>{mod}</span></td>"
            f"<td>{loc}</td><td>{conf}</td><td>{snippet}</td></tr>"
        )

    cards_html = []
    for i, hit in enumerate(hits, start=1):
        src = esc(str(hit.get("source", "unknown")))
        metadata = hit.get("metadata") or {}
        mod = esc(str(metadata.get("modality") or hit.get("modality", "unknown")))
        loc = esc(str(hit.get("timestamp") or "—"))
        score = hit.get("score")
        score_str = f"{float(score):.3f}" if score is not None else "—"
        transcript = hit.get("transcript")
        visual = hit.get("visual_summary")
        entities = hit.get("entities") or []

        body = []
        if transcript:
            body.append(f"<div class='section'><strong>Transcript / Text:</strong><p class='quote'>{esc(transcript)}</p></div>")
        if visual:
            body.append(f"<div class='section'><strong>Visual Analysis:</strong><p class='quote visual'>{esc(visual)}</p></div>")

        cards_html.append(f"""
        <div class="evidence-card">
          <div class="card-header">
            <h4>#{i} &bull; <code>{src}</code> <span class="badge mod">{mod}</span></h4>
            <span class="meta">Locator: <strong>{loc}</strong> &bull; Score: <strong>{score_str}</strong></span>
          </div>
          <div class="card-body">
            {''.join(body)}
            {'<div class="entities"><strong>Entities:</strong> ' + ' '.join(f'<span class="badge ent">{esc(e)}</span>' for e in entities) + '</div>' if entities else ''}
          </div>
        </div>
        """)

    status_badge = "<span class='badge success'>🟢 Grounded & Verified</span>" if grounded else "<span class='badge warning'>🟡 Extractive / Unverified</span>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Ceft Investigation Dossier: {esc(query)}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --border: #334155;
      --accent: #38bdf8;
      --accent-dim: rgba(56, 189, 248, 0.1);
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      margin: 0;
      padding: 32px 16px;
      line-height: 1.6;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
      margin-bottom: 32px;
    }}
    h1 {{
      font-size: 26px;
      margin: 0 0 12px;
      color: #fff;
    }}
    .meta-bar {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
      align-items: center;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge.success {{ background: #065f46; color: #34d399; }}
    .badge.warning {{ background: #854d0e; color: #fde047; }}
    .badge.mod {{ background: #1e3a8a; color: #93c5fd; }}
    .badge.cite {{ background: #4338ca; color: #e0e7ff; }}
    .badge.ent {{ background: #312e81; color: #c7d2fe; margin-right: 4px; }}
    .summary-box {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 32px;
    }}
    .summary-box h2 {{ margin-top: 0; font-size: 18px; color: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0 32px;
      background: var(--card-bg);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}
    th {{
      background: #0f172a;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.5px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
      background: rgba(255, 255, 255, 0.06);
      padding: 2px 4px;
      border-radius: 4px;
    }}
    .evidence-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 20px;
      overflow: hidden;
    }}
    .card-header {{
      padding: 12px 18px;
      background: rgba(0, 0, 0, 0.2);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .card-header h4 {{ margin: 0; font-size: 15px; font-weight: 600; }}
    .card-header .meta {{ font-size: 12px; color: var(--muted); }}
    .card-body {{ padding: 18px; }}
    .quote {{
      margin: 8px 0;
      padding: 10px 14px;
      background: rgba(0, 0, 0, 0.15);
      border-left: 3px solid #64748b;
      border-radius: 0 4px 4px 0;
      font-size: 14px;
    }}
    .quote.visual {{ border-left-color: #8b5cf6; }}
    .entities {{ margin-top: 14px; font-size: 13px; }}
    footer {{
      margin-top: 48px;
      padding-top: 18px;
      border-top: 1px solid var(--border);
      text-align: center;
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Investigation Dossier: {esc(query)}</h1>
      <div class="meta-bar">
        <span>Generated: <strong>{now}</strong></span>
        <span>{status_badge}</span>
        <span>Method: <strong>{esc(method.upper())}</strong></span>
      </div>
    </header>

    <div class="summary-box">
      <h2>Executive Findings & Provenance Synthesis</h2>
      <p>{esc(answer_text)}</p>
    </div>

    <h3>Evidence Provenance Matrix</h3>
    <table>
      <thead>
        <tr>
          <th>Citation</th>
          <th>Source</th>
          <th>Modality</th>
          <th>Locator</th>
          <th>Conf</th>
          <th>Evidence Excerpt</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>

    <h3>Detailed Multimodal Evidence</h3>
    <div class="evidence-list">
      {''.join(cards_html)}
    </div>

    <footer>
      Compiled by Ceft &bull; Multimodal Provenance-Aware Retrieval-Augmented Generation
    </footer>
  </div>
</body>
</html>"""

