// Timeline: chronological sequence of multimodal evidence (videos, audio,
// document pages, screenshots, and logs).

import { api } from "../api.js";
import { openEvidence } from "../evidence.js";
import { h, clear, icon, modChip, emptyState, skeletonLines, fill, put } from "../ui.js";
import { navigate } from "../main.js";

const MODALITIES = ["all", "video", "audio", "pdf", "image", "json"];

export function renderTimeline(root, params) {
  let disposed = false;
  let allEvents = [];
  let modalityFilter = params.get("modality") || "all";

  const filters = h("div", { class: "chips filter-row" });
  const host = h("div", { class: "timeline-container" }, h("div", { class: "card-pad" }, skeletonLines(6)));

  put(root,
    h("header", { class: "page-head" },
      h("div", {},
        h("h1", {}, "Event Timeline"),
        h("p", { class: "muted" }, "Multimodal evidence arranged chronologically across meetings, slides, postmortems, and logs."),
      ),
    ),
    h("div", { class: "toolbar" }, filters),
    host,
  );

  function renderFilters() {
    fill(filters,
      MODALITIES.map((m) =>
        h("button", {
          class: "chip",
          type: "button",
          "aria-pressed": String(modalityFilter === m),
          onclick: () => {
            modalityFilter = m;
            renderFilters();
            renderStream();
          },
        }, m.charAt(0).toUpperCase() + m.slice(1))
      )
    );
  }

  function renderStream() {
    clear(host);
    const visible = allEvents.filter((ev) => modalityFilter === "all" || ev.modality === modalityFilter);

    if (!visible.length) {
      put(host, emptyState({
        title: allEvents.length ? "No events in this modality" : "No timeline events found",
        body: allEvents.length ? "Switch to 'All' or another modality." : "Upload sources or seed sample data to view the timeline.",
      }));
      return;
    }

    const items = visible.map((ev) => {
      const open = () => {
        openEvidence({
          segment_id: ev.id,
          source_id: ev.source_id,
          source: ev.source,
          modality: ev.modality,
          timestamp: ev.time_display,
          transcript: ev.snippet,
          visual_summary: ev.visual_summary,
          frame_path: ev.frame_path,
        });
      };

      const entityBadges = (ev.entities || []).slice(0, 4).map((name) =>
        h("span", { class: "tag", style: "margin-right: 4px; font-size: 11px;" }, name)
      );

      const speakerBadge = (ev.speakers || []).length
        ? h("span", { class: "speaker", style: "font-size: 12px; color: var(--accent); margin-right: 8px;" },
            icon("user", 12), ` ${(ev.speakers || []).join(", ")}`)
        : null;

      return h("div", { class: "timeline-item card", style: "margin-bottom: 16px; padding: 16px; display: flex; gap: 16px; align-items: flex-start;" },
        h("div", { class: "timeline-badge", style: "min-width: 100px; font-size: 12px; font-weight: 600; color: var(--muted);" },
          h("div", { style: "color: var(--text); font-size: 13px;" }, ev.time_display || "—"),
          h("div", { style: "margin-top: 4px;" }, modChip(ev.modality)),
        ),
        h("div", { class: "timeline-content", style: "flex: 1;" },
          h("div", { style: "display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;" },
            h("a", {
              href: `#/sources/${ev.source_id}`,
              style: "font-weight: 600; font-family: monospace; font-size: 13px; text-decoration: none; color: var(--accent);",
            }, ev.source),
            h("button", {
              class: "btn btn-ghost btn-sm",
              type: "button",
              onclick: open,
              title: "Open in evidence drawer",
            }, "View Evidence"),
          ),
          speakerBadge,
          h("p", { style: "margin: 6px 0 8px; font-size: 13px; line-height: 1.5; color: var(--text);" }, ev.snippet || ev.visual_summary || "No excerpt."),
          entityBadges.length ? h("div", { style: "margin-top: 6px;" }, entityBadges) : null,
        ),
      );
    });

    put(host, h("div", { class: "timeline-list" }, items));
  }

  async function load() {
    try {
      allEvents = await api.timeline({ limit: 150 });
      if (disposed) return;
      renderFilters();
      renderStream();
    } catch (err) {
      if (disposed) return;
      clear(host);
      put(host, emptyState({ title: "Failed to load timeline", body: err.message }));
    }
  }

  renderFilters();
  load();

  return () => {
    disposed = true;
  };
}

