"""Gemini vision analysis for standalone images, video keyframes and PDF pages.

Besides a description and the OCR text, the model returns text blocks and
visual regions with bounding boxes, so evidence can later be traced to the
exact part of an image, and typed entities so the graph knows a "person"
from a "system" or a "metric".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas.knowledge import KnowledgeNode, MediaModality
from app.services import llm
from app.services.storage import storage_root


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be read or described by Gemini."""


_IMAGE_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

ENTITY_TYPES = (
    "person",
    "organization",
    "system",
    "component",
    "metric",
    "concept",
    "product",
    "location",
    "event",
)
IMAGE_TYPES = ("diagram", "chart", "slide", "screenshot", "document", "photo", "table", "other")

_PROMPT = f"""Analyze this image for a retrieval system that must answer questions
about what it shows and trace answers back to exact regions.

Return JSON with exactly these keys:
- "image_type": one of {list(IMAGE_TYPES)}
- "visual_summary": a detailed description of what is shown: diagram structure
  and arrows, chart axes and trends, layout, and what the image communicates.
- "ocr_text": all legible text, in reading order.
- "text_blocks": up to 40 objects {{"text": str, "box_2d": [ymin, xmin, ymax, xmax]}}
  for distinct blocks of legible text.
- "regions": up to 15 objects {{"label": str, "description": str,
  "box_2d": [ymin, xmin, ymax, xmax]}} for meaningful visual elements
  (diagram components, charts, tables, UI panels, people).
- "entities": objects {{"name": str, "type": one of {list(ENTITY_TYPES)}}} for named
  people, organizations, systems, components, metrics, products and key concepts.

box_2d coordinates are integers normalized to 0-1000 with the origin at the
top-left. Use empty arrays when nothing applies. Do not invent text that is
not legible."""


def _box(raw: Any) -> dict[str, float] | None:
    """Gemini ``box_2d`` ([ymin, xmin, ymax, xmax] in 0-1000) -> normalized x/y/w/h."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, float(v))) / 1000.0 for v in raw)
    except (TypeError, ValueError):
        return None
    if ymax <= ymin or xmax <= xmin:
        return None
    return {"x": round(xmin, 4), "y": round(ymin, 4),
            "width": round(xmax - xmin, 4), "height": round(ymax - ymin, 4)}


def _typed_entities(raw: Any) -> tuple[list[str], dict[str, str]]:
    names: list[str] = []
    types: dict[str, str] = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            kind = str(item.get("type") or "concept").strip().lower()
        else:
            name, kind = str(item).strip(), "concept"
        if not name or name in types:
            continue
        names.append(name)
        types[name] = kind if kind in ENTITY_TYPES else "concept"
    return names, types


def normalize_analysis(analysis: Any) -> dict[str, Any]:
    """Validate and normalize a raw model response into the processor contract."""
    if not isinstance(analysis, dict):
        raise ImageProcessingError("Vision model returned a non-object response.")
    blocks = []
    for block in analysis.get("text_blocks") or []:
        if isinstance(block, dict) and str(block.get("text") or "").strip():
            blocks.append({"text": str(block["text"]).strip(), "box": _box(block.get("box_2d"))})
    regions = []
    for region in analysis.get("regions") or []:
        if isinstance(region, dict) and str(region.get("label") or "").strip():
            regions.append(
                {
                    "label": str(region["label"]).strip(),
                    "description": str(region.get("description") or "").strip(),
                    "box": _box(region.get("box_2d")),
                }
            )
    names, types = _typed_entities(analysis.get("entities"))
    image_type = str(analysis.get("image_type") or "other").strip().lower()
    return {
        "image_type": image_type if image_type in IMAGE_TYPES else "other",
        "visual_summary": str(analysis.get("visual_summary") or "").strip(),
        "ocr_text": str(analysis.get("ocr_text") or "").strip(),
        "ocr_blocks": blocks,
        "regions": regions,
        "entities": names,
        "entity_types": types,
    }


def local_fallback_analysis(image_path: Path) -> dict[str, Any]:
    """Offline/Zero-API-key image analysis fallback.

    Extracts basic image properties, attempts local OCR if pytesseract is
    installed, and extracts keywords/entities from the filename and text.
    """
    from PIL import Image as PILImage

    ocr_text = ""
    ocr_blocks: list[dict[str, Any]] = []
    width, height = 0, 0
    try:
        with PILImage.open(image_path) as img:
            width, height = img.size
            try:
                import pytesseract

                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                n_boxes = len(data.get("text", []))
                for i in range(n_boxes):
                    t = str(data["text"][i]).strip()
                    if not t:
                        continue
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    if width > 0 and height > 0:
                        ocr_blocks.append({
                            "text": t,
                            "box": {
                                "x": round(x / width, 4),
                                "y": round(y / height, 4),
                                "width": round(w / width, 4),
                                "height": round(h / height, 4),
                            },
                        })
                ocr_text = pytesseract.image_to_string(img).strip()
            except Exception:
                pass
    except Exception as exc:
        raise ImageProcessingError(f"Local image processing failed for {image_path.name}: {exc}") from exc

    stem_lower = image_path.stem.lower()
    if any(k in stem_lower for k in ("chart", "graph", "metric", "plot")):
        image_type = "chart"
    elif any(k in stem_lower for k in ("diagram", "arch", "flow", "schema")):
        image_type = "diagram"
    elif any(k in stem_lower for k in ("slide", "deck", "presentation")):
        image_type = "slide"
    elif any(k in stem_lower for k in ("screen", "ui", "shot", "page")):
        image_type = "screenshot"
    else:
        image_type = "other"

    clean_stem = image_path.stem.replace("-", " ").replace("_", " ")
    if ocr_text:
        visual_summary = f"{image_type.capitalize()} '{clean_stem}' ({width}x{height}) showing text: {ocr_text[:160]}"
    else:
        visual_summary = f"{image_type.capitalize()} '{clean_stem}' ({width}x{height})"

    words = [w for w in clean_stem.split() if len(w) > 2]
    entities = list(dict.fromkeys(words))
    entity_types = {e: "concept" for e in entities}

    return {
        "image_type": image_type,
        "visual_summary": visual_summary,
        "ocr_text": ocr_text or clean_stem,
        "ocr_blocks": ocr_blocks,
        "regions": [],
        "entities": entities,
        "entity_types": entity_types,
        "local_fallback": True,
    }


def analyze_image(image_path: Path, *, client: Any | None = None) -> dict[str, Any]:
    """Describe an image: summary, OCR text and blocks, regions, typed entities."""
    if not image_path.is_file():
        raise ImageProcessingError(f"Image source does not exist: {image_path}")
    media_type = _IMAGE_MEDIA_TYPES.get(image_path.suffix.lower())
    if media_type is None:
        raise ImageProcessingError("Only PNG and JPEG images are supported.")
    try:
        part = llm.Image(data=image_path.read_bytes(), mime_type=media_type)
        raw = llm.generate_json([_PROMPT, part], client=client)
    except llm.LLMNotConfigured:
        return local_fallback_analysis(image_path)
    except llm.LLMError as exc:
        raise ImageProcessingError(f"Vision analysis failed for {image_path.name}: {exc}") from exc
    except ImportError as exc:
        raise ImageProcessingError("The model provider's SDK is required for image analysis.") from exc
    except OSError as exc:
        raise ImageProcessingError(f"Unable to read {image_path.name}: {exc}") from exc
    return normalize_analysis(raw)


def _servable_upload_path(image_path: Path) -> str:
    """URL of a persisted upload under the ``/uploads`` static mount."""
    try:
        relative = image_path.resolve().relative_to(storage_root() / "uploads")
    except ValueError:
        return f"/uploads/{image_path.name}"
    return f"/uploads/{relative.as_posix()}"


def process_image(file_path: str, *, client: Any | None = None) -> KnowledgeNode:
    """Create one image knowledge node from a local PNG or JPEG upload."""
    image_path = Path(file_path)
    analysis = analyze_image(image_path, client=client)
    confidence = 0.65 if analysis.get("local_fallback") else 0.85
    return KnowledgeNode(
        content=analysis["ocr_text"],
        transcript=analysis["ocr_text"],
        visual_summary=analysis["visual_summary"] or None,
        modality=MediaModality.IMAGE,
        source=image_path.name,
        confidence=confidence,
        frame_path=_servable_upload_path(image_path),
        entities=analysis["entities"],
        entity_types=analysis["entity_types"],
        provenance={"kind": "standalone_image"},
        attributes={
            "image_type": analysis["image_type"],
            "ocr_blocks": analysis["ocr_blocks"],
            "regions": analysis["regions"],
            "local_fallback": analysis.get("local_fallback", False),
        },
    )
