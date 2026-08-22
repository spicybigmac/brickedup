from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


INK = colors.HexColor("#1C1C1A")
PAPER = colors.HexColor("#F7F3EA")
WHITE = colors.HexColor("#FFFDF8")
YELLOW = colors.HexColor("#E8B843")
BLUE = colors.HexColor("#2854A6")
RED = colors.HexColor("#C84832")
GHOST = colors.HexColor("#D8D4CC")
GRID = colors.HexColor("#C9C3B8")


def _fallback_copy(piece_count: int, layer_count: int) -> dict[str, Any]:
    return {
        "title": "Your Brick Build",
        "intro": f"{piece_count} pieces · {layer_count} steps",
        "safety": "Small parts. Build on a flat surface.",
        "tips": ["Sort", "Align", "Press"],
    }


def generate_copy(bricks: list[dict[str, Any]], bom: dict[str, int]) -> dict[str, Any]:
    """Use Gemini only for a terse title; geometry remains the source of truth."""
    layer_count = len({brick["z"] for brick in bricks})
    fallback = _fallback_copy(len(bricks), layer_count)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback
    try:
        prompt = f"""Name an original brick sculpture in at most four words.
It contains {len(bricks)} pieces across {layer_count} layers. Parts: {json.dumps(bom)}.
Return JSON only: {{"title":"..."}}. Do not mention or imply affiliation with LEGO."""
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=30,
        )
        response.raise_for_status()
        title = json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"]).get("title")
        if title:
            return {**fallback, "title": str(title)[:60]}
    except Exception:
        pass
    return fallback


def _hex(value: str, fallback: colors.Color = RED) -> colors.Color:
    try:
        return colors.HexColor(value)
    except Exception:
        return fallback


def _shade(color: colors.Color, factor: float) -> colors.Color:
    return colors.Color(
        min(1, color.red * factor),
        min(1, color.green * factor),
        min(1, color.blue * factor),
    )


def _page_base(pdf: canvas.Canvas) -> None:
    page_width, page_height = A4
    pdf.setFillColor(PAPER)
    pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)


def _header(pdf: canvas.Canvas, label: str, number: int) -> None:
    page_width, page_height = A4
    pdf.setFillColor(BLUE)
    pdf.rect(0, page_height - 16 * mm, page_width, 16 * mm, fill=1, stroke=0)
    pdf.setFillColor(YELLOW)
    pdf.circle(13 * mm, page_height - 8 * mm, 5.2 * mm, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawCentredString(13 * mm, page_height - 10.2 * mm, str(number))
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(23 * mm, page_height - 10.5 * mm, label.upper())


def _bounds(bricks: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    if not bricks:
        return 0, 0, 1, 1
    min_x = min(int(brick["x"]) for brick in bricks)
    min_y = min(int(brick["y"]) for brick in bricks)
    max_x = max(int(brick["x"]) + int(brick.get("width", 1)) for brick in bricks)
    max_y = max(int(brick["y"]) + int(brick.get("depth", 1)) for brick in bricks)
    return min_x, min_y, max_x, max_y


def _plan_transform(
    bricks: list[dict[str, Any]],
    area: tuple[float, float, float, float],
) -> tuple[float, float, float, tuple[int, int, int, int]]:
    area_x, area_y, area_width, area_height = area
    bounds = _bounds(bricks)
    min_x, min_y, max_x, max_y = bounds
    studs_w, studs_d = max_x - min_x, max_y - min_y
    cell = min(area_width / max(studs_w, 1), area_height / max(studs_d, 1))
    cell = max(2.3, cell)
    origin_x = area_x + (area_width - studs_w * cell) / 2
    origin_y = area_y + (area_height - studs_d * cell) / 2
    return origin_x, origin_y, cell, bounds


def _brick_rect(
    brick: dict[str, Any],
    origin_x: float,
    origin_y: float,
    cell: float,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    min_x, min_y, _max_x, max_y = bounds
    x = origin_x + (int(brick["x"]) - min_x) * cell
    width = int(brick.get("width", 1)) * cell
    depth = int(brick.get("depth", 1)) * cell
    y = origin_y + (max_y - int(brick["y"]) - int(brick.get("depth", 1))) * cell
    return x, y, width, depth


def _draw_studs(
    pdf: canvas.Canvas,
    brick: dict[str, Any],
    rect: tuple[float, float, float, float],
    cell: float,
    color: colors.Color,
    ghost: bool = False,
) -> None:
    x, y, _width, height = rect
    brick_w = max(1, int(brick.get("width", 1)))
    brick_d = max(1, int(brick.get("depth", 1)))
    radius = max(0.65, min(2.1, cell * 0.18))
    for ix in range(brick_w):
        for iy in range(brick_d):
            stud_x = x + (ix + 0.5) * cell
            stud_y = y + height - (iy + 0.5) * cell
            pdf.setFillColor(_shade(color, 1.08) if not ghost else colors.HexColor("#E8E5DF"))
            pdf.setStrokeColor(colors.HexColor("#6F6C66") if ghost else INK)
            pdf.setLineWidth(0.35)
            pdf.circle(stud_x, stud_y, radius, fill=1, stroke=1)


def _draw_plan_brick(
    pdf: canvas.Canvas,
    brick: dict[str, Any],
    origin_x: float,
    origin_y: float,
    cell: float,
    bounds: tuple[int, int, int, int],
    ghost: bool = False,
) -> None:
    rect = _brick_rect(brick, origin_x, origin_y, cell, bounds)
    x, y, width, height = rect
    color = GHOST if ghost else _hex(str(brick.get("color", "#C84832")))
    pdf.setFillColor(color)
    pdf.setStrokeColor(colors.HexColor("#77736B") if ghost else INK)
    pdf.setLineWidth(0.65 if ghost else 1.15)
    radius = min(2.2, cell * 0.12)
    pdf.roundRect(x + 0.35, y + 0.35, width - 0.7, height - 0.7, radius, fill=1, stroke=1)
    _draw_studs(pdf, brick, rect, cell, color, ghost)


def _draw_grid(
    pdf: canvas.Canvas,
    origin_x: float,
    origin_y: float,
    cell: float,
    bounds: tuple[int, int, int, int],
) -> None:
    min_x, min_y, max_x, max_y = bounds
    width, depth = max_x - min_x, max_y - min_y
    pdf.setFillColor(WHITE)
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(1.2)
    pdf.rect(origin_x, origin_y, width * cell, depth * cell, fill=1, stroke=1)
    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.25)
    for index in range(1, width):
        x = origin_x + index * cell
        pdf.line(x, origin_y, x, origin_y + depth * cell)
    for index in range(1, depth):
        y = origin_y + index * cell
        pdf.line(origin_x, y, origin_x + width * cell, y)


def _topmost_support(bricks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the visible stud cells from everything already placed."""
    visible: dict[tuple[int, int], dict[str, Any]] = {}
    for brick in sorted(bricks, key=lambda value: int(value["z"])):
        for dx in range(int(brick.get("width", 1))):
            for dy in range(int(brick.get("depth", 1))):
                visible[(int(brick["x"]) + dx, int(brick["y"]) + dy)] = {
                    "x": int(brick["x"]) + dx,
                    "y": int(brick["y"]) + dy,
                    "z": int(brick["z"]),
                    "width": 1,
                    "depth": 1,
                    "color": "#D8D4CC",
                }
    return list(visible.values())


def _draw_plan(
    pdf: canvas.Canvas,
    all_bricks: list[dict[str, Any]],
    placed: list[dict[str, Any]],
    current: list[dict[str, Any]],
    area: tuple[float, float, float, float],
    show_support: bool = True,
) -> None:
    origin_x, origin_y, cell, bounds = _plan_transform(all_bricks, area)
    _draw_grid(pdf, origin_x, origin_y, cell, bounds)
    if show_support:
        for support in _topmost_support(placed):
            _draw_plan_brick(pdf, support, origin_x, origin_y, cell, bounds, ghost=True)
    for brick in current:
        _draw_plan_brick(pdf, brick, origin_x, origin_y, cell, bounds)

    # Fixed viewpoint marker keeps every page oriented the same way.
    board_mid = origin_x + (_bounds(all_bricks)[2] - _bounds(all_bricks)[0]) * cell / 2
    tip_y = origin_y - 3 * mm
    pdf.setStrokeColor(BLUE)
    pdf.setFillColor(BLUE)
    pdf.setLineWidth(1.5)
    pdf.line(board_mid, tip_y - 8 * mm, board_mid, tip_y - 1.5 * mm)
    arrow = pdf.beginPath()
    arrow.moveTo(board_mid - 2.2 * mm, tip_y - 3.5 * mm)
    arrow.lineTo(board_mid + 2.2 * mm, tip_y - 3.5 * mm)
    arrow.lineTo(board_mid, tip_y)
    arrow.close()
    pdf.drawPath(arrow, fill=1, stroke=0)
    pdf.setFont("Helvetica-Bold", 6.5)
    pdf.drawCentredString(board_mid, tip_y - 12 * mm, "FRONT")


def _draw_brick_icon(pdf: canvas.Canvas, x: float, y: float, brick: dict[str, Any], scale: float = 5.2) -> None:
    width = max(1, int(brick.get("width", 1)))
    depth = max(1, int(brick.get("depth", 1)))
    color = _hex(str(brick.get("color", "#C84832")))
    body_width, body_height = width * scale, max(7, depth * scale * 0.62)
    pdf.setFillColor(color)
    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.7)
    pdf.roundRect(x, y, body_width, body_height, 1.2, fill=1, stroke=1)
    for ix in range(width):
        pdf.setFillColor(_shade(color, 1.08))
        pdf.circle(x + (ix + 0.5) * scale, y + body_height, scale * 0.22, fill=1, stroke=1)


def _draw_layer_meter(pdf: canvas.Canvas, active: int, count: int) -> None:
    x, y = 181 * mm, 29 * mm
    height = 54 * mm
    unit = height / max(count, 1)
    for layer in range(count):
        pdf.setFillColor(RED if layer == active else GHOST)
        pdf.setStrokeColor(INK)
        pdf.rect(x, y + layer * unit, 9 * mm, max(1.5 * mm, unit - 0.7), fill=1, stroke=1)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 6)
    pdf.drawCentredString(x + 4.5 * mm, y - 4 * mm, "HEIGHT")


def _final_top_bricks(bricks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find the highest brick covering each stud for an uncluttered cover mosaic."""
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for brick in sorted(bricks, key=lambda value: int(value["z"])):
        for dx in range(int(brick.get("width", 1))):
            for dy in range(int(brick.get("depth", 1))):
                cells[(int(brick["x"]) + dx, int(brick["y"]) + dy)] = {
                    "x": int(brick["x"]) + dx,
                    "y": int(brick["y"]) + dy,
                    "z": int(brick["z"]),
                    "width": 1,
                    "depth": 1,
                    "color": brick.get("color", "#C84832"),
                }
    return list(cells.values())


def _build_steps(by_layer: dict[int, list[dict[str, Any]]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Split dense layers into readable steps with complete part callouts."""
    steps: list[tuple[int, list[dict[str, Any]]]] = []
    for z in sorted(by_layer):
        ordered = sorted(by_layer[z], key=lambda brick: (int(brick["y"]), int(brick["x"])))
        batch: list[dict[str, Any]] = []
        types: set[tuple[int, int, str]] = set()
        for brick in ordered:
            part_type = (
                int(brick.get("width", 1)),
                int(brick.get("depth", 1)),
                str(brick.get("color", "#C84832")),
            )
            if batch and (len(batch) >= 12 or (part_type not in types and len(types) >= 6)):
                steps.append((z, batch))
                batch, types = [], set()
            batch.append(brick)
            types.add(part_type)
        if batch:
            steps.append((z, batch))
    return steps


def build_pdf(
    output_path: Path,
    bricks: list[dict[str, Any]],
    bom: dict[str, int],
    copy: dict[str, Any],
) -> None:
    """Render exact top-down placement diagrams in a layer-by-layer manual."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=A4, pageCompression=1)
    page_width, page_height = A4
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for brick in bricks:
        by_layer[int(brick["z"])].append(brick)
    layers = sorted(by_layer)
    build_steps = _build_steps(by_layer)

    # Cover: final top-view mosaic rather than an ambiguous pseudo-3D drawing.
    _page_base(pdf)
    pdf.setFillColor(RED)
    pdf.rect(0, page_height - 46 * mm, page_width, 46 * mm, fill=1, stroke=0)
    pdf.setFillColor(YELLOW)
    pdf.roundRect(15 * mm, page_height - 57 * mm, page_width - 30 * mm, 25 * mm, 3 * mm, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 25)
    pdf.drawCentredString(page_width / 2, page_height - 49 * mm, str(copy.get("title", "Your Brick Build")).upper())
    if bricks:
        final = _final_top_bricks(bricks)
        _draw_plan(pdf, bricks, [], final, (24 * mm, 60 * mm, page_width - 48 * mm, 126 * mm), False)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawCentredString(page_width / 2, 27 * mm, f"{len(bricks)} PIECES   ·   {len(build_steps)} STEPS")
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(page_width / 2, 17 * mm, "BRICKED UP  ·  BUILDING INSTRUCTIONS")
    pdf.showPage()

    # Inventory: actual color and size for every part family, never truncated.
    inventory = Counter(
        (int(brick.get("width", 1)), int(brick.get("depth", 1)), str(brick.get("color", "#C84832")))
        for brick in bricks
    )
    columns, cell_width, cell_height = 3, (page_width - 30 * mm) / 3, 29 * mm
    inventory_items = list(inventory.items())
    pages = [inventory_items[index:index + 24] for index in range(0, len(inventory_items), 24)] or [[]]
    for inventory_page, entries in enumerate(pages, start=1):
        _page_base(pdf)
        page_label = "Parts" if len(pages) == 1 else f"Parts {inventory_page}/{len(pages)}"
        _header(pdf, page_label, 0)
        for index, ((width, depth, color), count) in enumerate(entries):
            column, row = index % columns, index // columns
            x = 15 * mm + column * cell_width
            y = page_height - 27 * mm - (row + 1) * cell_height
            pdf.setFillColor(WHITE)
            pdf.setStrokeColor(GRID)
            pdf.roundRect(x, y, cell_width - 4 * mm, cell_height - 4 * mm, 2 * mm, fill=1, stroke=1)
            sample = {"width": width, "depth": depth, "color": color}
            icon_scale = min(7.0, (cell_width - 27 * mm) / max(width, 1))
            _draw_brick_icon(pdf, x + 5 * mm, y + 9 * mm, sample, icon_scale)
            pdf.setFillColor(INK)
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawRightString(x + cell_width - 9 * mm, y + 14 * mm, f"× {count}")
            pdf.setFont("Helvetica", 7)
            pdf.drawRightString(x + cell_width - 9 * mm, y + 8 * mm, f"{width} × {depth}")
        pdf.setFillColor(YELLOW)
        pdf.rect(0, 0, page_width, 14 * mm, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawCentredString(page_width / 2, 5 * mm, f"TOTAL  {sum(bom.values()) if bom else len(bricks)}")
        pdf.showPage()

    # Use per-layer part callouts with a large, diagram-first build area.
    placed: list[dict[str, Any]] = []
    layer_indexes = {z: index for index, z in enumerate(layers)}
    for step, (z, current) in enumerate(build_steps, start=1):
        _page_base(pdf)
        _header(pdf, f"Step {step}", step)
        callouts = Counter(
            (int(brick.get("width", 1)), int(brick.get("depth", 1)), str(brick.get("color", "#C84832")))
            for brick in current
        )
        x = 16 * mm
        for (width, depth, color), count in callouts.items():
            sample = {"width": width, "depth": depth, "color": color}
            icon_scale = min(4.8, 15 * mm / max(width, 1))
            _draw_brick_icon(pdf, x, page_height - 35 * mm, sample, icon_scale)
            icon_width = width * icon_scale
            pdf.setFillColor(INK)
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(x + icon_width + 2.2 * mm, page_height - 31 * mm, f"×{count}")
            x += max(27 * mm, icon_width + 9 * mm)

        # The large red arrow means place this row's parts onto the ghosted support.
        arrow_x = page_width / 2
        arrow_top, arrow_tip = page_height - 49 * mm, page_height - 65 * mm
        pdf.setStrokeColor(RED)
        pdf.setFillColor(RED)
        pdf.setLineWidth(3)
        pdf.line(arrow_x, arrow_top, arrow_x, arrow_tip + 3 * mm)
        arrow = pdf.beginPath()
        arrow.moveTo(arrow_x - 4 * mm, arrow_tip + 5 * mm)
        arrow.lineTo(arrow_x + 4 * mm, arrow_tip + 5 * mm)
        arrow.lineTo(arrow_x, arrow_tip)
        arrow.close()
        pdf.drawPath(arrow, fill=1, stroke=0)

        _draw_plan(pdf, bricks, placed, current, (17 * mm, 36 * mm, 160 * mm, 170 * mm), True)
        _draw_layer_meter(pdf, layer_indexes[z], len(layers))
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawRightString(page_width - 14 * mm, 12 * mm, f"LAYER {z + 1}  ·  {len(current)} PIECES")
        pdf.showPage()
        placed.extend(current)

    pdf.save()
