from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


BRICK_TYPES = [
    (4, 2, "3001"),
    (3, 2, "3002"),
    (2, 2, "3003"),
    (4, 1, "3010"),
    (3, 1, "3622"),
    (2, 1, "3004"),
    (1, 1, "3005"),
]


def _candidates(z: int) -> list[tuple[int, int, str]]:
    """Alternate preferred orientation by layer to improve overlap between seams."""
    pieces: list[tuple[int, int, str]] = []
    for width, depth, part in BRICK_TYPES:
        orientations = [(width, depth)]
        if width != depth:
            orientations.append((depth, width))
        if z % 2:
            orientations.reverse()
        pieces.extend((w, d, part) for w, d in orientations)
    return pieces


def greedy_pack(voxels: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Cover every occupied voxel exactly once with the largest fitting brick."""
    layers: dict[tuple[int, str], set[tuple[int, int]]] = defaultdict(set)
    for voxel in voxels:
        layers[(int(voxel["z"]), voxel.get("color", "#C84832"))].add(
            (int(voxel["x"]), int(voxel["y"]))
        )

    bricks: list[dict[str, Any]] = []
    bom: Counter[str] = Counter()
    for (z, color), cells in sorted(layers.items()):
        remaining = set(cells)
        for y, x in sorted((y, x) for x, y in cells):
            if (x, y) not in remaining:
                continue
            placed = None
            for width, depth, part in _candidates(z):
                footprint = {
                    (x + dx, y + dy)
                    for dx in range(width)
                    for dy in range(depth)
                }
                if footprint <= remaining:
                    placed = (width, depth, part, footprint)
                    break
            assert placed is not None
            width, depth, part, footprint = placed
            remaining.difference_update(footprint)
            label = f"{min(width, depth)}x{max(width, depth)}"
            bricks.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "width": width,
                    "depth": depth,
                    "height": 1,
                    "color": color,
                    "part_id": part,
                    "name": f"Brick {label}",
                }
            )
            bom[label] += 1
    return bricks, dict(sorted(bom.items()))


def dimensions(items: list[dict[str, Any]]) -> dict[str, int]:
    if not items:
        return {"width": 0, "depth": 0, "height": 0}
    return {
        "width": max(int(i["x"]) + int(i.get("width", 1)) for i in items),
        "depth": max(int(i["y"]) + int(i.get("depth", 1)) for i in items),
        "height": max(int(i["z"]) + int(i.get("height", 1)) for i in items),
    }
