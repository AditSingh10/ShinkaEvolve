"""Deterministic shelf baseline for 2D rectangle strip packing."""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
def pack_rectangles(instance: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Pack all rectangles using a simple rotation-aware shelf heuristic."""
    del seed  # The baseline is deterministic; evolved strategies may use it.
    strip_width = float(instance["strip_width"])
    rotation_enabled = instance["rotation_enabled"]
    rectangles = sorted(
        instance["rectangles"],
        key=lambda rectangle: (
            -max(rectangle["width"], rectangle["height"]),
            -(rectangle["width"] * rectangle["height"]),
            rectangle["id"],
        ),
    )

    placements: list[dict[str, Any]] = []
    shelf_y = 0.0
    shelf_x = 0.0
    shelf_height = 0.0

    for rectangle in rectangles:
        width = float(rectangle["width"])
        height = float(rectangle["height"])
        orientations = [(False, width, height)]
        if rotation_enabled and width != height:
            orientations.append((True, height, width))
        orientations = [item for item in orientations if item[1] <= strip_width]

        remaining_width = strip_width - shelf_x
        fitting = [item for item in orientations if item[1] <= remaining_width]
        if fitting:
            rotated, placed_width, placed_height = min(
                fitting,
                key=lambda item: (
                    max(shelf_height, item[2]) - shelf_height,
                    remaining_width - item[1],
                    item[2],
                    item[0],
                ),
            )
        else:
            shelf_y += shelf_height
            shelf_x = 0.0
            shelf_height = 0.0
            rotated, placed_width, placed_height = min(
                orientations,
                key=lambda item: (item[2], item[1], item[0]),
            )

        placements.append(
            {
                "id": rectangle["id"],
                "x": shelf_x,
                "y": shelf_y,
                "rotated": rotated,
            }
        )
        shelf_x += placed_width
        shelf_height = max(shelf_height, placed_height)

    return placements


# EVOLVE-BLOCK-END


def run_packing(instance: dict[str, Any], seed: int) -> dict[str, Any]:
    """Run the candidate packer using the evaluator-provided instance."""
    return {"placements": pack_rectangles(instance, seed)}
