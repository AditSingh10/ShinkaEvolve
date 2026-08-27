"""Evaluator for the 2D rectangle strip packing example."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any


PLACEMENT_FIELDS = {"id", "x", "y", "rotated"}

EVALUATION_INSTANCES = [
    {
        "instance_id": "mixed-10-rotation",
        "strip_width": 10.0,
        "rotation_enabled": True,
        "seed": 104729,
        "rectangles": [
            {"id": 0, "width": 6.0, "height": 4.0},
            {"id": 1, "width": 4.0, "height": 6.0},
            {"id": 2, "width": 5.0, "height": 3.0},
            {"id": 3, "width": 3.0, "height": 7.0},
            {"id": 4, "width": 2.0, "height": 8.0},
            {"id": 5, "width": 8.0, "height": 2.0},
            {"id": 6, "width": 3.0, "height": 3.0},
        ],
    },
    {
        "instance_id": "fixed-orientation-12",
        "strip_width": 12.0,
        "rotation_enabled": False,
        "seed": 130363,
        "rectangles": [
            {"id": 0, "width": 7.0, "height": 4.0},
            {"id": 1, "width": 5.0, "height": 5.0},
            {"id": 2, "width": 4.0, "height": 8.0},
            {"id": 3, "width": 8.0, "height": 3.0},
            {"id": 4, "width": 3.0, "height": 6.0},
            {"id": 5, "width": 2.0, "height": 7.0},
            {"id": 6, "width": 6.0, "height": 2.0},
        ],
    },
    {
        "instance_id": "narrow-9-rotation",
        "strip_width": 9.0,
        "rotation_enabled": True,
        "seed": 155921,
        "rectangles": [
            {"id": 0, "width": 8.0, "height": 3.0},
            {"id": 1, "width": 7.0, "height": 2.0},
            {"id": 2, "width": 5.0, "height": 5.0},
            {"id": 3, "width": 4.0, "height": 6.0},
            {"id": 4, "width": 3.0, "height": 7.0},
            {"id": 5, "width": 2.0, "height": 4.0},
            {"id": 6, "width": 1.0, "height": 8.0},
            {"id": 7, "width": 3.0, "height": 3.0},
        ],
    },
    {
        "instance_id": "small-pieces-11",
        "strip_width": 11.0,
        "rotation_enabled": True,
        "seed": 180503,
        "rectangles": [
            {"id": 0, "width": 6.0, "height": 6.0},
            {"id": 1, "width": 5.0, "height": 4.0},
            {"id": 2, "width": 4.0, "height": 5.0},
            {"id": 3, "width": 3.0, "height": 3.0},
            {"id": 4, "width": 3.0, "height": 2.0},
            {"id": 5, "width": 2.0, "height": 3.0},
            {"id": 6, "width": 2.0, "height": 2.0},
            {"id": 7, "width": 1.0, "height": 7.0},
            {"id": 8, "width": 7.0, "height": 1.0},
        ],
    },
]


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _instance_geometry(
    instance: Any,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a mapping.")

    strip_width = instance.get("strip_width")
    if not _is_finite_real(strip_width) or strip_width <= 0:
        raise ValueError("Strip width must be a positive finite number.")
    if type(instance.get("rotation_enabled")) is not bool:
        raise ValueError("Instance rotation_enabled must be boolean.")
    if not isinstance(instance.get("seed"), int) or isinstance(
        instance.get("seed"), bool
    ):
        raise ValueError("Instance seed must be an integer.")

    rectangles = instance.get("rectangles")
    if not isinstance(rectangles, list) or not rectangles:
        raise ValueError("Instance rectangles must be a nonempty list.")

    by_id: dict[int, dict[str, Any]] = {}
    total_area = 0.0
    minimum_height = 0.0
    rotation_enabled = instance["rotation_enabled"]

    for rectangle in rectangles:
        if not isinstance(rectangle, dict):
            raise ValueError("Every input rectangle must be a mapping.")
        rectangle_id = rectangle.get("id")
        if (
            not isinstance(rectangle_id, int)
            or isinstance(rectangle_id, bool)
            or rectangle_id < 0
        ):
            raise ValueError("Rectangle IDs must be nonnegative integers.")
        if rectangle_id in by_id:
            raise ValueError(f"Input rectangle ID {rectangle_id} is duplicated.")

        width = rectangle.get("width")
        height = rectangle.get("height")
        if not _is_finite_real(width) or width <= 0:
            raise ValueError(
                f"Rectangle {rectangle_id} width must be positive and finite."
            )
        if not _is_finite_real(height) or height <= 0:
            raise ValueError(
                f"Rectangle {rectangle_id} height must be positive and finite."
            )

        width = float(width)
        height = float(height)
        feasible_heights = []
        if width <= strip_width:
            feasible_heights.append(height)
        if rotation_enabled and height <= strip_width:
            feasible_heights.append(width)
        if not feasible_heights:
            raise ValueError(
                f"Rectangle {rectangle_id} cannot fit within the strip width."
            )

        by_id[rectangle_id] = {
            "id": rectangle_id,
            "width": width,
            "height": height,
        }
        total_area += width * height
        minimum_height = max(minimum_height, min(feasible_heights))

    area_lower_bound = total_area / float(strip_width)
    stats = {
        "valid": False,
        "used_height": None,
        "total_area": total_area,
        "area_lower_bound": area_lower_bound,
        "geometric_lower_bound": minimum_height,
        "lower_bound": max(area_lower_bound, minimum_height),
        "utilization": 0.0,
    }
    return by_id, stats


def validate_placements(instance: Any, result: Any) -> tuple[bool, str, dict[str, Any]]:
    """Validate and independently measure a candidate placement result."""
    try:
        rectangles_by_id, stats = _instance_geometry(instance)
    except ValueError as exc:
        return (
            False,
            f"Invalid evaluator instance: {exc}",
            {
                "valid": False,
                "used_height": None,
                "total_area": 0.0,
                "area_lower_bound": 0.0,
                "geometric_lower_bound": 0.0,
                "lower_bound": 0.0,
                "utilization": 0.0,
            },
        )

    def invalid(message: str) -> tuple[bool, str, dict[str, Any]]:
        return False, message, stats

    if not isinstance(result, dict):
        return invalid("Result must be a mapping containing a placements list.")
    placements = result.get("placements")
    if not isinstance(placements, list):
        return invalid("Result placements must be a list.")

    expected_ids = set(rectangles_by_id)
    seen_ids: set[int] = set()
    geometry: list[dict[str, float | int]] = []

    for index, placement in enumerate(placements):
        if not isinstance(placement, dict):
            return invalid(f"Placement {index} must be a mapping.")
        if set(placement) != PLACEMENT_FIELDS:
            return invalid(
                f"Placement {index} fields must be exactly {sorted(PLACEMENT_FIELDS)}."
            )

        rectangle_id = placement["id"]
        if (
            not isinstance(rectangle_id, int)
            or isinstance(rectangle_id, bool)
            or rectangle_id < 0
        ):
            return invalid(f"Placement {index} has an invalid rectangle ID.")
        if rectangle_id in seen_ids:
            return invalid(f"Duplicate placement for rectangle ID {rectangle_id}.")
        if rectangle_id not in expected_ids:
            return invalid(f"Unknown rectangle ID {rectangle_id}.")
        seen_ids.add(rectangle_id)

        rotated = placement["rotated"]
        if type(rotated) is not bool:
            return invalid(f"Placement {rectangle_id} rotation marker must be boolean.")
        if rotated and not instance["rotation_enabled"]:
            return invalid(
                f"Placement {rectangle_id} uses rotation when rotation is forbidden."
            )

        x = placement["x"]
        y = placement["y"]
        if not _is_finite_real(x) or not _is_finite_real(y):
            return invalid(
                f"Placement {rectangle_id} coordinates must be finite real numbers."
            )
        x = float(x)
        y = float(y)
        if x < 0 or y < 0:
            return invalid(f"Placement {rectangle_id} coordinates must be nonnegative.")

        rectangle = rectangles_by_id[rectangle_id]
        width = rectangle["height"] if rotated else rectangle["width"]
        height = rectangle["width"] if rotated else rectangle["height"]
        if x + width > float(instance["strip_width"]):
            return invalid(f"Placement {rectangle_id} extends past the strip width.")

        geometry.append(
            {
                "id": rectangle_id,
                "left": x,
                "right": x + width,
                "bottom": y,
                "top": y + height,
            }
        )

    missing_ids = expected_ids - seen_ids
    if missing_ids:
        return invalid(f"Missing rectangle IDs: {sorted(missing_ids)}.")

    for first_index, first in enumerate(geometry):
        for second in geometry[first_index + 1 :]:
            horizontal_overlap = max(first["left"], second["left"]) < min(
                first["right"], second["right"]
            )
            vertical_overlap = max(first["bottom"], second["bottom"]) < min(
                first["top"], second["top"]
            )
            if horizontal_overlap and vertical_overlap:
                return invalid(f"Rectangles {first['id']} and {second['id']} overlap.")

    used_height = max(item["top"] for item in geometry)
    stats["valid"] = True
    stats["used_height"] = used_height
    stats["utilization"] = stats["total_area"] / (
        float(instance["strip_width"]) * used_height
    )
    return True, "", stats


def score_result(instance: Any, result: Any) -> tuple[dict[str, Any], bool, str]:
    """Return independently computed metrics for one candidate result."""
    valid, error, stats = validate_placements(instance, result)
    stats["score"] = 0.0
    if valid:
        stats["score"] = min(1.0, stats["lower_bound"] / stats["used_height"])
    return stats, valid, error


def _load_candidate(program_path: str) -> Any:
    path = Path(program_path)
    spec = importlib.util.spec_from_file_location(
        f"rectangle_strip_packing_candidate_{abs(hash(path.resolve()))}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_packing"):
        raise AttributeError("Candidate must define run_packing(instance, seed).")
    return module


def _failure_metrics(error: str, num_instances: int) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "public": {
            "valid": False,
            "num_instances": num_instances,
            "num_valid_instances": 0,
            "error": error,
        },
        "private": {},
    }


def evaluate_candidate(
    program_path: str, instances: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any], bool, str]:
    """Load and evaluate a candidate against independent instance copies."""
    evaluation_instances = EVALUATION_INSTANCES if instances is None else instances
    try:
        module = _load_candidate(program_path)
        per_instance = []
        for instance in evaluation_instances:
            candidate_instance = copy.deepcopy(instance)
            snapshot = json.dumps(
                candidate_instance,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            result = module.run_packing(
                candidate_instance,
                seed=instance["seed"],
            )
            try:
                after = json.dumps(
                    candidate_instance,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Candidate mutated the evaluator input instance."
                ) from exc
            if after != snapshot:
                raise ValueError("Candidate mutated the evaluator input instance.")

            instance_metrics, valid, error = score_result(instance, result)
            instance_metrics["instance_id"] = instance.get("instance_id", "unknown")
            if not valid:
                raise ValueError(
                    f"Instance {instance_metrics['instance_id']} is invalid: {error}"
                )
            per_instance.append(instance_metrics)

        num_instances = len(per_instance)
        if num_instances == 0:
            raise ValueError("No evaluation instances were provided.")
        combined_score = sum(item["score"] for item in per_instance) / num_instances
        metrics = {
            "combined_score": combined_score,
            "public": {
                "valid": True,
                "num_instances": num_instances,
                "num_valid_instances": num_instances,
                "mean_used_height": sum(item["used_height"] for item in per_instance)
                / num_instances,
                "mean_utilization": sum(item["utilization"] for item in per_instance)
                / num_instances,
                "instances": per_instance,
            },
            "private": {},
        }
        return metrics, True, ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return _failure_metrics(error, len(evaluation_instances)), False, error


def main(program_path: str, results_dir: str) -> None:
    """Evaluate a candidate and write ShinkaEvolve result artifacts."""
    os.makedirs(results_dir, exist_ok=True)
    metrics, correct, error = evaluate_candidate(program_path)
    results_path = Path(results_dir)
    (results_path / "metrics.json").write_text(
        json.dumps(metrics, indent=4, allow_nan=False), encoding="utf-8"
    )
    (results_path / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=4),
        encoding="utf-8",
    )

    print(f"Evaluated program: {program_path}")
    print(f"Results saved to: {results_dir}")
    print(f"Correct: {correct}")
    if error:
        print(f"Error: {error}")
    print(f"Combined score: {metrics['combined_score']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a 2D rectangle strip packing candidate"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to a Python candidate defining run_packing(instance, seed)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
