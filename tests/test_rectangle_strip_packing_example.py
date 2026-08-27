from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "rectangle_strip_packing"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator():
    return _load_module("rectangle_strip_packing_eval", EXAMPLE_DIR / "evaluate.py")


@pytest.fixture
def instance():
    return {
        "instance_id": "unit-test",
        "strip_width": 5.0,
        "rotation_enabled": True,
        "seed": 123,
        "rectangles": [
            {"id": 0, "width": 2.0, "height": 2.0},
            {"id": 1, "width": 3.0, "height": 1.0},
            {"id": 2, "width": 1.0, "height": 2.0},
        ],
    }


@pytest.fixture
def compact_result():
    return {
        "placements": [
            {"id": 0, "x": 0.0, "y": 0.0, "rotated": False},
            {"id": 1, "x": 2.0, "y": 0.0, "rotated": False},
            {"id": 2, "x": 2.0, "y": 1.0, "rotated": False},
        ]
    }


def _assert_invalid(evaluator, instance, result, message_fragment: str):
    valid, error, stats = evaluator.validate_placements(instance, result)
    assert valid is False
    assert message_fragment.lower() in error.lower()
    assert stats["valid"] is False


def _write_candidate(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_valid_placement_recomputes_geometry(evaluator, instance, compact_result):
    valid, error, stats = evaluator.validate_placements(instance, compact_result)

    assert valid is True
    assert error == ""
    assert stats["used_height"] == 3.0
    assert stats["total_area"] == 9.0
    assert stats["utilization"] == pytest.approx(0.6)
    assert stats["lower_bound"] == 2.0


@pytest.mark.parametrize(
    "result",
    [None, 7.5, {"reported_height": 1.0}, {"placements": "not-a-list"}],
)
def test_malformed_return_value_is_rejected(evaluator, instance, result):
    _assert_invalid(evaluator, instance, result, "placements")


def test_missing_required_placement_field_is_rejected(evaluator, instance):
    result = {
        "placements": [
            {"id": 0, "x": 0.0, "y": 0.0},
            {"id": 1, "x": 2.0, "y": 0.0, "rotated": False},
            {"id": 2, "x": 2.0, "y": 1.0, "rotated": False},
        ]
    }
    _assert_invalid(evaluator, instance, result, "fields")


def test_missing_rectangle_is_rejected(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result["placements"].pop()
    _assert_invalid(evaluator, instance, result, "missing")


def test_duplicate_rectangle_is_rejected(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result["placements"].append(copy.deepcopy(result["placements"][0]))
    _assert_invalid(evaluator, instance, result, "duplicate")


def test_unknown_rectangle_is_rejected(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result["placements"][2]["id"] = 99
    _assert_invalid(evaluator, instance, result, "unknown")


def test_negative_coordinate_is_rejected(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result["placements"][0]["x"] = -0.01
    _assert_invalid(evaluator, instance, result, "nonnegative")


def test_out_of_bounds_placement_is_rejected(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result["placements"][1]["x"] = 2.01
    _assert_invalid(evaluator, instance, result, "strip width")


def test_overlapping_rectangles_are_rejected(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result["placements"][1]["x"] = 1.0
    result["placements"][1]["y"] = 1.0
    _assert_invalid(evaluator, instance, result, "overlap")


def test_forbidden_rotation_is_rejected(evaluator, instance, compact_result):
    no_rotation_instance = copy.deepcopy(instance)
    no_rotation_instance["rotation_enabled"] = False
    result = copy.deepcopy(compact_result)
    result["placements"][0]["rotated"] = True
    _assert_invalid(evaluator, no_rotation_instance, result, "rotation")


def test_non_boolean_rotation_marker_is_rejected(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result["placements"][0]["rotated"] = 1
    _assert_invalid(evaluator, instance, result, "boolean")


def test_candidate_cannot_report_resized_dimensions(
    evaluator, instance, compact_result
):
    result = copy.deepcopy(compact_result)
    result["placements"][0]["width"] = 0.1
    _assert_invalid(evaluator, instance, result, "fields")


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinate_is_rejected(
    evaluator, instance, compact_result, bad_value
):
    result = copy.deepcopy(compact_result)
    result["placements"][0]["y"] = bad_value
    _assert_invalid(evaluator, instance, result, "finite")


def test_fake_reported_height_is_ignored(evaluator, instance, compact_result):
    result = copy.deepcopy(compact_result)
    result.update(
        {
            "reported_height": 0.01,
            "combined_score": 1_000_000,
            "valid": True,
            "utilization": 1.0,
        }
    )

    valid, _, stats = evaluator.validate_placements(instance, result)

    assert valid is True
    assert stats["used_height"] == 3.0
    assert stats["utilization"] == pytest.approx(0.6)


def test_input_mutation_is_detected(evaluator, instance, tmp_path):
    candidate = tmp_path / "mutating_candidate.py"
    _write_candidate(
        candidate,
        """
def run_packing(instance, seed):
    instance["strip_width"] = 1000.0
    return {"placements": []}
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(str(candidate), [instance])

    assert correct is False
    assert "mutated" in error.lower()
    assert metrics["combined_score"] == 0.0
    assert instance["strip_width"] == 5.0


def test_baseline_is_deterministic_under_fixed_seed(instance):
    baseline = _load_module(
        "rectangle_strip_packing_initial", EXAMPLE_DIR / "initial.py"
    )

    first = baseline.run_packing(copy.deepcopy(instance), seed=123)
    second = baseline.run_packing(copy.deepcopy(instance), seed=123)

    assert first == second


def test_score_prefers_lower_valid_height(evaluator, instance, compact_result):
    taller_result = copy.deepcopy(compact_result)
    taller_result["placements"][2]["y"] = 3.0

    compact_metrics, compact_correct, _ = evaluator.score_result(
        instance, compact_result
    )
    taller_metrics, taller_correct, _ = evaluator.score_result(instance, taller_result)

    assert compact_correct is True
    assert taller_correct is True
    assert compact_metrics["used_height"] == 3.0
    assert taller_metrics["used_height"] == 5.0
    assert compact_metrics["score"] > taller_metrics["score"]


def test_end_to_end_evaluation_produces_metrics_and_correct_json(evaluator, tmp_path):
    candidate = tmp_path / "candidate.py"
    _write_candidate(
        candidate,
        """
def run_packing(instance, seed):
    placements = []
    y = 0.0
    for rectangle in instance["rectangles"]:
        placements.append({
            "id": rectangle["id"],
            "x": 0.0,
            "y": y,
            "rotated": False,
        })
        y += rectangle["height"]
    return {"placements": placements, "reported_height": 0.0}
""",
    )
    results_dir = tmp_path / "results"

    evaluator.main(str(candidate), str(results_dir))

    metrics_path = results_dir / "metrics.json"
    correct_path = results_dir / "correct.json"
    assert metrics_path.is_file()
    assert correct_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    correct = json.loads(correct_path.read_text(encoding="utf-8"))
    assert correct == {"correct": True, "error": ""}
    assert 0.0 < metrics["combined_score"] <= 1.0
    assert metrics["public"]["num_instances"] > 1
    assert (
        metrics["public"]["num_valid_instances"] == metrics["public"]["num_instances"]
    )


def test_candidate_exception_is_recorded(evaluator, tmp_path):
    candidate = tmp_path / "raising_candidate.py"
    _write_candidate(
        candidate,
        """
def run_packing(instance, seed):
    raise RuntimeError("deliberate candidate failure")
""",
    )
    results_dir = tmp_path / "results"

    evaluator.main(str(candidate), str(results_dir))

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    assert correct["correct"] is False
    assert "deliberate candidate failure" in correct["error"]
    assert metrics["combined_score"] == 0.0
