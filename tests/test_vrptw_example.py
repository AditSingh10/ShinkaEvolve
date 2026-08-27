from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "vrptw"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator():
    return _load_module("vrptw_eval", EXAMPLE_DIR / "evaluate.py")


@pytest.fixture
def instance():
    matrix = [
        [0.0, 2.0, 4.0, 6.0, 8.0],
        [2.0, 0.0, 2.0, 4.0, 6.0],
        [4.0, 2.0, 0.0, 2.0, 4.0],
        [6.0, 4.0, 2.0, 0.0, 2.0],
        [8.0, 6.0, 4.0, 2.0, 0.0],
    ]
    return {
        "instance_id": "unit-test",
        "seed": 123,
        "vehicle_capacity": 5.0,
        "max_vehicles": 3,
        "depot": {
            "id": 0,
            "demand": 0.0,
            "service_time": 0.0,
            "time_window": [0.0, 50.0],
        },
        "customers": [
            {
                "id": 1,
                "demand": 2.0,
                "service_time": 2.0,
                "time_window": [2.0, 10.0],
            },
            {
                "id": 2,
                "demand": 3.0,
                "service_time": 1.0,
                "time_window": [0.0, 20.0],
            },
            {
                "id": 3,
                "demand": 2.0,
                "service_time": 1.0,
                "time_window": [8.0, 25.0],
            },
            {
                "id": 4,
                "demand": 2.0,
                "service_time": 1.0,
                "time_window": [0.0, 30.0],
            },
        ],
        "distance_matrix": copy.deepcopy(matrix),
        "travel_time_matrix": copy.deepcopy(matrix),
    }


@pytest.fixture
def valid_result():
    return {"routes": [[0, 1, 2, 0], [0, 3, 4, 0]]}


def _assert_invalid(evaluator, instance, result, message_fragment: str):
    valid, error, stats = evaluator.validate_routes(instance, result)
    assert valid is False
    assert message_fragment.lower() in error.lower()
    assert stats["valid"] is False


def _write_candidate(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _loose_instance(instance):
    changed = copy.deepcopy(instance)
    changed["vehicle_capacity"] = 10.0
    changed["depot"]["time_window"] = [0.0, 200.0]
    for customer in changed["customers"]:
        customer["time_window"] = [0.0, 200.0]
    return changed


def test_valid_solution_recomputes_route_metrics(evaluator, instance, valid_result):
    valid, error, stats = evaluator.validate_routes(instance, valid_result)

    assert valid is True
    assert error == ""
    assert stats["vehicles_used"] == 2
    assert stats["total_distance"] == 24.0
    assert stats["total_travel_time"] == 24.0
    assert stats["route_loads"] == [5.0, 4.0]
    assert stats["route_service_start_times"] == [[2.0, 6.0], [8.0, 11.0]]
    assert stats["route_return_times"] == [11.0, 20.0]


@pytest.mark.parametrize(
    "result",
    [None, 9.0, {"combined_score": 999.0}, {"routes": [0, 1, 0]}],
)
def test_malformed_return_value_is_rejected(evaluator, instance, result):
    _assert_invalid(evaluator, instance, result, "routes")


def test_missing_customer_is_rejected(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result["routes"][1] = [0, 3, 0]
    _assert_invalid(evaluator, instance, result, "missing")


def test_duplicate_customer_is_rejected(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result["routes"][1] = [0, 3, 4, 1, 0]
    _assert_invalid(evaluator, instance, result, "duplicate")


def test_unknown_customer_is_rejected(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result["routes"][1][2] = 99
    _assert_invalid(evaluator, instance, result, "unknown")


def test_negative_customer_id_is_rejected(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result["routes"][1][2] = -1
    _assert_invalid(evaluator, instance, result, "nonnegative")


@pytest.mark.parametrize("bad_id", [1.5, "1", None, math.nan, math.inf])
def test_non_integer_customer_id_is_rejected(evaluator, instance, valid_result, bad_id):
    result = copy.deepcopy(valid_result)
    result["routes"][0][1] = bad_id
    _assert_invalid(evaluator, instance, result, "integer")


def test_route_must_start_at_depot(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result["routes"][0][0] = 1
    _assert_invalid(evaluator, instance, result, "start")


def test_route_must_end_at_depot(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result["routes"][0][-1] = 2
    _assert_invalid(evaluator, instance, result, "end")


def test_depot_in_route_middle_is_rejected(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result["routes"][0] = [0, 1, 0, 2, 0]
    _assert_invalid(evaluator, instance, result, "middle")


def test_too_many_routes_are_rejected(evaluator, instance):
    result = {"routes": [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]]}
    _assert_invalid(evaluator, instance, result, "vehicle")


def test_capacity_violation_is_rejected(evaluator, instance):
    result = {"routes": [[0, 1, 2, 3, 0], [0, 4, 0]]}
    _assert_invalid(evaluator, instance, result, "capacity")


def test_time_window_violation_is_rejected(evaluator, instance):
    changed = copy.deepcopy(instance)
    changed["vehicle_capacity"] = 20.0
    result = {"routes": [[0, 4, 3, 2, 1, 0]]}
    _assert_invalid(evaluator, changed, result, "time window")


def test_arriving_early_waits_for_time_window(evaluator, instance, valid_result):
    valid, _, stats = evaluator.validate_routes(instance, valid_result)

    assert valid is True
    assert stats["route_arrival_times"][1][0] == 6.0
    assert stats["route_service_start_times"][1][0] == 8.0


def test_service_time_advances_route_clock(evaluator, instance, valid_result):
    changed = copy.deepcopy(instance)
    changed["customers"][0]["service_time"] = 10.0
    changed["customers"][1]["time_window"] = [0.0, 13.0]
    _assert_invalid(evaluator, changed, valid_result, "time window")


def test_fake_reported_metrics_are_ignored(evaluator, instance, valid_result):
    result = copy.deepcopy(valid_result)
    result.update(
        {
            "reported_distance": 0.0,
            "reported_vehicle_count": 0,
            "arrival_times": [[0.0]],
            "valid": True,
            "combined_score": 1_000_000.0,
        }
    )

    valid, _, stats = evaluator.validate_routes(instance, result)

    assert valid is True
    assert stats["vehicles_used"] == 2
    assert stats["total_distance"] == 24.0
    assert stats["route_service_start_times"] == [[2.0, 6.0], [8.0, 11.0]]


def test_input_mutation_is_detected(evaluator, instance, tmp_path):
    candidate = tmp_path / "mutating_candidate.py"
    _write_candidate(
        candidate,
        """
def run_vrptw(instance, seed):
    instance["vehicle_capacity"] = 1000000.0
    return {"routes": []}
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(str(candidate), [instance])

    assert correct is False
    assert "mutated" in error.lower()
    assert metrics["combined_score"] == 0.0
    assert instance["vehicle_capacity"] == 5.0


def test_baseline_is_deterministic_under_fixed_seed(instance):
    baseline = _load_module("vrptw_initial", EXAMPLE_DIR / "initial.py")

    first = baseline.run_vrptw(copy.deepcopy(instance), seed=123)
    second = baseline.run_vrptw(copy.deepcopy(instance), seed=123)

    assert first == second


def test_score_prefers_fewer_vehicles(evaluator, instance):
    changed = _loose_instance(instance)
    one_vehicle = {"routes": [[0, 2, 4, 1, 3, 0]]}
    two_vehicles = {"routes": [[0, 1, 0], [0, 2, 3, 4, 0]]}

    one_metrics, one_correct, _ = evaluator.score_result(changed, one_vehicle)
    two_metrics, two_correct, _ = evaluator.score_result(changed, two_vehicles)

    assert one_correct is True
    assert two_correct is True
    assert one_metrics["total_distance"] == 24.0
    assert two_metrics["total_distance"] == 20.0
    assert one_metrics["score"] > two_metrics["score"]


def test_score_prefers_shorter_distance_for_equal_vehicle_count(evaluator, instance):
    changed = _loose_instance(instance)
    shorter = {"routes": [[0, 1, 0], [0, 2, 3, 4, 0]]}
    longer = {"routes": [[0, 1, 2, 0], [0, 3, 4, 0]]}

    short_metrics, short_correct, _ = evaluator.score_result(changed, shorter)
    long_metrics, long_correct, _ = evaluator.score_result(changed, longer)

    assert short_correct is True
    assert long_correct is True
    assert short_metrics["vehicles_used"] == long_metrics["vehicles_used"] == 2
    assert short_metrics["total_distance"] == 20.0
    assert long_metrics["total_distance"] == 24.0
    assert short_metrics["score"] > long_metrics["score"]


def test_end_to_end_baseline_produces_metrics_and_correct_json(evaluator, tmp_path):
    results_dir = tmp_path / "results"

    evaluator.main(str(EXAMPLE_DIR / "initial.py"), str(results_dir))

    metrics_path = results_dir / "metrics.json"
    correct_path = results_dir / "correct.json"
    assert metrics_path.is_file()
    assert correct_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    correct = json.loads(correct_path.read_text(encoding="utf-8"))
    assert correct == {"correct": True, "error": ""}
    assert metrics["combined_score"] > 0.0
    assert metrics["public"]["num_instances"] > 1
    assert (
        metrics["public"]["num_valid_instances"] == metrics["public"]["num_instances"]
    )
    assert metrics["public"]["runtime_seconds"] >= 0.0


def test_candidate_exception_is_recorded(evaluator, tmp_path):
    candidate = tmp_path / "raising_candidate.py"
    _write_candidate(
        candidate,
        """
def run_vrptw(instance, seed):
    raise RuntimeError("deliberate VRPTW candidate failure")
""",
    )
    results_dir = tmp_path / "results"

    evaluator.main(str(candidate), str(results_dir))

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    assert correct["correct"] is False
    assert "deliberate VRPTW candidate failure" in correct["error"]
    assert metrics["combined_score"] == 0.0
