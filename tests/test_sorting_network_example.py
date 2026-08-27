from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "sorting_network"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator():
    return _load_module("sorting_network_eval", EXAMPLE_DIR / "evaluate.py")


@pytest.fixture
def instance():
    return {
        "instance_id": "unit-test-4",
        "n_wires": 4,
        "objective": "depth_then_size",
        "seed": 123,
    }


@pytest.fixture
def network_4():
    return {
        "network": [
            [(0, 1), (2, 3)],
            [(0, 2), (1, 3)],
            [(1, 2)],
        ]
    }


def _assert_invalid(evaluator, instance, result, message_fragment: str):
    valid, error, stats = evaluator.validate_network(instance, result)
    assert valid is False
    assert message_fragment.lower() in error.lower()
    assert stats["valid"] is False


def _write_candidate(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_boolean_inputs_are_generated_exhaustively(evaluator):
    assert evaluator.generate_boolean_inputs(2) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_apply_network_uses_fixed_layered_comparators(evaluator, network_4):
    output = evaluator.apply_network([1, 0, 1, 0], network_4["network"])

    assert output == [0, 0, 1, 1]


def test_correctness_helper_reports_counterexample(evaluator):
    check = evaluator.check_network(3, [[(0, 1)]])

    assert check["num_boolean_inputs_tested"] == 8
    assert check["failed_input_count"] > 0
    assert check["first_failed_input"] == [0, 1, 0]
    assert check["first_failed_output"] == [0, 1, 0]


def test_valid_network_for_two_wires(evaluator):
    two_wire_instance = {
        "instance_id": "two",
        "n_wires": 2,
        "objective": "depth_then_size",
        "seed": 2,
    }

    valid, error, stats = evaluator.validate_network(
        two_wire_instance, {"network": [[(0, 1)]]}
    )

    assert valid is True
    assert error == ""
    assert stats["depth"] == 1
    assert stats["comparator_count"] == 1
    assert stats["num_boolean_inputs_tested"] == 4
    assert stats["failed_input_count"] == 0


def test_valid_network_for_four_wires(evaluator, instance, network_4):
    valid, error, stats = evaluator.validate_network(instance, network_4)

    assert valid is True
    assert error == ""
    assert stats["depth"] == 3
    assert stats["comparator_count"] == 5
    assert stats["num_boolean_inputs_tested"] == 16
    assert stats["failed_input_count"] == 0


def test_incorrect_network_fails_exhaustive_check(evaluator, instance):
    valid, error, stats = evaluator.validate_network(
        instance, {"network": [[(0, 1), (2, 3)]]}
    )

    assert valid is False
    assert "does not sort" in error.lower()
    assert stats["num_boolean_inputs_tested"] == 16
    assert stats["failed_input_count"] > 0
    assert stats["first_failed_input"] is not None
    assert stats["first_failed_output"] is not None


@pytest.mark.parametrize(
    "result",
    [None, 42, [[(0, 1)]], {"reported_correct": True}],
)
def test_malformed_return_value_is_rejected(evaluator, instance, result):
    _assert_invalid(evaluator, instance, result, "network")


def test_malformed_layer_structure_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [None]}, "layer")


def test_malformed_comparator_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [["0,1"]]}, "comparator")


def test_comparator_with_too_few_fields_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[(0,)]]}, "exactly two")


def test_comparator_with_too_many_fields_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[(0, 1, 2)]]}, "exactly two")


def test_negative_wire_index_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[(-1, 1)]]}, "nonnegative")


def test_out_of_range_wire_index_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[(0, 4)]]}, "range")


@pytest.mark.parametrize("bad_wire", [1.0, "1", None, True])
def test_non_integer_wire_index_is_rejected(evaluator, instance, bad_wire):
    _assert_invalid(evaluator, instance, {"network": [[(0, bad_wire)]]}, "integer")


def test_comparator_cannot_compare_wire_with_itself(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[(2, 2)]]}, "itself")


def test_reversed_comparator_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[(3, 1)]]}, "less than")


def test_overlapping_comparators_in_layer_are_rejected(evaluator, instance):
    _assert_invalid(
        evaluator,
        instance,
        {"network": [[(0, 1), (1, 2)]]},
        "more than one comparator",
    )


def test_empty_layer_is_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[]]}, "empty layer")


def test_empty_network_is_invalid_for_multiple_wires(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": []}, "empty network")


def test_sorted_outputs_cannot_replace_comparator_network(evaluator, instance):
    _assert_invalid(evaluator, instance, {"network": [[0, 1, 2, 3]]}, "comparator")


def test_fake_reported_metrics_are_ignored(evaluator, instance, network_4):
    result = copy.deepcopy(network_4)
    result.update(
        {
            "reported_depth": 0,
            "reported_comparator_count": 0,
            "reported_correct": False,
            "failed_input_count": 99,
            "combined_score": 1_000_000.0,
        }
    )

    valid, _, stats = evaluator.validate_network(instance, result)

    assert valid is True
    assert stats["depth"] == 3
    assert stats["comparator_count"] == 5
    assert stats["failed_input_count"] == 0


def test_input_mutation_is_detected(evaluator, instance, tmp_path):
    candidate = tmp_path / "mutating_candidate.py"
    _write_candidate(
        candidate,
        """
def run_sorting_network(instance, seed):
    instance["n_wires"] = 2
    return {"network": [[(0, 1)]]}
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(str(candidate), [instance])

    assert correct is False
    assert "mutated" in error.lower()
    assert metrics["combined_score"] == 0.0
    assert instance["n_wires"] == 4


def test_baseline_is_deterministic_under_fixed_seed(instance):
    baseline = _load_module("sorting_network_initial", EXAMPLE_DIR / "initial.py")

    first = baseline.run_sorting_network(copy.deepcopy(instance), seed=123)
    second = baseline.run_sorting_network(copy.deepcopy(instance), seed=123)

    assert first == second


def test_score_prefers_lower_depth_for_correct_networks(evaluator):
    instance = {
        "instance_id": "depth-score",
        "n_wires": 3,
        "objective": "depth_then_size",
        "seed": 3,
    }
    shallow = {"network": [[(0, 1)], [(1, 2)], [(0, 1)]]}
    deep = {"network": [[(0, 1)], [(1, 2)], [(0, 1)], [(0, 1)]]}

    shallow_metrics, shallow_correct, _ = evaluator.score_result(instance, shallow)
    deep_metrics, deep_correct, _ = evaluator.score_result(instance, deep)

    assert shallow_correct is True
    assert deep_correct is True
    assert shallow_metrics["depth"] == 3
    assert deep_metrics["depth"] == 4
    assert shallow_metrics["score"] > deep_metrics["score"]


def test_score_prefers_fewer_comparators_at_equal_depth(evaluator, instance, network_4):
    smaller = network_4
    larger = copy.deepcopy(network_4)
    larger["network"][2].append((0, 3))

    small_metrics, small_correct, _ = evaluator.score_result(instance, smaller)
    large_metrics, large_correct, _ = evaluator.score_result(instance, larger)

    assert small_correct is True
    assert large_correct is True
    assert small_metrics["depth"] == large_metrics["depth"] == 3
    assert small_metrics["comparator_count"] == 5
    assert large_metrics["comparator_count"] == 6
    assert small_metrics["score"] > large_metrics["score"]


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
def run_sorting_network(instance, seed):
    raise RuntimeError("deliberate sorting network failure")
""",
    )
    results_dir = tmp_path / "results"

    evaluator.main(str(candidate), str(results_dir))

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    assert correct["correct"] is False
    assert "deliberate sorting network failure" in correct["error"]
    assert metrics["combined_score"] == 0.0
