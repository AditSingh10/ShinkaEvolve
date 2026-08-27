from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "autoresearch"
VOCAB_SIZE = 256


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator():
    return _load_module("autoresearch_eval", EXAMPLE_DIR / "evaluate.py")


@pytest.fixture
def instance():
    return {
        "instance_id": "unit-pattern",
        "training_bytes": b"abababababab",
        "validation_contexts": [b"a", b"ab"],
        "vocab_size": VOCAB_SIZE,
        "seed": 123,
        "limits": {
            "max_runtime_seconds": 0.5,
            "max_context_bytes": 32,
            "max_training_bytes": 1_000,
        },
    }


@pytest.fixture
def targets():
    return bytes([ord("b"), ord("a")])


def _uniform_rows(count: int) -> list[list[float]]:
    return [[1.0 / VOCAB_SIZE] * VOCAB_SIZE for _ in range(count)]


def _target_rows(targets: bytes, target_probability: float) -> list[list[float]]:
    other_probability = (1.0 - target_probability) / (VOCAB_SIZE - 1)
    rows = []
    for target in targets:
        row = [other_probability] * VOCAB_SIZE
        row[target] = target_probability
        rows.append(row)
    return rows


def _assert_invalid(evaluator, instance, result, message_fragment: str):
    valid, error, stats, probabilities = evaluator.validate_predictions(
        instance, result
    )
    assert valid is False
    assert message_fragment.lower() in error.lower()
    assert stats["valid"] is False
    assert probabilities is None


def _write_candidate(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_valid_probability_predictions_are_normalized(evaluator, instance):
    valid, error, stats, probabilities = evaluator.validate_predictions(
        instance, {"probabilities": _uniform_rows(2)}
    )

    assert valid is True
    assert error == ""
    assert stats["prediction_shape"] == [2, VOCAB_SIZE]
    assert stats["prediction_format"] == "probabilities"
    assert stats["all_finite"] is True
    assert stats["max_normalization_error"] == pytest.approx(0.0)
    assert probabilities is not None


def test_valid_log_probabilities_are_converted(evaluator, instance):
    log_probability = -math.log(VOCAB_SIZE)

    valid, error, stats, probabilities = evaluator.validate_predictions(
        instance, {"log_probabilities": [[log_probability] * VOCAB_SIZE] * 2}
    )

    assert valid is True
    assert error == ""
    assert stats["prediction_format"] == "log_probabilities"
    assert probabilities is not None
    assert probabilities[0][0] == pytest.approx(1.0 / VOCAB_SIZE)


@pytest.mark.parametrize(
    "result",
    [None, 1.5, "loss=0", [0.1], {"validation_loss": 0.0}],
)
def test_malformed_return_value_is_rejected(evaluator, instance, result):
    _assert_invalid(evaluator, instance, result, "prediction")


def test_missing_predictions_are_rejected(evaluator, instance):
    _assert_invalid(evaluator, instance, {"metadata": {}}, "prediction")


@pytest.mark.parametrize("count", [1, 3])
def test_wrong_number_of_predictions_is_rejected(evaluator, instance, count):
    _assert_invalid(
        evaluator,
        instance,
        {"probabilities": _uniform_rows(count)},
        "exactly 2",
    )


@pytest.mark.parametrize("width", [255, 257])
def test_wrong_vocabulary_size_is_rejected(evaluator, instance, width):
    rows = [[1.0 / width] * width for _ in range(2)]
    _assert_invalid(evaluator, instance, {"probabilities": rows}, "256")


def test_inconsistent_prediction_row_lengths_are_rejected(evaluator, instance):
    rows = [_uniform_rows(1)[0], [1.0 / 255] * 255]
    _assert_invalid(evaluator, instance, {"probabilities": rows}, "256")


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_prediction_value_is_rejected(evaluator, instance, bad_value):
    rows = _uniform_rows(2)
    rows[0][17] = bad_value
    _assert_invalid(evaluator, instance, {"probabilities": rows}, "finite")


def test_negative_probability_is_rejected(evaluator, instance):
    rows = _uniform_rows(2)
    rows[0][0] = -0.1
    rows[0][1] += 0.1 + 1.0 / VOCAB_SIZE
    _assert_invalid(evaluator, instance, {"probabilities": rows}, "nonnegative")


def test_unnormalized_probability_distribution_is_rejected(evaluator, instance):
    _assert_invalid(
        evaluator,
        instance,
        {"probabilities": [[0.0] * VOCAB_SIZE for _ in range(2)]},
        "normalized",
    )


def test_unnormalized_log_probability_distribution_is_rejected(evaluator, instance):
    _assert_invalid(
        evaluator,
        instance,
        {"log_probabilities": [[0.0] * VOCAB_SIZE for _ in range(2)]},
        "log-sum-exp",
    )


def test_loss_is_computed_from_hidden_targets(evaluator, instance, targets):
    result = {"probabilities": _target_rows(targets, 0.5)}

    metrics, correct, error = evaluator.score_predictions(instance, targets, result)

    assert correct is True
    assert error == ""
    assert metrics["mean_validation_loss"] == pytest.approx(math.log(2.0))
    assert metrics["mean_validation_bits_per_byte"] == pytest.approx(1.0)
    assert metrics["score"] == pytest.approx(0.5)


def test_fake_reported_metrics_are_ignored(evaluator, instance, targets):
    result = {
        "probabilities": _uniform_rows(2),
        "validation_loss": 0.0,
        "bits_per_byte": 0.0,
        "combined_score": 999.0,
        "correct": True,
        "runtime_seconds": 0.0,
    }

    metrics, correct, _ = evaluator.score_predictions(instance, targets, result)

    assert correct is True
    assert metrics["mean_validation_loss"] == pytest.approx(math.log(VOCAB_SIZE))
    assert metrics["mean_validation_bits_per_byte"] == pytest.approx(8.0)
    assert metrics["score"] == pytest.approx(1.0 / 9.0)


def test_hidden_validation_targets_are_not_exposed(evaluator, instance, tmp_path):
    candidate = tmp_path / "visibility_candidate.py"
    _write_candidate(
        candidate,
        """
def run_autoresearch(instance, seed):
    forbidden = {"targets", "validation_targets", "target_bytes", "validation_bytes"}
    assert forbidden.isdisjoint(instance)
    assert all(isinstance(context, bytes) for context in instance["validation_contexts"])
    rows = len(instance["validation_contexts"])
    width = instance["vocab_size"]
    return {"probabilities": [[1.0 / width] * width for _ in range(rows)]}
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        str(candidate), [(instance, bytes([98, 97]))]
    )

    assert correct is True
    assert error == ""
    assert metrics["public"]["num_valid_instances"] == 1


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        ('instance["training_bytes"] = b"leaked"', "training"),
        ('instance["limits"]["max_context_bytes"] = 999999', "limits"),
    ],
)
def test_candidate_input_mutation_is_detected(
    evaluator, instance, tmp_path, mutation, expected_field
):
    candidate = tmp_path / f"mutating_{expected_field}.py"
    _write_candidate(
        candidate,
        f"""
def run_autoresearch(instance, seed):
    {mutation}
    width = instance["vocab_size"]
    return {{"probabilities": [[1.0 / width] * width for _ in range(2)]}}
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        str(candidate), [(instance, bytes([98, 97]))]
    )

    assert correct is False
    assert "mutated" in error.lower()
    assert metrics["combined_score"] == 0.0
    assert instance["training_bytes"] == b"abababababab"
    assert instance["limits"]["max_context_bytes"] == 32


def test_candidate_timeout_is_recorded(evaluator, instance, tmp_path):
    timed_instance = copy.deepcopy(instance)
    timed_instance["limits"]["max_runtime_seconds"] = 0.05
    candidate = tmp_path / "slow_candidate.py"
    _write_candidate(
        candidate,
        """
import time

def run_autoresearch(instance, seed):
    time.sleep(2.0)
    return {"probabilities": []}
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        str(candidate), [(timed_instance, bytes([98, 97]))]
    )

    assert correct is False
    assert "time" in error.lower()
    assert metrics["combined_score"] == 0.0


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM behavior is POSIX-specific")
def test_timeout_force_kills_candidate_that_ignores_sigterm(
    evaluator, instance, tmp_path
):
    timed_instance = copy.deepcopy(instance)
    timed_instance["limits"]["max_runtime_seconds"] = 0.1
    pid_path = tmp_path / "candidate.pid"
    candidate = tmp_path / "sigterm_ignoring_candidate.py"
    _write_candidate(
        candidate,
        f"""
import os
from pathlib import Path
import signal
import time

def run_autoresearch(instance, seed):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path({str(pid_path)!r}).write_text(str(os.getpid()))
    while True:
        time.sleep(0.1)
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        str(candidate), [(timed_instance, bytes([98, 97]))]
    )

    assert pid_path.is_file()
    candidate_pid = int(pid_path.read_text(encoding="utf-8"))
    matching_children = [
        child
        for child in evaluator.multiprocessing.active_children()
        if child.pid == candidate_pid
    ]
    try:
        assert matching_children == []
        assert correct is False
        assert "time" in error.lower()
        assert metrics["combined_score"] == 0.0
    finally:
        for child in matching_children:
            child.kill()
            child.join(timeout=1.0)


def test_spawn_mode_supports_dynamically_loaded_evaluator(
    evaluator, instance, tmp_path, monkeypatch
):
    candidate = tmp_path / "spawn_candidate.py"
    _write_candidate(
        candidate,
        """
def run_autoresearch(instance, seed):
    rows = len(instance["validation_contexts"])
    width = instance["vocab_size"]
    return {"probabilities": [[1.0 / width] * width for _ in range(rows)]}
""",
    )
    monkeypatch.setattr(
        evaluator.multiprocessing, "get_all_start_methods", lambda: ["spawn"]
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        str(candidate), [(instance, bytes([98, 97]))]
    )

    assert correct is True
    assert error == ""
    assert metrics["public"]["num_valid_instances"] == 1


def test_candidate_exception_is_recorded(evaluator, instance, tmp_path):
    candidate = tmp_path / "raising_candidate.py"
    _write_candidate(
        candidate,
        """
def run_autoresearch(instance, seed):
    raise RuntimeError("deliberate autoresearch failure")
""",
    )

    metrics, correct, error = evaluator.evaluate_candidate(
        str(candidate), [(instance, bytes([98, 97]))]
    )

    assert correct is False
    assert "deliberate autoresearch failure" in error
    assert metrics["combined_score"] == 0.0


def test_baseline_is_valid_and_deterministic(instance):
    baseline = _load_module("autoresearch_initial", EXAMPLE_DIR / "initial.py")

    first = baseline.run_autoresearch(copy.deepcopy(instance), seed=123)
    second = baseline.run_autoresearch(copy.deepcopy(instance), seed=123)

    assert first == second
    assert len(first["probabilities"]) == 2
    assert all(len(row) == VOCAB_SIZE for row in first["probabilities"])
    assert all(sum(row) == pytest.approx(1.0) for row in first["probabilities"])


def test_score_prefers_lower_validation_loss(evaluator, instance, targets):
    better = {"probabilities": _target_rows(targets, 0.8)}
    worse = {"probabilities": _target_rows(targets, 0.4)}

    better_metrics, better_correct, _ = evaluator.score_predictions(
        instance, targets, better
    )
    worse_metrics, worse_correct, _ = evaluator.score_predictions(
        instance, targets, worse
    )

    assert better_correct is True
    assert worse_correct is True
    assert (
        better_metrics["mean_validation_loss"] < worse_metrics["mean_validation_loss"]
    )
    assert better_metrics["score"] > worse_metrics["score"]


def test_end_to_end_baseline_produces_metrics_and_correct_json(evaluator, tmp_path):
    results_dir = tmp_path / "results"

    evaluator.main(str(EXAMPLE_DIR / "initial.py"), str(results_dir))

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    correct = json.loads((results_dir / "correct.json").read_text(encoding="utf-8"))
    assert correct == {"correct": True, "error": ""}
    assert metrics["combined_score"] > 0.0
    assert metrics["public"]["num_instances"] > 1
    assert (
        metrics["public"]["num_valid_instances"] == metrics["public"]["num_instances"]
    )
    assert math.isfinite(metrics["public"]["mean_validation_loss"])
    assert math.isfinite(metrics["public"]["mean_validation_bits_per_byte"])
    assert metrics["public"]["runtime_seconds"] >= 0.0
