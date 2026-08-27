"""Evaluator for the lightweight Autoresearch byte-prediction example."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import multiprocessing
import os
import sys
from numbers import Real
from pathlib import Path
from typing import Any


VOCAB_SIZE = 256
NORMALIZATION_TOLERANCE = 1e-6


def _case(
    instance_id: str,
    training_bytes: bytes,
    examples: list[tuple[bytes, int]],
    seed: int,
) -> tuple[dict[str, Any], bytes]:
    """Build one public candidate input and its evaluator-only target bytes."""
    contexts = [context for context, _ in examples]
    targets = bytes(target for _, target in examples)
    return (
        {
            "instance_id": instance_id,
            "training_bytes": training_bytes,
            "validation_contexts": contexts,
            "vocab_size": VOCAB_SIZE,
            "seed": seed,
            "limits": {
                "max_runtime_seconds": 1.0,
                "max_context_bytes": 64,
                "max_training_bytes": 20_000,
            },
        },
        targets,
    )


def evaluation_cases() -> list[tuple[dict[str, Any], bytes]]:
    """Return small deterministic cases while keeping targets out of candidate inputs."""
    return [
        _case(
            "cyclic-pattern",
            b"abcabcabcabc|bcabcabcabca|cabcabcabcab|" * 24,
            [
                (b"abcab", ord("c")),
                (b"bcabc", ord("a")),
                (b"cabcab", ord("c")),
                (b"abcabc", ord("a")),
            ],
            104729,
        ),
        _case(
            "field-grammar",
            (b"name:alice;color:amber\nname:bob;color:blue\nname:carol;color:cyan\n")
            * 18,
            [
                (b"record name:ali", ord("c")),
                (b"record color:blu", ord("e")),
                (b"record name:bo", ord("b")),
                (b"record color:cya", ord("n")),
            ],
            130363,
        ),
        _case(
            "local-language",
            (
                b"the small fox runs home. the small owl flies home. "
                b"the bright fox sleeps well. the bright owl sees well. "
            )
            * 16,
            [
                (b"a small fox runs ho", ord("m")),
                (b"a bright owl sees we", ord("l")),
                (b"the small owl fl", ord("i")),
                (b"the bright fox sle", ord("e")),
            ],
            155921,
        ),
    ]


def _instance_data(instance: Any) -> tuple[int, int, float]:
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a mapping.")
    training_bytes = instance.get("training_bytes")
    if not isinstance(training_bytes, bytes) or not training_bytes:
        raise ValueError("training_bytes must be a nonempty bytes value.")
    contexts = instance.get("validation_contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("validation_contexts must be a nonempty list.")
    if any(not isinstance(context, bytes) for context in contexts):
        raise ValueError("Every validation context must be bytes.")
    vocab_size = instance.get("vocab_size")
    if vocab_size != VOCAB_SIZE or isinstance(vocab_size, bool):
        raise ValueError(f"vocab_size must be exactly {VOCAB_SIZE}.")
    seed = instance.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    limits = instance.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("limits must be a mapping.")
    runtime = limits.get("max_runtime_seconds")
    max_context = limits.get("max_context_bytes")
    max_training = limits.get("max_training_bytes")
    if (
        not isinstance(runtime, Real)
        or isinstance(runtime, bool)
        or not math.isfinite(float(runtime))
        or runtime <= 0
    ):
        raise ValueError("max_runtime_seconds must be positive and finite.")
    if not isinstance(max_context, int) or isinstance(max_context, bool):
        raise ValueError("max_context_bytes must be an integer.")
    if not isinstance(max_training, int) or isinstance(max_training, bool):
        raise ValueError("max_training_bytes must be an integer.")
    if max_context <= 0 or any(len(context) > max_context for context in contexts):
        raise ValueError("A validation context exceeds max_context_bytes.")
    if max_training <= 0 or len(training_bytes) > max_training:
        raise ValueError("training_bytes exceeds max_training_bytes.")
    return len(contexts), vocab_size, float(runtime)


def _empty_stats() -> dict[str, Any]:
    return {
        "valid": False,
        "prediction_shape": None,
        "prediction_format": None,
        "all_finite": False,
        "max_normalization_error": None,
        "mean_validation_loss": None,
        "mean_validation_bits_per_byte": None,
        "score": 0.0,
    }


def validate_predictions(
    instance: Any, result: Any
) -> tuple[bool, str, dict[str, Any], list[list[float]] | None]:
    """Validate and normalize a candidate's probability distributions."""
    stats = _empty_stats()

    def invalid(message: str):
        return False, message, stats, None

    try:
        prediction_count, vocab_size, _ = _instance_data(instance)
    except ValueError as exc:
        return invalid(f"Invalid evaluator instance: {exc}")
    if not isinstance(result, dict):
        return invalid("Result must be a mapping containing predictions.")
    has_probabilities = "probabilities" in result
    has_log_probabilities = "log_probabilities" in result
    if has_probabilities == has_log_probabilities:
        return invalid(
            "Result must contain exactly one prediction field: probabilities or "
            "log_probabilities."
        )
    prediction_format = "probabilities" if has_probabilities else "log_probabilities"
    rows = result[prediction_format]
    if not isinstance(rows, (list, tuple)):
        return invalid("Predictions must be a list or tuple of distributions.")
    if len(rows) != prediction_count:
        return invalid(
            f"Predictions must contain exactly {prediction_count} distributions."
        )

    probabilities: list[list[float]] = []
    max_error = 0.0
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            return invalid(f"Prediction row {row_index} must be a list or tuple.")
        if len(row) != vocab_size:
            return invalid(
                f"Prediction row {row_index} must contain exactly {vocab_size} values."
            )
        numeric_row = []
        for value in row:
            if not isinstance(value, Real) or isinstance(value, bool):
                return invalid("Every prediction value must be a real finite number.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                return invalid("Every prediction value must be finite.")
            numeric_row.append(numeric_value)

        if prediction_format == "probabilities":
            if any(value < 0.0 for value in numeric_row):
                return invalid("Every probability must be nonnegative.")
            total = math.fsum(numeric_row)
            error = abs(total - 1.0)
            if error > NORMALIZATION_TOLERANCE:
                return invalid("Every probability distribution must be normalized.")
            normalized_row = numeric_row
        else:
            maximum = max(numeric_row)
            log_sum_exp = maximum + math.log(
                math.fsum(math.exp(value - maximum) for value in numeric_row)
            )
            error = abs(log_sum_exp)
            if error > NORMALIZATION_TOLERANCE:
                return invalid(
                    "Every log-probability row must have log-sum-exp equal to zero."
                )
            normalized_row = [math.exp(value) for value in numeric_row]
        max_error = max(max_error, error)
        probabilities.append(normalized_row)

    stats.update(
        {
            "valid": True,
            "prediction_shape": [prediction_count, vocab_size],
            "prediction_format": prediction_format,
            "all_finite": True,
            "max_normalization_error": max_error,
        }
    )
    return True, "", stats, probabilities


def score_predictions(
    instance: Any, targets: bytes, result: Any
) -> tuple[dict[str, Any], bool, str]:
    """Compute loss and score from evaluator-owned target bytes."""
    valid, error, stats, probabilities = validate_predictions(instance, result)
    if not valid or probabilities is None:
        return stats, False, error
    if not isinstance(targets, bytes) or len(targets) != len(probabilities):
        stats["valid"] = False
        return stats, False, "Evaluator targets do not match prediction count."

    target_probabilities = [
        row[target] for row, target in zip(probabilities, targets, strict=True)
    ]
    if any(probability <= 0.0 for probability in target_probabilities):
        stats["valid"] = False
        return stats, False, "Validation loss is non-finite due to zero target mass."
    losses = [-math.log(probability) for probability in target_probabilities]
    mean_loss = math.fsum(losses) / len(losses)
    bits_per_byte = mean_loss / math.log(2.0)
    if not math.isfinite(mean_loss) or not math.isfinite(bits_per_byte):
        stats["valid"] = False
        return stats, False, "Validation loss must be finite."
    stats.update(
        {
            "mean_validation_loss": mean_loss,
            "mean_validation_bits_per_byte": bits_per_byte,
            "score": 1.0 / (1.0 + bits_per_byte),
        }
    )
    return stats, True, ""


def _worker_target():
    """Load the worker by an importable name so multiprocessing spawn can use it."""
    example_directory = str(Path(__file__).resolve().parent)
    if example_directory not in sys.path:
        sys.path.insert(0, example_directory)
    return importlib.import_module("autoresearch_worker").candidate_worker


def _stop_process(process: multiprocessing.Process) -> None:
    """Stop and reap a child, escalating when graceful termination is ignored."""
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
    if process.is_alive():
        raise RuntimeError(f"Could not stop candidate process {process.pid}.")


def _run_with_timeout(
    program_path: str, instance: dict[str, Any], timeout: float
) -> dict[str, Any]:
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_target(),
        args=(program_path, copy.deepcopy(instance), child_connection),
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(timeout):
            _stop_process(process)
            return {
                "status": "timeout",
                "error": f"Candidate exceeded time budget of {timeout:.3f} seconds.",
                "runtime_seconds": timeout,
            }
        try:
            payload = parent_connection.recv()
        except EOFError:
            payload = {
                "status": "error",
                "error": "Candidate process exited without returning predictions.",
                "runtime_seconds": 0.0,
            }
        process.join(timeout=1.0)
        if process.is_alive():
            _stop_process(process)
        return payload
    finally:
        parent_connection.close()


def _failure_metrics(error: str, num_instances: int, runtime: float) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "public": {
            "valid": False,
            "num_instances": num_instances,
            "num_valid_instances": 0,
            "runtime_seconds": runtime,
            "error": error,
        },
        "private": {},
    }


def evaluate_candidate(
    program_path: str,
    cases: list[tuple[dict[str, Any], bytes]] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    """Evaluate a candidate on isolated inputs and evaluator-owned targets."""
    evaluation_data = evaluation_cases() if cases is None else cases
    runtime_seconds = 0.0
    try:
        if not evaluation_data:
            raise ValueError("No evaluation instances were provided.")
        per_instance = []
        for instance, targets in evaluation_data:
            _, _, timeout = _instance_data(instance)
            execution = _run_with_timeout(program_path, instance, timeout)
            runtime_seconds += float(execution.get("runtime_seconds", 0.0))
            if execution.get("status") != "ok":
                raise RuntimeError(
                    execution.get("error", "Candidate execution failed.")
                )
            if execution.get("mutated"):
                raise ValueError("Candidate mutated the evaluator input instance.")

            instance_metrics, valid, error = score_predictions(
                instance, targets, execution["result"]
            )
            instance_metrics["instance_id"] = instance.get("instance_id", "unknown")
            instance_metrics["runtime_seconds"] = execution["runtime_seconds"]
            if not valid:
                raise ValueError(
                    f"Instance {instance_metrics['instance_id']} is invalid: {error}"
                )
            per_instance.append(instance_metrics)

        num_instances = len(per_instance)
        mean_loss = (
            math.fsum(item["mean_validation_loss"] for item in per_instance)
            / num_instances
        )
        mean_bits = (
            math.fsum(item["mean_validation_bits_per_byte"] for item in per_instance)
            / num_instances
        )
        metrics = {
            "combined_score": math.fsum(item["score"] for item in per_instance)
            / num_instances,
            "public": {
                "valid": True,
                "num_instances": num_instances,
                "num_valid_instances": num_instances,
                "runtime_seconds": runtime_seconds,
                "mean_validation_loss": mean_loss,
                "mean_validation_bits_per_byte": mean_bits,
                "instances": per_instance,
            },
            "private": {},
        }
        return metrics, True, ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return (
            _failure_metrics(error, len(evaluation_data), runtime_seconds),
            False,
            error,
        )


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
        description="Evaluate a lightweight Autoresearch candidate"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to a candidate defining run_autoresearch(instance, seed)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
