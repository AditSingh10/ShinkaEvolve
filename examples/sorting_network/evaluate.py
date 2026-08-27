"""Evaluator for the Sorting Network Synthesis example."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any


OBJECTIVES = {"depth_then_size", "size_then_depth"}
MAX_EXHAUSTIVE_WIRES = 10

EVALUATION_INSTANCES = [
    {
        "instance_id": "two-wires-depth",
        "n_wires": 2,
        "objective": "depth_then_size",
        "seed": 104729,
    },
    {
        "instance_id": "three-wires-size",
        "n_wires": 3,
        "objective": "size_then_depth",
        "seed": 130363,
    },
    {
        "instance_id": "four-wires-depth",
        "n_wires": 4,
        "objective": "depth_then_size",
        "seed": 155921,
    },
    {
        "instance_id": "five-wires-depth",
        "n_wires": 5,
        "objective": "depth_then_size",
        "seed": 180503,
    },
    {
        "instance_id": "six-wires-size",
        "n_wires": 6,
        "objective": "size_then_depth",
        "seed": 205019,
    },
]


def generate_boolean_inputs(n_wires: int) -> list[tuple[int, ...]]:
    """Return every Boolean input vector for a wire count."""
    return list(itertools.product((0, 1), repeat=n_wires))


def apply_network(values: list[int] | tuple[int, ...], network: list[Any]) -> list[int]:
    """Apply a fixed layered comparator network to one input vector."""
    output = list(values)
    for layer in network:
        for first, second in layer:
            if output[first] > output[second]:
                output[first], output[second] = output[second], output[first]
    return output


def check_network(n_wires: int, network: list[Any]) -> dict[str, Any]:
    """Exhaustively check a structurally valid network on Boolean inputs."""
    failed_input_count = 0
    first_failed_input = None
    first_failed_output = None
    boolean_inputs = generate_boolean_inputs(n_wires)
    for input_values in boolean_inputs:
        output = apply_network(input_values, network)
        if any(output[index] > output[index + 1] for index in range(n_wires - 1)):
            failed_input_count += 1
            if first_failed_input is None:
                first_failed_input = list(input_values)
                first_failed_output = output
    return {
        "num_boolean_inputs_tested": len(boolean_inputs),
        "failed_input_count": failed_input_count,
        "first_failed_input": first_failed_input,
        "first_failed_output": first_failed_output,
    }


def _instance_data(instance: Any) -> tuple[int, str]:
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a mapping.")
    n_wires = instance.get("n_wires")
    if not isinstance(n_wires, int) or isinstance(n_wires, bool):
        raise ValueError("n_wires must be an integer.")
    if n_wires < 2 or n_wires > MAX_EXHAUSTIVE_WIRES:
        raise ValueError(
            f"n_wires must be between 2 and {MAX_EXHAUSTIVE_WIRES} for exhaustive checking."
        )
    objective = instance.get("objective")
    if objective not in OBJECTIVES:
        raise ValueError(f"Objective must be one of {sorted(OBJECTIVES)}.")
    seed = instance.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Instance seed must be an integer.")
    return n_wires, objective


def _empty_stats(n_wires: int | None = None, objective: str | None = None):
    return {
        "valid": False,
        "n_wires": n_wires,
        "objective_mode": objective,
        "depth": 0,
        "comparator_count": 0,
        "num_boolean_inputs_tested": 0,
        "failed_input_count": 0,
        "first_failed_input": None,
        "first_failed_output": None,
        "network_cost": None,
        "score": 0.0,
    }


def validate_network(instance: Any, result: Any) -> tuple[bool, str, dict[str, Any]]:
    """Validate network structure and exhaustively prove Boolean sorting."""
    try:
        n_wires, objective = _instance_data(instance)
    except ValueError as exc:
        return False, f"Invalid evaluator instance: {exc}", _empty_stats()
    stats = _empty_stats(n_wires, objective)

    def invalid(message: str) -> tuple[bool, str, dict[str, Any]]:
        return False, message, stats

    if not isinstance(result, dict):
        return invalid("Result must be a mapping containing a network list.")
    network = result.get("network")
    if not isinstance(network, list):
        return invalid("Result network must be a list of layers.")
    if not network:
        return invalid("An empty network cannot sort more than one wire.")

    normalized_network: list[list[tuple[int, int]]] = []
    comparator_count = 0
    for layer_index, layer in enumerate(network):
        if not isinstance(layer, (list, tuple)):
            return invalid(f"Network layer {layer_index} must be a list or tuple.")
        if not layer:
            return invalid(f"Network contains an empty layer at index {layer_index}.")
        used_wires: set[int] = set()
        normalized_layer = []
        for comparator_index, comparator in enumerate(layer):
            if not isinstance(comparator, (list, tuple)):
                return invalid(
                    f"Comparator {comparator_index} in layer {layer_index} is malformed."
                )
            if len(comparator) != 2:
                return invalid(
                    "Every comparator must contain exactly two wire indices."
                )
            first, second = comparator
            if (
                not isinstance(first, int)
                or isinstance(first, bool)
                or not isinstance(second, int)
                or isinstance(second, bool)
            ):
                return invalid("Every comparator wire index must be an integer.")
            if first < 0 or second < 0:
                return invalid("Comparator wire indices must be nonnegative.")
            if first >= n_wires or second >= n_wires:
                return invalid(
                    f"Comparator wire index is outside the valid range 0..{n_wires - 1}."
                )
            if first == second:
                return invalid("A comparator cannot compare a wire with itself.")
            if first > second:
                return invalid(
                    "Comparator first index must be less than its second index."
                )
            if first in used_wires or second in used_wires:
                return invalid(
                    f"A wire appears in more than one comparator in layer {layer_index}."
                )
            used_wires.update((first, second))
            normalized_layer.append((first, second))
            comparator_count += 1
        normalized_network.append(normalized_layer)

    stats["depth"] = len(normalized_network)
    stats["comparator_count"] = comparator_count
    correctness = check_network(n_wires, normalized_network)
    stats.update(correctness)
    if stats["failed_input_count"]:
        return invalid(
            f"Network does not sort {stats['failed_input_count']} Boolean inputs."
        )

    stats["valid"] = True
    return True, "", stats


def score_result(instance: Any, result: Any) -> tuple[dict[str, Any], bool, str]:
    """Validate one network and calculate its lexicographic objective score."""
    valid, error, stats = validate_network(instance, result)
    if valid:
        depth = stats["depth"]
        size = stats["comparator_count"]
        if stats["objective_mode"] == "depth_then_size":
            network_cost = depth + size / (size + 1.0)
        else:
            network_cost = size + depth / (depth + 1.0)
        stats["network_cost"] = network_cost
        stats["score"] = 1.0 / (1.0 + network_cost)
    return stats, valid, error


def _load_candidate(program_path: str) -> Any:
    path = Path(program_path)
    spec = importlib.util.spec_from_file_location(
        f"sorting_network_candidate_{abs(hash(path.resolve()))}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_sorting_network"):
        raise AttributeError(
            "Candidate must define run_sorting_network(instance, seed)."
        )
    return module


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
    program_path: str, instances: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any], bool, str]:
    """Load a candidate and evaluate one fixed network per instance."""
    evaluation_instances = EVALUATION_INSTANCES if instances is None else instances
    runtime_seconds = 0.0
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
            start = time.perf_counter()
            result = module.run_sorting_network(
                candidate_instance, seed=instance["seed"]
            )
            runtime_seconds += time.perf_counter() - start
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
        metrics = {
            "combined_score": sum(item["score"] for item in per_instance)
            / num_instances,
            "public": {
                "valid": True,
                "num_instances": num_instances,
                "num_valid_instances": num_instances,
                "runtime_seconds": runtime_seconds,
                "mean_depth": sum(item["depth"] for item in per_instance)
                / num_instances,
                "mean_comparator_count": sum(
                    item["comparator_count"] for item in per_instance
                )
                / num_instances,
                "instances": per_instance,
            },
            "private": {},
        }
        return metrics, True, ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return (
            _failure_metrics(error, len(evaluation_instances), runtime_seconds),
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
        description="Evaluate a Sorting Network Synthesis candidate"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to a candidate defining run_sorting_network(instance, seed)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
