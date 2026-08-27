"""Evaluator for the Vehicle Routing Problem with Time Windows example."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import numbers
import os
import time
from pathlib import Path
from typing import Any


def _euclidean_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    return [
        [round(math.hypot(x1 - x2, y1 - y2), 6) for x2, y2 in points]
        for x1, y1 in points
    ]


def _make_instance(
    instance_id: str,
    seed: int,
    vehicle_capacity: float,
    max_vehicles: int,
    depot_point: tuple[float, float],
    depot_latest: float,
    customer_data: list[tuple[float, float, float, float, float, float]],
) -> dict[str, Any]:
    points = [depot_point] + [(item[0], item[1]) for item in customer_data]
    matrix = _euclidean_matrix(points)
    customers = [
        {
            "id": index,
            "demand": demand,
            "service_time": service_time,
            "time_window": [earliest, latest],
        }
        for index, (_, _, demand, service_time, earliest, latest) in enumerate(
            customer_data, start=1
        )
    ]
    return {
        "instance_id": instance_id,
        "seed": seed,
        "vehicle_capacity": vehicle_capacity,
        "max_vehicles": max_vehicles,
        "depot": {
            "id": 0,
            "demand": 0.0,
            "service_time": 0.0,
            "time_window": [0.0, depot_latest],
        },
        "customers": customers,
        "distance_matrix": matrix,
        "travel_time_matrix": copy.deepcopy(matrix),
    }


EVALUATION_INSTANCES = [
    _make_instance(
        "mixed-windows",
        104729,
        7.0,
        3,
        (0.0, 0.0),
        100.0,
        [
            (2.0, 1.0, 2.0, 1.0, 0.0, 24.0),
            (4.0, 1.0, 3.0, 2.0, 3.0, 34.0),
            (6.0, 2.0, 2.0, 1.0, 8.0, 45.0),
            (1.0, 5.0, 2.0, 1.0, 5.0, 36.0),
            (3.0, 6.0, 3.0, 2.0, 12.0, 52.0),
            (6.0, 6.0, 2.0, 1.0, 18.0, 64.0),
        ],
    ),
    _make_instance(
        "clustered-customers",
        130363,
        8.0,
        4,
        (5.0, 5.0),
        120.0,
        [
            (1.0, 1.0, 4.0, 1.0, 0.0, 36.0),
            (2.0, 2.0, 3.0, 1.0, 2.0, 42.0),
            (8.0, 1.0, 3.0, 2.0, 0.0, 45.0),
            (9.0, 2.0, 2.0, 1.0, 8.0, 54.0),
            (1.0, 9.0, 4.0, 2.0, 10.0, 62.0),
            (2.0, 8.0, 2.0, 1.0, 14.0, 70.0),
            (9.0, 9.0, 2.0, 1.0, 18.0, 80.0),
        ],
    ),
    _make_instance(
        "capacity-and-slack",
        155921,
        10.0,
        3,
        (0.0, 5.0),
        130.0,
        [
            (2.0, 5.0, 2.0, 1.0, 0.0, 30.0),
            (4.0, 4.0, 2.0, 2.0, 4.0, 40.0),
            (6.0, 5.0, 3.0, 1.0, 8.0, 48.0),
            (8.0, 4.0, 3.0, 2.0, 12.0, 60.0),
            (3.0, 8.0, 2.0, 1.0, 16.0, 66.0),
            (5.0, 9.0, 3.0, 1.0, 20.0, 76.0),
            (7.0, 8.0, 2.0, 2.0, 24.0, 86.0),
            (9.0, 7.0, 2.0, 1.0, 28.0, 96.0),
        ],
    ),
]


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_time_window(
    node: dict[str, Any], label: str, optional: bool = False
) -> tuple[float, float | None]:
    window = node.get("time_window")
    if window is None and optional:
        return 0.0, None
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError(f"{label} time window must contain earliest and latest.")
    earliest, latest = window
    if not _is_finite_real(earliest) or not _is_finite_real(latest):
        raise ValueError(f"{label} time window values must be finite.")
    earliest = float(earliest)
    latest = float(latest)
    if earliest > latest:
        raise ValueError(f"{label} time window earliest exceeds latest.")
    return earliest, latest


def _validate_matrix(matrix: Any, size: int, label: str) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError(f"{label} must be a square {size} by {size} matrix.")
    normalized = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{label} must be a square {size} by {size} matrix.")
        normalized_row = []
        for value in row:
            if not _is_finite_real(value) or value < 0:
                raise ValueError(f"{label} values must be finite and nonnegative.")
            normalized_row.append(float(value))
        normalized.append(normalized_row)
    return normalized


def _instance_data(instance: Any) -> dict[str, Any]:
    if not isinstance(instance, dict):
        raise ValueError("Instance must be a mapping.")
    seed = instance.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Instance seed must be an integer.")
    capacity = instance.get("vehicle_capacity")
    if not _is_finite_real(capacity) or capacity <= 0:
        raise ValueError("Vehicle capacity must be positive and finite.")
    max_vehicles = instance.get("max_vehicles")
    if (
        not isinstance(max_vehicles, int)
        or isinstance(max_vehicles, bool)
        or max_vehicles <= 0
    ):
        raise ValueError("Maximum vehicle count must be a positive integer.")

    depot = instance.get("depot")
    if not isinstance(depot, dict) or depot.get("id") != 0:
        raise ValueError("Depot must be a mapping with ID 0.")
    if not _is_finite_real(depot.get("demand")) or depot["demand"] != 0:
        raise ValueError("Depot demand must be zero.")
    if not _is_finite_real(depot.get("service_time")) or depot["service_time"] < 0:
        raise ValueError("Depot service time must be finite and nonnegative.")
    depot_earliest, depot_latest = _read_time_window(depot, "Depot", optional=True)

    customers = instance.get("customers")
    if not isinstance(customers, list) or not customers:
        raise ValueError("Customers must be a nonempty list.")
    nodes: dict[int, dict[str, Any]] = {
        0: {
            "id": 0,
            "demand": 0.0,
            "service_time": float(depot["service_time"]),
            "earliest": depot_earliest,
            "latest": depot_latest,
        }
    }
    for customer in customers:
        if not isinstance(customer, dict):
            raise ValueError("Every customer must be a mapping.")
        customer_id = customer.get("id")
        if (
            not isinstance(customer_id, int)
            or isinstance(customer_id, bool)
            or customer_id <= 0
        ):
            raise ValueError("Customer IDs must be positive integers.")
        if customer_id in nodes:
            raise ValueError(f"Customer ID {customer_id} is duplicated in the input.")
        demand = customer.get("demand")
        service_time = customer.get("service_time")
        if not _is_finite_real(demand) or demand < 0:
            raise ValueError(
                f"Customer {customer_id} demand must be finite and nonnegative."
            )
        if demand > capacity:
            raise ValueError(f"Customer {customer_id} demand exceeds vehicle capacity.")
        if not _is_finite_real(service_time) or service_time < 0:
            raise ValueError(
                f"Customer {customer_id} service time must be finite and nonnegative."
            )
        earliest, latest = _read_time_window(customer, f"Customer {customer_id}")
        nodes[customer_id] = {
            "id": customer_id,
            "demand": float(demand),
            "service_time": float(service_time),
            "earliest": earliest,
            "latest": latest,
        }

    expected_ids = set(range(1, len(customers) + 1))
    if set(nodes) - {0} != expected_ids:
        raise ValueError("Customer IDs must be contiguous integers starting at 1.")
    size = len(nodes)
    distances = _validate_matrix(
        instance.get("distance_matrix"), size, "Distance matrix"
    )
    travel_source = instance.get("travel_time_matrix", instance.get("distance_matrix"))
    travel_times = _validate_matrix(travel_source, size, "Travel-time matrix")
    maximum_distance = max(max(row) for row in distances)
    vehicle_penalty = maximum_distance * (len(customers) + max_vehicles) + 1.0
    return {
        "nodes": nodes,
        "customer_ids": expected_ids,
        "capacity": float(capacity),
        "max_vehicles": max_vehicles,
        "distances": distances,
        "travel_times": travel_times,
        "vehicle_penalty": vehicle_penalty,
    }


def _empty_stats(vehicle_penalty: float = 0.0) -> dict[str, Any]:
    return {
        "valid": False,
        "vehicles_used": 0,
        "total_distance": 0.0,
        "total_travel_time": 0.0,
        "route_loads": [],
        "route_arrival_times": [],
        "route_service_start_times": [],
        "route_return_times": [],
        "route_durations": [],
        "route_feasible": [],
        "vehicle_penalty": vehicle_penalty,
        "combined_cost": None,
        "score": 0.0,
    }


def validate_routes(instance: Any, result: Any) -> tuple[bool, str, dict[str, Any]]:
    """Strictly validate routes and recompute all route metrics."""
    try:
        data = _instance_data(instance)
    except ValueError as exc:
        return False, f"Invalid evaluator instance: {exc}", _empty_stats()
    stats = _empty_stats(data["vehicle_penalty"])

    def invalid(message: str) -> tuple[bool, str, dict[str, Any]]:
        return False, message, stats

    if not isinstance(result, dict):
        return invalid("Result must be a mapping containing a routes list.")
    routes = result.get("routes")
    if not isinstance(routes, list):
        return invalid("Result routes must be a list of route lists.")
    if not routes:
        return invalid("Result routes cannot be empty when customers exist.")
    if len(routes) > data["max_vehicles"]:
        return invalid(
            f"Routes use {len(routes)} vehicles, exceeding the vehicle limit."
        )

    seen: set[int] = set()
    for route_index, route in enumerate(routes):
        if not isinstance(route, list):
            return invalid("Result routes must contain only route lists.")
        if len(route) < 3:
            return invalid(f"Route {route_index} must visit at least one customer.")
        if route[0] != 0:
            return invalid(f"Route {route_index} must start at depot 0.")
        if route[-1] != 0:
            return invalid(f"Route {route_index} must end at depot 0.")
        if 0 in route[1:-1]:
            return invalid(f"Depot 0 appears in the middle of route {route_index}.")
        for node_id in route:
            if not isinstance(node_id, int) or isinstance(node_id, bool):
                return invalid("Every route node ID must be an integer.")
            if node_id < 0:
                return invalid("Every route node ID must be nonnegative.")
            if node_id not in data["nodes"]:
                return invalid(f"Unknown customer ID {node_id}.")
        for customer_id in route[1:-1]:
            if customer_id in seen:
                return invalid(f"Duplicate visit to customer ID {customer_id}.")
            seen.add(customer_id)

    missing = data["customer_ids"] - seen
    if missing:
        return invalid(f"Missing customer IDs: {sorted(missing)}.")

    total_distance = 0.0
    total_travel_time = 0.0
    depot = data["nodes"][0]
    for route_index, route in enumerate(routes):
        load = sum(data["nodes"][node_id]["demand"] for node_id in route[1:-1])
        if load > data["capacity"]:
            return invalid(f"Route {route_index} load {load} exceeds vehicle capacity.")

        clock = depot["earliest"] + depot["service_time"]
        route_start = depot["earliest"]
        arrivals = []
        service_starts = []
        route_distance = 0.0
        route_travel_time = 0.0
        previous = 0
        for customer_id in route[1:-1]:
            distance = data["distances"][previous][customer_id]
            travel_time = data["travel_times"][previous][customer_id]
            route_distance += distance
            route_travel_time += travel_time
            arrival = clock + travel_time
            customer = data["nodes"][customer_id]
            service_start = max(arrival, customer["earliest"])
            if service_start > customer["latest"]:
                return invalid(
                    f"Customer {customer_id} is served after its time window."
                )
            arrivals.append(arrival)
            service_starts.append(service_start)
            clock = service_start + customer["service_time"]
            previous = customer_id

        return_distance = data["distances"][previous][0]
        return_travel_time = data["travel_times"][previous][0]
        route_distance += return_distance
        route_travel_time += return_travel_time
        return_time = clock + return_travel_time
        if depot["latest"] is not None and return_time > depot["latest"]:
            return invalid(f"Route {route_index} returns after the depot time window.")

        total_distance += route_distance
        total_travel_time += route_travel_time
        stats["route_loads"].append(load)
        stats["route_arrival_times"].append(arrivals)
        stats["route_service_start_times"].append(service_starts)
        stats["route_return_times"].append(return_time)
        stats["route_durations"].append(return_time - route_start)
        stats["route_feasible"].append(True)

    stats["valid"] = True
    stats["vehicles_used"] = len(routes)
    stats["total_distance"] = total_distance
    stats["total_travel_time"] = total_travel_time
    return True, "", stats


def score_result(instance: Any, result: Any) -> tuple[dict[str, Any], bool, str]:
    """Validate one solution and compute its minimization-derived score."""
    valid, error, stats = validate_routes(instance, result)
    if valid:
        combined_cost = (
            stats["vehicle_penalty"] * stats["vehicles_used"] + stats["total_distance"]
        )
        stats["combined_cost"] = combined_cost
        stats["score"] = 1.0 / (1.0 + combined_cost)
    return stats, valid, error


def _load_candidate(program_path: str) -> Any:
    path = Path(program_path)
    spec = importlib.util.spec_from_file_location(
        f"vrptw_candidate_{abs(hash(path.resolve()))}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_vrptw"):
        raise AttributeError("Candidate must define run_vrptw(instance, seed).")
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
    """Load a candidate and evaluate it against independent instance copies."""
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
            result = module.run_vrptw(candidate_instance, seed=instance["seed"])
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
                "mean_vehicles_used": sum(
                    item["vehicles_used"] for item in per_instance
                )
                / num_instances,
                "mean_total_distance": sum(
                    item["total_distance"] for item in per_instance
                )
                / num_instances,
                "mean_total_travel_time": sum(
                    item["total_travel_time"] for item in per_instance
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
    parser = argparse.ArgumentParser(description="Evaluate a VRPTW candidate")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to a Python candidate defining run_vrptw(instance, seed)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
