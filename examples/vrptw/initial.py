"""Deterministic greedy baseline for Vehicle Routing with Time Windows."""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
def build_routes(instance: dict[str, Any], seed: int) -> list[list[int]]:
    """Build capacity- and time-feasible routes using greedy extension."""
    del seed  # The baseline is deterministic; evolved strategies may use it.
    customers = {customer["id"]: customer for customer in instance["customers"]}
    unserved = set(customers)
    capacity = float(instance["vehicle_capacity"])
    travel_times = instance.get("travel_time_matrix", instance["distance_matrix"])
    distances = instance["distance_matrix"]
    depot = instance["depot"]
    depot_window = depot.get("time_window", [0.0, None])
    depot_earliest = float(depot_window[0])
    depot_latest = depot_window[1]
    routes: list[list[int]] = []

    while unserved:
        route = [0]
        load = 0.0
        clock = depot_earliest + float(depot.get("service_time", 0.0))
        current = 0

        while True:
            feasible = []
            for customer_id in unserved:
                customer = customers[customer_id]
                demand = float(customer["demand"])
                if load + demand > capacity:
                    continue
                arrival = clock + float(travel_times[current][customer_id])
                earliest, latest = customer["time_window"]
                service_start = max(arrival, float(earliest))
                if service_start > float(latest):
                    continue
                departure = service_start + float(customer["service_time"])
                return_time = departure + float(travel_times[customer_id][0])
                if depot_latest is not None and return_time > float(depot_latest):
                    continue
                feasible.append(
                    (
                        float(latest),
                        float(distances[current][customer_id]),
                        -demand,
                        customer_id,
                        service_start,
                    )
                )

            if not feasible:
                break
            _, _, _, customer_id, service_start = min(feasible)
            customer = customers[customer_id]
            route.append(customer_id)
            load += float(customer["demand"])
            clock = service_start + float(customer["service_time"])
            current = customer_id
            unserved.remove(customer_id)

        if len(route) == 1:
            raise ValueError("No feasible route can serve the remaining customers.")
        route.append(0)
        routes.append(route)

    return routes


# EVOLVE-BLOCK-END


def run_vrptw(instance: dict[str, Any], seed: int) -> dict[str, Any]:
    """Run the candidate route builder for one evaluator-provided instance."""
    return {"routes": build_routes(instance, seed)}
