"""Deterministic odd-even transposition sorting-network baseline."""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
def build_network(instance: dict[str, Any], seed: int) -> list[list[tuple[int, int]]]:
    """Construct an odd-even transposition network for any supported wire count."""
    del seed  # The baseline is deterministic; evolved strategies may use it.
    n_wires = instance["n_wires"]
    network = []
    for phase in range(n_wires):
        start = phase % 2
        layer = [(wire, wire + 1) for wire in range(start, n_wires - 1, 2)]
        if layer:
            network.append(layer)
    return network


# EVOLVE-BLOCK-END


def run_sorting_network(instance: dict[str, Any], seed: int) -> dict[str, Any]:
    """Run the candidate network builder for one evaluator-provided instance."""
    return {"network": build_network(instance, seed)}
