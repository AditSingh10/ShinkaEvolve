"""Deterministic byte n-gram baseline for the Autoresearch example."""

from __future__ import annotations

from typing import Any


# EVOLVE-BLOCK-START
def predict_probabilities(instance: dict[str, Any], seed: int) -> list[list[float]]:
    """Predict with longest-suffix counts and a smoothed unigram fallback."""
    del seed  # The baseline is deterministic; evolved strategies may use it.
    training = instance["training_bytes"]
    vocab_size = instance["vocab_size"]
    contexts = instance["validation_contexts"]

    unigram_counts = [0.25] * vocab_size
    for token in training:
        unigram_counts[token] += 1.0

    predictions = []
    for context in contexts:
        counts = unigram_counts.copy()
        for order in range(min(8, len(context)), 0, -1):
            suffix = context[-order:]
            matches = []
            for position in range(order, len(training)):
                if training[position - order : position] == suffix:
                    matches.append(training[position])
            if matches:
                counts = [0.05] * vocab_size
                for token in matches:
                    counts[token] += 1.0
                break
        total = sum(counts)
        predictions.append([count / total for count in counts])
    return predictions


# EVOLVE-BLOCK-END


def run_autoresearch(instance: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return one full byte-probability row per validation context."""
    return {"probabilities": predict_probabilities(instance, seed)}
