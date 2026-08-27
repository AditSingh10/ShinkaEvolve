"""Importable child-process worker for the Autoresearch evaluator."""

from __future__ import annotations

import copy
import importlib.util
import time
from pathlib import Path
from typing import Any


def _load_candidate(program_path: str) -> Any:
    path = Path(program_path)
    spec = importlib.util.spec_from_file_location(
        f"autoresearch_candidate_{abs(hash(path.resolve()))}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_autoresearch"):
        raise AttributeError("Candidate must define run_autoresearch(instance, seed).")
    return module


def candidate_worker(
    program_path: str, instance: dict[str, Any], connection: Any
) -> None:
    """Run untrusted candidate code and return its result over a pipe."""
    try:
        module = _load_candidate(program_path)
        snapshot = copy.deepcopy(instance)
        start = time.perf_counter()
        result = module.run_autoresearch(instance, seed=instance["seed"])
        runtime = time.perf_counter() - start
        try:
            mutated = instance != snapshot
        except Exception:
            mutated = True
        connection.send(
            {
                "status": "ok",
                "result": result,
                "mutated": bool(mutated),
                "runtime_seconds": runtime,
            }
        )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_seconds": 0.0,
                }
            )
        except Exception:
            pass
    finally:
        connection.close()
