#!/usr/bin/env python3
"""Run the canonical Circle Packing evaluator without importing scheduler backends.

The public ``shinka.core`` package initializer imports Slurm/Docker launch code that
creates an AFS-home cache even for a local evaluation. This entrypoint exposes the
existing ``run_shinka_eval`` implementation to the unchanged canonical evaluator
without loading that unrelated scheduler initialization path.
"""
import importlib.util
import runpy
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WRAP_EVAL = REPO / "shinka/core/wrap_eval.py"
CANONICAL_EVALUATOR = REPO / "examples/circle_packing/evaluate.py"

spec = importlib.util.spec_from_file_location("ablation_wrap_eval", WRAP_EVAL)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load {WRAP_EVAL}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

core_stub = types.ModuleType("shinka.core")
core_stub.run_shinka_eval = module.run_shinka_eval
sys.modules["shinka.core"] = core_stub
runpy.run_path(str(CANONICAL_EVALUATOR), run_name="__main__")
