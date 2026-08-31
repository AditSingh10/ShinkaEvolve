"""Run evolution against an explicitly prepared external Autoresearch checkout."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import adapter
import setup_upstream

if TYPE_CHECKING:
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

MANIFEST_PATH = Path(__file__).with_name("upstream_manifest.json")
EVALUATOR_TIMEOUT_SECONDS = 600.0
OUTER_EVALUATION_TIMEOUT = "00:12:00"
MAX_EVALUATION_JOBS = 1
MAX_PROPOSAL_JOBS = 2
MAX_DB_WORKERS = 2


class RunnerPrerequisiteError(ValueError):
    """Raised when the operator has not prepared a usable external checkout."""


def resolve_upstream_root(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the operator-provided external checkout without creating it."""
    environment = os.environ if environ is None else environ
    configured_root = environment.get("AUTORESEARCH_ROOT", "").strip()
    if not configured_root:
        raise RunnerPrerequisiteError(
            "AUTORESEARCH_ROOT must identify a checkout prepared with "
            "setup_upstream.py."
        )
    return Path(configured_root).expanduser().resolve()


def require_cuda_visible_devices(environ: Mapping[str, str] | None = None) -> str:
    """Require the GPU assignment inherited from the operator or scheduler."""
    environment = os.environ if environ is None else environ
    try:
        return adapter.require_single_cuda_device(environment)
    except adapter.AdapterError as error:
        raise RunnerPrerequisiteError(
            f"{error} Local Shinka jobs do not allocate GPUs."
        ) from error


def preflight_evolution(
    root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    manifest_path: str | Path = MANIFEST_PATH,
) -> Path:
    """Validate the external checkout, frozen seed, and inherited GPU assignment."""
    require_cuda_visible_devices(environ)
    upstream_root = Path(root).expanduser().resolve()
    manifest = adapter.load_manifest(manifest_path)
    setup_upstream._validated_root(upstream_root, manifest)
    seed_path = upstream_root / "train.py"
    if not seed_path.is_file():
        raise RunnerPrerequisiteError(f"Upstream train.py is missing: {seed_path}")
    setup_upstream.require_readiness(upstream_root, environ=environ)
    return seed_path


def _task_prompt() -> str:
    return """
You are optimizing Andrej Karpathy's official Autoresearch train.py program.
Each candidate runs in a prepared external checkout for the official five-minute
training budget on the GPU selected by the inherited CUDA_VISIBLE_DEVICES. Local
Shinka does not allocate GPUs: this runner uses one evaluation job, and multi-GPU
experiments require separately assigned runner processes or a GPU-aware scheduler.

This is a cooperative benchmark. Evolve only the candidate train.py program; do
not modify prepare.py, pyproject.toml, uv.lock, or .python-version; do not clone,
install, download, access external services, bypass the evaluator, or fabricate
the final summary. Preserve a valid runnable Python program and let the official
training/evaluation path report the actual finite val_bpb. Seek substantive,
structurally diverse improvements across architecture, attention, optimization,
normalization, batching, scheduling, and the training loop rather than repeated
single-hyperparameter tweaks.
""".strip()


def build_configs(
    root: str | Path,
) -> tuple[EvolutionConfig, LocalJobConfig, DatabaseConfig]:
    """Build runner configuration for one external checkout without side effects."""
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    upstream_root = Path(root).expanduser().resolve()
    job_config = LocalJobConfig(
        eval_program_path=str(Path(__file__).with_name("evaluate.py").resolve()),
        extra_cmd_args={
            "upstream-root": str(upstream_root),
            "timeout-seconds": EVALUATOR_TIMEOUT_SECONDS,
        },
        time=OUTER_EVALUATION_TIMEOUT,
    )
    evo_config = EvolutionConfig(
        task_sys_msg=_task_prompt(),
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=100,
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        llm_models=["gpt-5-mini"],
        llm_kwargs={
            "temperatures": [0.0, 0.5, 1.0],
            "reasoning_efforts": ["medium"],
            "max_tokens": 32768,
        },
        embedding_model="text-embedding-3-small",
        code_embed_sim_threshold=0.995,
        init_program_path=str(upstream_root / "train.py"),
        results_dir="results_autoresearch",
        max_novelty_attempts=1,
    )
    db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=2,
        archive_size=40,
        elite_selection_ratio=0.3,
        num_archive_inspirations=4,
        num_top_k_inspirations=2,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
    )
    return evo_config, job_config, db_config


def build_runner(root: str | Path) -> ShinkaEvolveRunner:
    """Construct the one-job runner after preflight has checked prerequisites."""
    from shinka.core import ShinkaEvolveRunner

    evo_config, job_config, db_config = build_configs(root)
    return ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=MAX_EVALUATION_JOBS,
        max_proposal_jobs=MAX_PROPOSAL_JOBS,
        max_db_workers=MAX_DB_WORKERS,
        verbose=True,
    )


def main() -> None:
    """Preflight the external setup, then start evolution without setup side effects."""
    upstream_root = resolve_upstream_root()
    preflight_evolution(upstream_root)
    build_runner(upstream_root).run()


if __name__ == "__main__":
    main()
