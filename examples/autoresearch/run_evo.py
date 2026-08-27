#!/usr/bin/env python3
"""Run ShinkaEvolve on the lightweight Autoresearch example."""

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


job_config = LocalJobConfig(
    eval_program_path="evaluate.py",
    time="00:02:00",
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

task_sys_msg = """
You are optimizing a CPU-safe byte-level next-token predictor for a lightweight
Autoresearch benchmark. For each instance, run_autoresearch(instance, seed)
receives training_bytes, validation_contexts, a 256-byte vocabulary, and strict
compute limits. Return one normalized, finite 256-way distribution per context.
Validation targets remain evaluator-side. The evaluator independently checks
shape, normalization, finiteness, mutation, timeout, cross-entropy, and
bits-per-byte; candidate-reported metrics are ignored.

Minimize validation bits-per-byte with structurally different strategies, not
just repeated learning-rate, width, depth, or smoothing tweaks. Useful avenues:
- unigram and interpolated n-gram models with adaptive or entropy-based backoff;
- longest-suffix matching, prediction by partial matching, context trees, and
  compression-inspired recency-weighted dictionaries;
- explicit cycle, delimiter, repeated-substring, grammar, and run detection;
- small deterministic CPU models such as an embedding MLP, RNN, GRU, or compact
  attention-like model with bounded training;
- optimizer, schedule, clipping, batching, curriculum, and sampling changes;
- hybrids or ensembles combining statistical predictors and tiny learned models;
- safe probability floors, stable softmax, and uniform fallback distributions.

Keep execution deterministic for the supplied seed, local, fast, and modest in
memory. Preserve run_autoresearch(instance, seed). Do not access external
services, require a GPU, fabricate loss metadata, mutate inputs, monkeypatch the
evaluator, hardcode validation answers, or return anything other than actual
probability or normalized log-probability rows.
"""

evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=100,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=["gpt-5-mini"],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["medium"],
        max_tokens=32768,
    ),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    init_program_path="initial.py",
    results_dir="results_autoresearch",
    max_novelty_attempts=1,
)


MAX_EVALUATION_JOBS = 4
MAX_PROPOSAL_JOBS = 2
MAX_DB_WORKERS = 2


def main() -> None:
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=MAX_EVALUATION_JOBS,
        max_proposal_jobs=MAX_PROPOSAL_JOBS,
        max_db_workers=MAX_DB_WORKERS,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
