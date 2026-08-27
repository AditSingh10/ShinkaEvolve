#!/usr/bin/env python3
"""Run ShinkaEvolve on the Sorting Network Synthesis example."""

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
You are optimizing Python constructions for Sorting Network Synthesis.

For each instance, return one fixed layered comparator network for n_wires.
Every comparator must be a canonical pair (i, j) with 0 <= i < j < n_wires.
Comparators inside a layer run in parallel, so their wires must be disjoint.
The same data-independent network must sort every Boolean input vector.

The evaluator validates the structure, exhaustively tests all 2^n Boolean
inputs, and independently recomputes depth, comparator count, counterexamples,
and score. Candidate-reported correctness or cost metadata is ignored. The
objective mode is either depth_then_size or size_then_depth.

Explore structurally different network families rather than repeatedly adding,
deleting, or swapping one comparator around a fixed template. Useful families
and transformations include:
- odd-even transposition and dependency-safe layer compaction;
- recursive bitonic sorting and merge networks, including arbitrary-size
  pruning or specialized small cases;
- recursive odd-even merge constructions;
- bubble-like comparator sequences followed by dependency-aware scheduling;
- forward, reverse, or seeded comparator pruning with exhaustive revalidation;
- counterexample-guided repair that targets remaining output inversions;
- bounded layer enumeration or behavior-signature search for small wire counts;
- hybrids that construct several families, prune and compact each, and return
  the best independently validated network.

Keep runtime bounded and deterministic for the supplied seed. Preserve
run_sorting_network(instance, seed) and return actual comparator layers. Do not
hardcode only the visible wire counts, fabricate metrics, return sorted outputs
or a sorting function, branch on test inputs, or spend generations solely on
tiny local edits to one network template.
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
    results_dir="results_sorting_network",
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
