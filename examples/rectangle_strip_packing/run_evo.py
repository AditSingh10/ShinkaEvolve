#!/usr/bin/env python3
"""Run ShinkaEvolve on the rectangle strip packing example."""

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
You are optimizing a Python heuristic for 2D Rectangle Strip Packing.

For each instance, place every axis-aligned rectangle exactly once in a
fixed-width, unbounded-height strip. Coordinates are the rectangle's
bottom-left corner. Rotation by 90 degrees is permitted only when the
instance enables it. Placements must use nonnegative finite coordinates,
remain within the strip width, and have no positive-area overlap. Touching
edges and corners is valid.

The evaluator derives dimensions from the original rectangle data, checks
that the input was not mutated, validates all geometry, and recomputes used
height. Candidate-reported scores, heights, utilization, dimensions, or
validity metadata are ignored. A valid solution's score improves as its used
height approaches independently computed lower bounds.

Explore structurally different packing methods, not only small coefficient or
sort-key changes. Useful families include:
- rotation-aware shelf variants with first-fit or best-fit shelf selection;
- bottom-left placement using geometric candidate points and gap-aware ties;
- skyline placement with waste minimization, cleanup, or bounded lookahead;
- maximal-rectangle free-space splitting and pruning;
- guillotine packings with alternative split choices;
- deterministic local repair, subset repacking, or downward/left compaction;
- bounded seeded search over orderings and rotations;
- hybrids that construct several valid layouts and return the lowest one.

Keep runtime predictable and deterministic for the supplied seed. Preserve the
run_packing(instance, seed) interface and return actual placements. Do not
hardcode layouts for the visible instances, fake metadata, hide invalid
geometry behind claimed metrics, or spend generations only tuning constants
around first-fit decreasing.
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
    results_dir="results_rectangle_strip_packing",
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
