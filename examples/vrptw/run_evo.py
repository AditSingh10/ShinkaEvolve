#!/usr/bin/env python3
"""Run ShinkaEvolve on the Vehicle Routing with Time Windows example."""

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
You are optimizing a Python heuristic for the Vehicle Routing Problem with
Time Windows (VRPTW).

Each route must start and end at depot 0, serve every customer exactly once,
respect vehicle capacity and the maximum vehicle count, and begin service
inside every customer's time window. Vehicles may wait after arriving early.
Travel time, waiting, customer service time, and any depot return deadline all
affect feasibility.

The evaluator uses the original demands, matrices, service times, and time
windows to validate routes and recompute all metrics. Candidate-reported
distance, vehicle count, timings, feasibility, and scores are ignored. The
objective strongly prefers fewer vehicles, then shorter total distance.

Explore structurally different algorithms rather than repeatedly tuning one
weighted nearest-neighbor score. Useful families include:
- nearest-feasible, earliest-deadline, minimum-slack, and demand-aware greedy;
- cheapest or regret insertion with waiting- and deadline-aware feasibility;
- Clarke-Wright-style savings with capacity/time-window merge checks;
- sweep, geographic, demand-balanced, or time-window clustering followed by
  insertion and repair;
- relocate, swap, segment reversal, chain moves, route merging, and explicit
  vehicle-removal attempts;
- deterministic large-neighborhood search with related or worst-route removal
  and feasibility-first repair;
- dynamic programming or exhaustive repair for very small customer subsets;
- bounded hybrids that build several solutions and return the best valid one.

Keep runtime predictable and deterministic for the supplied seed. Preserve
run_vrptw(instance, seed) and return actual routes. Do not hardcode visible
instances, return partial coverage, ignore capacity or time windows, fabricate
metrics, or spend generations only changing coefficients around one heuristic.
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
    results_dir="results_vrptw",
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
