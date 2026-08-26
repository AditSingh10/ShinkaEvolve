#!/usr/bin/env python3
# LOCAL-model baseline (no aux, no LLM bandit): repoints the stock Go example at the
# vLLM Qwen2.5-32B server. Matches our circle/2048 baseline knobs so the wasted-proposal
# breakdown is comparable across problems: full patches only (Qwen can't do verbatim diffs),
# temp 0.7, local embed + local novelty judge, meta off.
import run_evo as stock
from shinka.core import ShinkaEvolveRunner, EvolutionConfig

LOCAL_GEN = "local/qwen32b@http://localhost:8000/v1"
LOCAL_EMB = "local/qwen-embed@http://localhost:8001/v1"

evo_config = EvolutionConfig(
    task_sys_msg=stock.task_sys_msg,
    patch_types=["full"],
    patch_type_probs=[1.0],
    num_generations=100,
    max_patch_resamples=4,
    max_patch_attempts=1,
    job_type="local",
    language="go",
    llm_models=[LOCAL_GEN],
    llm_kwargs=dict(temperatures=[0.7], reasoning_efforts=[""], max_tokens=8192),
    embedding_model=LOCAL_EMB,
    code_embed_sim_threshold=0.99,
    novelty_llm_models=[LOCAL_GEN],
    novelty_llm_kwargs=dict(temperatures=[0.0]),
    meta_rec_interval=None,
    init_program_path="initial.go",
    results_dir="results/baseline_local",
    max_novelty_attempts=1,
)


STEER_KW = dict(
    enable_diversity_controller=True, enable_validity_controller=True,
    enable_family_based_islands=True, enable_mode_prompt_conditioning=True,
    enable_mode_specific_island_sampling=True, enable_mode_specific_inspiration_sampling=True,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", default="baseline_local")
    ap.add_argument("--steer", action="store_true", help="enable both steering controllers")
    a = ap.parse_args()
    evo_config.results_dir = f"results/{a.run_tag}"
    if a.steer:
        evo_config.steering_kwargs = dict(STEER_KW)
    ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=stock.job_config,
        db_config=stock.db_config,
        max_evaluation_jobs=1,
        max_proposal_jobs=2,
        max_db_workers=1,
        verbose=True,
    ).run()


if __name__ == "__main__":
    main()
