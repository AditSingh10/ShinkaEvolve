#!/usr/bin/env python3
import argparse

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# === PILOT MOD (aux-eval): resolve the active aux for this run (env PILOT_AUX). ===
import pilot_config
# === END PILOT MOD (aux-eval) ===

search_task_sys_msg = """You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

Key directions to explore:
1. The optimal arrangement likely involves variable-sized circles
2. A pure hexagonal arrangement may not be optimal due to edge effects
3. The densest known circle packings often use a hybrid approach
4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
5. Consider strategic placement of circles at square corners and edges
6. Adjusting the pattern to place larger circles at the center and smaller at the edges
7. The math literature suggests special arrangements for specific values of n
8. You can use the scipy optimize package (e.g. LP or SLSQP) to optimize the radii given center locations and constraints

Be creative and try to find a new solution better than the best known result."""


def main(
    config_path: str,
    smoke: bool = False,
    run_tag: str = "",
    force_text_feedback: bool = False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg

    # === PILOT MOD (aux-eval): per-condition wiring. Does NOT touch the loop/oracle. ===
    aux = pilot_config.active_aux()  # "none" | "m1" | "m2" | "m3"
    # Normally: treatments turn text feedback ON, control leaves it OFF (true baseline).
    # For the CONTROLLED experiment, pass --force_text_feedback so BOTH arms run with the
    # flag ON; then the only prompt difference between arms is the note *content* (control's
    # note is empty because aux="none"), not the code path.
    config["evo_config"]["use_text_feedback"] = force_text_feedback or (aux != "none")
    # Output dir: per condition, plus an optional per-replicate tag so runs never collide.
    tag = f"_{run_tag}" if run_tag else ""
    config["evo_config"]["results_dir"] = f"results/pilot_{aux}{tag}"
    if smoke:
        # Cheap prototype validation: a few generations, tiny budget.
        config["evo_config"]["num_generations"] = 5
        config["evo_config"]["max_api_costs"] = 1.0
        config["evo_config"]["results_dir"] = f"results/smoke_{aux}{tag}"
    print(
        f"[PILOT MOD] active_aux={aux}  use_text_feedback="
        f"{config['evo_config']['use_text_feedback']}  "
        f"results_dir={config['evo_config']['results_dir']}  smoke={smoke}"
    )
    # === END PILOT MOD (aux-eval) ===

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="00:05:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_small.yaml")
    # === PILOT MOD (aux-eval): smoke toggle + controlled-experiment flags. ===
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Few generations + tiny budget, for prototype validation only.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Suffix for results_dir (e.g. rep1) so replicate runs never collide.",
    )
    parser.add_argument(
        "--force_text_feedback",
        action="store_true",
        help="Force use_text_feedback=True in BOTH arms (controlled experiment): the only "
        "prompt difference then is the note content, not the code path.",
    )
    # === END PILOT MOD (aux-eval) ===
    args = parser.parse_args()
    main(
        args.config_path,
        smoke=args.smoke,
        run_tag=args.run_tag,
        force_text_feedback=args.force_text_feedback,
    )
