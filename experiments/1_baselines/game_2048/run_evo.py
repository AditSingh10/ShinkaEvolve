#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): game_2048 launcher. Aux-guided parent SELECTION only. ===
# The aux-selection feature toggles purely on db_config.aux_directions (non-empty =>
# treatment, {} => control). No text feedback to the model: aux scores stay private.
import argparse

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

search_task_sys_msg = """You are an expert game-playing agent and Python programmer. \
You are improving a heuristic strategy for the game 2048 on a 4x4 board.

The function get_best_move(board) receives a 4x4 numpy array where each cell is 0 (empty) \
or an integer exponent (a cell holding value 2^k is stored as k; the 2048 tile is 11). \
It must return one of Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT.

The goal is to reach the highest possible tile in as few moves as possible. Strong 2048 \
strategies typically: keep the largest tile anchored in a corner; maintain a monotonic \
(ordered) gradient across rows/columns; keep the board smooth so neighbouring tiles can \
merge; preserve empty cells; and avoid moves that break the corner structure. Consider \
look-ahead / expectimax-style evaluation of candidate moves. Return a valid move for \
every board state and keep the function fast (it runs under a per-move time limit)."""


def main(config_path: str, run_tag: str = "", smoke: bool = False):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg
    config["evo_config"]["use_text_feedback"] = False  # pure selection intervention

    tag = run_tag or "default"
    config["evo_config"]["results_dir"] = f"results/{tag}"
    if smoke:
        config["evo_config"]["num_generations"] = 4
        config["evo_config"]["results_dir"] = f"results/smoke_{tag}"

    print(
        f"[ALG2 2048] config={config_path} results_dir={config['evo_config']['results_dir']} "
        f"aux_directions={config['db_config'].get('aux_directions')} smoke={smoke}"
    )

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(eval_program_path="evaluate.py", time="00:05:00")
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
    parser.add_argument("--config_path", type=str, default="shinka_alg2.yaml")
    parser.add_argument("--run_tag", type=str, default="",
                        help="results_dir suffix (e.g. treatment_rep1) so runs never collide.")
    parser.add_argument("--smoke", action="store_true",
                        help="4 generations, for prototype validation only.")
    args = parser.parse_args()
    main(args.config_path, run_tag=args.run_tag, smoke=args.smoke)
# === END ALG2 MOD (aux-selection) ===
