# 1 — Baselines (stock ShinkaEvolve)

Unmodified evolutionary search: **no auxiliary objectives, no steering**. These are the control
arm every other experiment is compared against.

| problem | folder | config | run |
|---|---|---|---|
| circle packing (n=26) | `circle_packing/` | `shinka_alg2_control.yaml` (`aux_directions: {}`) | `PILOT_AUX=none python run_evo.py --config_path shinka_alg2_control.yaml --run_tag control_rep1` |
| 2048 | `game_2048/` | `shinka_alg2_control.yaml` | `python run_evo.py --config_path shinka_alg2_control.yaml --run_tag control_rep1` |
| Go / Collatz | `go_collatz/` | inline (`run_evo_local.py`) | `python run_evo_local.py --run_tag baseline_local` |
| Julia / primes | `julia_primes/` | inline (`run_evo_local.py`) | `python run_evo_local.py --run_tag baseline_local` |

**Objectives.** Circle: maximize sum of 26 radii in the unit square (best ≈ 2.6). 2048: `max_tile/512 − 0.002·moves`. Go/Julia: `accuracy·100 − runtime` (correctness-gated speed).

**Failure modes** (why proposals are wasted): circle = overlap/out-of-bounds + Python crashes; 2048 = illegal moves / timeouts; Go/Julia = **compile errors** + wrong answers.

Baseline valid-yield: circle ~13%, 2048 ~36%, Go ~7%, Julia ~9% (medians, n=5). See `../../results_summary/`.
