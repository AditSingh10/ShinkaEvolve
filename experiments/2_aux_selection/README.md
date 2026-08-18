# 2 — Auxiliary-objective parent selection (Algorithm 2)

A two-level non-stationary bandit ([`shinka/database/aux_selection.py`](../../shinka/database/aux_selection.py))
chooses which parent to breed from using **auxiliary (helper) objectives** instead of the oracle
score alone. The auxes are computed in `aux_evals.py`, stored privately, and **never shown to the
model** — so any effect is attributable purely to *which parent was selected*.

| problem | folder | auxiliary objectives (`aux_evals.py`) |
|---|---|---|
| circle packing | `circle_packing/` | **caging** (how boxed-in a circle is), **hole** (wasted gap area), **connect** (contact-graph connectivity) |
| 2048 | `game_2048/` | **empty** (free cells), **monotonic** (row/col ordering), **smooth** (neighbor closeness), **corner** (max tile in a corner), **merge** (mergeable pairs) |

**Arms.**
- Treatment (aux ON): `shinka_alg2.yaml` — `aux_directions` set to the helper objectives.
- Control (aux OFF): `shinka_alg2_control.yaml` — `aux_directions: {}` (= stock weighted sampling).

**Run** (5 reps each, e.g. circle):
```bash
PILOT_AUX=none python run_evo.py --config_path shinka_alg2.yaml         --run_tag treatment_rep1
PILOT_AUX=none python run_evo.py --config_path shinka_alg2_control.yaml --run_tag control_rep1
```

**Finding** (`../../results_summary/aux_selection_summary.csv`): on circle, aux selection raised
valid-yield **16% → 23%** with the ceiling unchanged (~2.45); on 2048 it was neutral. The bandit's
β (oracle-vs-aux) and per-aux Q traces are logged to `aux_bandit_history.jsonl` per run.
