# 3 — Event-triggered search steering (validity + diversity)

Two controllers ([`shinka/core/steering.py`](../../shinka/core/steering.py), tested by
[`test_steering.py`](../../shinka/core/test_steering.py), 46 checks) **observe the search and
intervene only when it starts failing**, then release when it recovers:

- **Validity controller (λ):** tracks per-class failure rates (runtime / **timeout** / **compile** /
  wrong_answer / malformed / boundary / overlap) with a dual-ascent λ; when a failure class
  persists it injects the recurring error witnesses into the mutation prompt.
- **Diversity controller (γ):** clusters admitted programs into *algorithm families* (LLM summary →
  embedding), measures family entropy vs a stage-target, and when the population collapses into one
  family it steers parent/inspiration selection toward other families.

With all controllers off the code path is **identical to the baseline** (`NORMAL` mode).

## Configs & how to run (per problem folder)

| config | meaning |
|---|---|
| `shinka_alg2.yaml` | baseline (3-aux base, **no steering**) |
| `shinka_steer_full.yaml` | **both** controllers on |
| `shinka_valonly.yaml` (circle) | **validity only** (diversity off) — the fix |

```bash
# circle, matched ablation (5 base + 5 steer, interleaved)
bash run_steer_ablation.sh
# circle validity-only
bash run_valonly.sh
# 2048 / go / julia
bash run_2048_ablation.sh    # 2048
bash run_go_ablation.sh      # go   (needs go on PATH)
bash run_julia_ablation.sh   # julia
```
Per-proposal controller state is logged to `steering.jsonl` in each run dir;
`STEERING ACT [MODE] gen N` log lines mark when a steering prompt block was injected.

## Findings (`../../results_summary/steering_summary.csv`)

| problem | dominant failure | baseline yield | full-steer yield | baseline ceiling | full-steer ceiling |
|---|---|---|---|---|---|
| circle | overlap/runtime | 12.9% | 22.9% | **2.46** | **2.00** ⬇ |
| 2048 | runtime/timeout | 36.3% | 34.3% | 0.75 | 0.63 ⬇ |
| Go | **compile** | 7.0% | **17.0%** | 99.73 | 99.74 |
| Julia | **compile** | 9.0% | **23.0%** (p=0.044) | 99.76 | 99.75 |

- **Diversity steering destabilizes problems with ceiling headroom** (circle, 2048) — it starves the
  best algorithm family and forbids the small refinements that reach the peak (see
  `../../notes/`). **Validity-only** (`shinka_valonly.yaml`) keeps the yield gain (13%→23%, p=0.028)
  while mostly preserving the ceiling (2.15).
- On **saturated** problems (Go/Julia) there is no peak to lose, so full steering's validity half
  (fixing compile errors) is a clear yield win.

See `../../results_summary/steering_diagnostic_circle.png` (controllers diagnosing a baseline run)
and `steering_live_vs_replay_validation.png` (the wired controllers match the offline validator).
