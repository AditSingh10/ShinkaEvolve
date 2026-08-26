# AuxEvolve — Experiments on Steering LLM-Driven Evolutionary Search

A fork of [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) with three families of
experiments studying **why and when helper signals speed up LLM-driven evolutionary code search**.
All experiments run local Qwen2.5-32B (via vLLM); no API keys.

> **Where is each experiment?** Everything is under [`experiments/`](experiments/), one folder per
> *experiment family × problem*. Findings (figures + CSV tables) are in
> [`results_summary/`](results_summary/); design/pre-registration docs are in [`notes/`](notes/).

## The three experiment families

| # | Folder | What it studies | Problems |
|---|---|---|---|
| **1** | [`experiments/1_baselines/`](experiments/1_baselines/) | **Stock ShinkaEvolve** — no aux, no steering (the control) | circle packing, 2048, Go, Julia |
| **2** | [`experiments/2_aux_selection/`](experiments/2_aux_selection/) | **Algorithm 2 — auxiliary-objective parent selection** (a bandit picks parents by helper objectives). Circle: caging/hole/connect. 2048: empty/monotonic/smooth/**corner**/merge board heuristics. | circle packing, 2048 |
| **3** | [`experiments/3_steering/`](experiments/3_steering/) | **Event-triggered search steering** — validity + diversity controllers that intervene only when the search starts failing | circle packing, 2048, Go, Julia |

## Modified engine (vs stock ShinkaEvolve)

The evolutionary engine lives in [`shinka/`](shinka/). Our changes:
- `shinka/core/steering.py` — the steering controllers (FamilyManager, Diversity/Validity controllers, SteeringPolicy) + `test_steering.py` (46 unit tests).
- `shinka/database/aux_selection.py` — the two-level auxiliary-objective bandit (Algorithm 2).
- `shinka/core/async_runner.py`, `shinka/database/dbase.py`, `shinka/core/config.py` — hooks to wire both features in (all gated; off ⇒ stock behavior).

## Headline findings ([`results_summary/`](results_summary/))

- **Aux selection (Alg 2)** raised circle valid-yield (16%→23%) with no ceiling change; neutral on 2048. See `aux_selection_summary.csv`.
- **Steering** — full (validity+diversity) vs baseline, `steering_summary.csv`:
  - **Diversity steering hurts problems with ceiling headroom** (circle 2.46→2.00, 2048 0.75→0.63) by routing reproduction away from the best algorithm family.
  - **Validity-only steering** raises circle yield **13%→23% (p=0.028)** while mostly preserving the ceiling.
  - On **Go/Julia** (compile-error dominated, objective saturated) **full steering helps yield** (Julia 9%→23%, p=0.044) at no ceiling cost.
  - Rule: *diversity pays off only when there is no peak performance to sacrifice.*

## Reproducing

Each `experiments/**/` folder has its config(s), a run script, and a README. Requires a local
vLLM server (Qwen2.5-32B at `:8000`, Qwen3-Embedding at `:8001`); Go/Julia experiments need those
toolchains on `PATH`. Raw run outputs (`results/`, `gen_*/`, `*.sqlite`) are git-ignored — see
`.gitignore`; only curated summaries + figures are committed.
