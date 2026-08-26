# Pre-registration — Do auxiliary objectives speed up the search? (and by what mechanism)

Registered before analysis. Extends the existing n=3 runs to **n=5** per cell (reps 1–3 already
done; adding reps 4–5 with identical configs, so all 5 are poolable).

## Design
- 3 problems: circle packing (n=26), JSSP (ft06), 2048.
- 2 arms: **AUX** = fixed hand-designed auxes on (caging/hole/connect | per-job flow-times |
  board heuristics), `beta_init=0.70`; **NO-AUX** = oracle-only control (`aux_directions: {}`).
  Everything else identical (crowding archive, 2 islands, 200 gens, Qwen2.5-32B, temp 0.7).
- n = 5 reps per arm per problem.

## Metrics (fixed now — no post-hoc threshold picking)
| metric | definition | threshold T |
|---|---|---|
| **Speed** (primary) | first generation best-so-far ≥ T | circle **2.40**, JSSP **1.24**, 2048 **0.50** |
| Ceiling | final best-so-far | — |
| Valid-yield | valid ÷ total proposals | — |
| Diversion-gap (aux) | (main-%ile − aux-%ile) of parents bred from | — |
| Divergence (aux) | mean JS(aux parent-dist, oracle parent-dist) | — |

## Hypotheses & decision rule
- **H1 (speed):** AUX reaches T in fewer generations than NO-AUX. One-sided Mann-Whitney U,
  α=0.05, per problem. Runs never reaching T are censored at the worst rank (max_gen+1).
- **H2 (ceiling):** AUX final best > NO-AUX. Same test.
- **H3 (valid-yield):** AUX valid-yield ≠ NO-AUX. Two-sided Mann-Whitney.
- **Mechanism (the real question):** within AUX runs (pooled across reps, per problem and
  overall), Spearman ρ of per-run **speed** vs each of **diversion-gap**, **divergence**,
  **valid-yield**. A mechanism is **SUPPORTED** iff (a) |ρ| > 0.5 with p < 0.05 **and**
  (b) H1 holds for that problem; otherwise **REJECTED** as the explanation.

## Report
Per-problem table (speed/ceiling/valid-yield medians + MW p), and the mechanism-correlation
table. State plainly which of {diversion, divergence, valid-yield} survives, or that none does.
