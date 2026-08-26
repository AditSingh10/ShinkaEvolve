#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): correctness tests for the selection machinery. ===
"""Validate build_aux_probabilities() and TwoLevelAuxBandit BEFORE wiring into the loop.

Checks the distribution math, that the reused AsymmetricUCB sub-bandit is genuinely
learning (not silently falling back to uniform), and that the two-level bandit converges
to the right behaviour under controlled synthetic rewards.
"""
import numpy as np

from shinka.database.aux_selection import (
    MAIN_ARM,
    TwoLevelAuxBandit,
    build_aux_probabilities,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))


# ------------------------------------------------------------------ distribution
print("\n== build_aux_probabilities ==")
vals = [0.1, 0.2, 0.3, 0.4, 0.9]          # last is an outlier
zero_children = [0] * 5

p_max = build_aux_probabilities(vals, +1, zero_children)
p_min = build_aux_probabilities(vals, -1, zero_children)

check("sums to 1 / finite / non-negative",
      abs(p_max.sum() - 1) < 1e-9 and np.isfinite(p_max).all() and (p_max >= 0).all())
check("direction=+1 favours HIGH aux values",
      p_max[-1] > p_max[0] and np.argmax(p_max) >= 3, f"p={np.round(p_max,3)}")
check("direction=-1 favours LOW aux values",
      p_min[0] > p_min[-1] and np.argmax(p_min) <= 1, f"p={np.round(p_min,3)}")
check("direction flips the ordering", np.argmax(p_max) != np.argmax(p_min))

p_const = build_aux_probabilities([0.5] * 5, +1, zero_children)
check("constant aux -> uniform (dead arm degrades gracefully, no crash)",
      np.allclose(p_const, 0.2, atol=1e-6), f"p={np.round(p_const,3)}")

p_div = build_aux_probabilities([0.5] * 5, +1, [0, 0, 0, 0, 9])
check("diversity term penalises over-used parents", p_div[-1] < p_div[0],
      f"used-9 p={p_div[-1]:.4f} vs unused p={p_div[0]:.4f}")

# outlier robustness: median/MAD should not let one huge value collapse the rest
p_out = build_aux_probabilities([0.1, 0.2, 0.3, 0.4, 1e6], +1, zero_children)
check("robust to a wild outlier (median/MAD)", np.isfinite(p_out).all() and p_out[:4].sum() > 0.01,
      f"non-outlier mass={p_out[:4].sum():.3f}")

# ------------------------------------------------- within-pool softmax really learns
print("\n== within-pool softmax over q_aux ==")
aux = ["caging", "hole", "connect"]
b = TwoLevelAuxBandit(aux_names=aux, seed=0)

p0 = b._softmax_probs()
check("softmax starts ~uniform (all q_aux=0), valid prob vector",
      abs(p0.sum() - 1) < 1e-9 and np.allclose(p0, 1/3, atol=0.02),
      f"p={np.round(p0,3)}")

for _ in range(40):                      # make 'hole' the best arm
    b.update("hole", 1.0)
    b.update("caging", 0.0)
    b.update("connect", 0.0)
p1 = b._softmax_probs()
check("softmax shifts mass to the higher-q aux",
      p1[aux.index("hole")] > p0[aux.index("hole")] + 0.1,
      f"P(hole): {p0[aux.index('hole')]:.3f} -> {p1[aux.index('hole')]:.3f}")
check("epsilon floor keeps every aux alive (>= eps/3)",
      (p1 >= b.epsilon/3 - 1e-9).all(), f"min p={p1.min():.3f}, floor={b.epsilon/3:.3f}")

# ------------------------------------------------------------------ beta control
print("\n== two-level bandit dynamics ==")
b = TwoLevelAuxBandit(aux_names=aux, seed=1)
arms = [b.select_arm() for _ in range(4000)]
frac_main = arms.count(MAIN_ARM) / len(arms)
check("selects only valid arms", set(arms) <= {MAIN_ARM, *aux})
check("respects beta_init=0.70 for P(main)", abs(frac_main - 0.70) < 0.03,
      f"P(main)={frac_main:.3f}")


def simulate(true_mean, n=1500, seed=2, noise=0.02):
    """Run the bandit against controlled per-arm reward distributions."""
    rng = np.random.default_rng(seed)
    bd = TwoLevelAuxBandit(aux_names=aux, seed=seed)
    for _ in range(n):
        a = bd.select_arm()
        bd.update(a, rng.normal(true_mean[a], noise))
    return bd


# scenario 1: one aux genuinely beats the oracle arm
bd = simulate({MAIN_ARM: 0.005, "caging": 0.005, "hole": 0.030, "connect": -0.005})
best_aux = max(bd.q_aux, key=bd.q_aux.get)
check("useful aux -> beta DECREASES below init (trusts structural exploration more)",
      bd.beta < 0.85 - 1e-6, f"beta={bd.beta:.3f}")
check("useful aux -> that aux has the highest Q", best_aux == "hole",
      f"Q={ {k: round(v,4) for k,v in bd.q_aux.items()} }")
check("harmful aux gets the lowest Q", min(bd.q_aux, key=bd.q_aux.get) == "connect")
check("beta stays within [floor, cap]", bd.beta_floor - 1e-9 <= bd.beta <= bd.beta_cap + 1e-9)

# scenario 2: every aux is worse than the oracle arm
bd2 = simulate({MAIN_ARM: 0.030, "caging": 0.0, "hole": 0.0, "connect": 0.0}, seed=3)
check("useless auxes -> beta RISES to the cap (graceful degradation to oracle-only)",
      bd2.beta >= bd2.beta_cap - 1e-9, f"beta={bd2.beta:.3f} (cap={bd2.beta_cap})")
check("aux pool still sampled (exploration floor, never starved)",
      sum(bd2.pulls[k] for k in aux) > 20,
      f"aux pulls={ {k: bd2.pulls[k] for k in aux} }")

# scenario 3: non-stationarity -- an aux is good, then stops being good
rng = np.random.default_rng(4)
bd3 = TwoLevelAuxBandit(aux_names=aux, seed=4)
for i in range(1200):
    a = bd3.select_arm()
    mean = 0.005
    if a == "hole":
        mean = 0.04 if i < 700 else -0.02      # regime flip
    bd3.update(a, rng.normal(mean, 0.02))
    if i == 690:
        q_hole_before = bd3.q_aux["hole"]
check("non-stationary: Q tracks a regime flip (recency-weighted, forgets stale evidence)",
      bd3.q_aux["hole"] < q_hole_before, f"Q(hole) {q_hole_before:+.4f} -> {bd3.q_aux['hole']:+.4f}")

# ------------------------------------- REALISTIC BUDGET: does it learn in 200 gens?
print("\n== realistic budget (n=200, the actual main-run length) ==")
bd200 = simulate({MAIN_ARM: 0.005, "caging": 0.005, "hole": 0.030, "connect": -0.005},
                 n=200, seed=7)
aux_pulls = {k: bd200.pulls[k] for k in aux}
check("each aux gets enough pulls to be evaluated in 200 gens",
      min(aux_pulls.values()) >= 8, f"pulls={aux_pulls}")
check("identifies the useful aux within 200 gens",
      max(bd200.q_aux, key=bd200.q_aux.get) == "hole",
      f"Q={ {k: round(v,4) for k,v in bd200.q_aux.items()} }")
check("beta responds (drops below init) within 200 gens",
      bd200.beta < 0.70 - 1e-6, f"beta={bd200.beta:.3f}")

bd200b = simulate({MAIN_ARM: 0.030, "caging": 0.0, "hole": 0.0, "connect": 0.0},
                  n=200, seed=8)
check("useless auxes in 200 gens -> beta rises (no false positive)",
      bd200b.beta > 0.70, f"beta={bd200b.beta:.3f}")

# ------------------------------------------------------------------------ summary
n_pass = sum(1 for _, ok in results if ok)
print(f"\n{n_pass}/{len(results)} checks passed")
for name, ok in results:
    if not ok:
        print(f"  FAILED: {name}")
# === END ALG2 MOD (aux-selection) ===
