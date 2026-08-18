#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): tests for the 2048 board auxes + divergence helpers. ===
"""Validate each aux on hand-built boards where the right answer is obvious, and check
the aux-vs-oracle divergence helpers. Run BEFORE launching the experiment."""
import numpy as np

from aux_evals import (
    a_empty, a_monotonic, a_smooth, a_corner, a_merge, compute_all, AUX_NAMES,
)
from shinka.database.aux_selection import (
    oracle_probabilities, js_divergence, total_variation, aux_oracle_divergences,
)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# boards are EXPONENTS: 0 empty, k => tile 2**k
EMPTY = np.zeros((4, 4), dtype=int)
FULL = np.arange(1, 17, dtype=int).reshape(4, 4)          # 16 distinct nonzero tiles
CORNER_MAX = np.array([[11, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
CENTER_MAX = np.array([[0, 0, 0, 0], [0, 11, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
MONO = np.array([[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]])   # rows sorted
CHECKER = np.array([[1, 5, 1, 5], [5, 1, 5, 1], [1, 5, 1, 5], [5, 1, 5, 1]])  # rough
ALL_TWOS = np.full((4, 4), 1, dtype=int)                 # every cell = tile "2"

print("\n== a_empty ==")
check("all-empty board -> 1.0", abs(a_empty(EMPTY) - 1.0) < 1e-9)
check("full board -> 0.0", abs(a_empty(FULL) - 0.0) < 1e-9)
check("half-ish board in (0,1)", 0 < a_empty(CORNER_MAX) < 1)

print("\n== a_corner ==")
check("max tile in corner -> 1.0", abs(a_corner(CORNER_MAX) - 1.0) < 1e-9)
check("max tile in center -> 0.0", abs(a_corner(CENTER_MAX) - 0.0) < 1e-9)

print("\n== a_monotonic ==")
check("perfectly ordered rows score high", a_monotonic(MONO) > 0.9, f"{a_monotonic(MONO):.3f}")
check("checkerboard scores lower than monotonic", a_monotonic(CHECKER) < a_monotonic(MONO),
      f"checker={a_monotonic(CHECKER):.3f} < mono={a_monotonic(MONO):.3f}")
check("monotonic in [0,1]", 0 <= a_monotonic(CHECKER) <= 1)

print("\n== a_smooth ==")
check("uniform board is maximally smooth (=1.0)", abs(a_smooth(ALL_TWOS) - 1.0) < 1e-9,
      f"{a_smooth(ALL_TWOS):.3f}")
check("high-contrast board is less smooth", a_smooth(CHECKER) < a_smooth(ALL_TWOS),
      f"checker={a_smooth(CHECKER):.3f} < uniform={a_smooth(ALL_TWOS):.3f}")
check("smooth in (0,1]", 0 < a_smooth(CHECKER) <= 1)

print("\n== a_merge ==")
check("all-equal board has high merge potential", a_merge(ALL_TWOS) > 0.9,
      f"{a_merge(ALL_TWOS):.3f}")
check("all-distinct board has zero merges", abs(a_merge(FULL) - 0.0) < 1e-9)
check("merge in [0,1]", 0 <= a_merge(CHECKER) <= 1)

print("\n== trajectory averaging + compute_all ==")
traj = np.stack([EMPTY, ALL_TWOS, CORNER_MAX])
allv = compute_all(traj)
check("compute_all returns all five auxes", set(allv) == set(AUX_NAMES), f"{list(allv)}")
check("all aux values finite", all(np.isfinite(v) for v in allv.values()), f"{allv}")
check("empty over trajectory = mean of per-board", abs(allv["empty"] -
      np.mean([a_empty(EMPTY), a_empty(ALL_TWOS), a_empty(CORNER_MAX)])) < 1e-9)

print("\n== divergence helpers ==")
oracle = [0.1, 0.5, 0.9, 0.3, 0.7]
children = [0, 0, 0, 0, 0]
p_or = oracle_probabilities(oracle, children)
check("oracle probs sum to 1 / favour higher score", abs(p_or.sum() - 1) < 1e-9
      and np.argmax(p_or) == 2, f"p={np.round(p_or,3)}")
check("JS(p,p)=0 (identical dists -> no-op aux)", abs(js_divergence(p_or, p_or)) < 1e-12)
check("TV(p,p)=0", abs(total_variation(p_or, p_or)) < 1e-12)

# an aux perfectly ANTI-correlated with the oracle should diverge strongly
anti = {"anti": [-v for v in oracle]}           # low where oracle is high
div = aux_oracle_divergences(["anti"], {"anti": +1}, oracle, children, anti, residualize=False)
check("anti-correlated aux -> large JS divergence", div["anti"]["js"] > 0.3,
      f"js={div['anti']['js']:.3f}, tv={div['anti']['tv']:.3f}")

# RAW aux identical to the oracle == a no-op: same distribution, ~0 divergence.
same = {"same": list(oracle)}
raw = aux_oracle_divergences(["same"], {"same": +1}, oracle, children, same, residualize=False)
check("RAW aux == oracle -> ~0 divergence (this is the no-op residualization fixes)",
      raw["same"]["js"] < 1e-6, f"js={raw['same']['js']:.3e}")

# Residualizing that SAME aux removes the oracle-predictable part -> residual ~0 ->
# selection falls back to ~uniform, which DIVERGES from the oracle's peaked dist. This is
# the whole point: residualization turns a redundant aux into an independent signal.
res = aux_oracle_divergences(["same"], {"same": +1}, oracle, children, same, residualize=True)
check("residualization INCREASES divergence of a redundant aux (not a no-op)",
      res["same"]["js"] > raw["same"]["js"] + 0.1,
      f"raw js={raw['same']['js']:.3f} -> residualized js={res['same']['js']:.3f}")

n_pass = sum(1 for _, ok in results if ok)
print(f"\n{n_pass}/{len(results)} checks passed")
for name, ok in results:
    if not ok:
        print(f"  FAILED: {name}")
# === END ALG2 MOD (aux-selection) ===
