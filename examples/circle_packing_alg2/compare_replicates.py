#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): aggregate the replicate study. ===
"""Pool independent replicates per arm and test whether aux-guided selection beats
oracle-only selection on the held-out oracle.

Reports per-replicate finals, arm means/spread, an exact rank-sum (Mann-Whitney U)
p-value, and -- crucially -- the within-arm spread, which is the noise floor any
claimed effect must clear.
"""
import argparse
import glob
import itertools
import os
import sqlite3

import numpy as np


def best_valid(run_dir):
    db = os.path.join(run_dir, "programs.sqlite")
    if not os.path.exists(db):
        return None
    con = sqlite3.connect(db)
    r = con.execute(
        "select max(combined_score) from programs where correct=1"
    ).fetchone()[0]
    return float(r) if r is not None else None


def curve(run_dir):
    """best-valid-so-far by generation."""
    con = sqlite3.connect(os.path.join(run_dir, "programs.sqlite"))
    con.row_factory = sqlite3.Row
    b, out = 0.0, {}
    for r in con.execute(
        "select generation, combined_score, correct from programs order by generation, rowid"
    ):
        if r["correct"] and r["combined_score"] > b:
            b = r["combined_score"]
        out[r["generation"]] = b
    return out


def mannwhitney_p(a, b):
    """Exact two-sided rank-sum p-value (tiny samples, so enumerate)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None
    obs = sum(1 for x in a for y in b if x > y) + 0.5 * sum(
        1 for x in a for y in b if x == y
    )
    pool = list(a) + list(b)
    cnt = tot = 0
    for idx in itertools.combinations(range(n1 + n2), n1):
        g1 = [pool[i] for i in idx]
        g2 = [pool[i] for i in range(n1 + n2) if i not in idx]
        u = sum(1 for x in g1 for y in g2 if x > y) + 0.5 * sum(
            1 for x in g1 for y in g2 if x == y
        )
        tot += 1
        if abs(u - n1 * n2 / 2) >= abs(obs - n1 * n2 / 2):
            cnt += 1
    return cnt / tot


def main(root):
    arms = {}
    for arm in ("treatment", "control"):
        runs = sorted(glob.glob(os.path.join(root, f"pilot_none_{arm}_rep*")))
        vals = [(os.path.basename(d).split("_")[-1], best_valid(d)) for d in runs]
        arms[arm] = [(t, v) for t, v in vals if v is not None]

    print("=== per-replicate best valid oracle score ===")
    for arm, vals in arms.items():
        s = "  ".join(f"{t}:{v:.4f}" for t, v in vals)
        print(f"  {arm:>9}: {s or '(none yet)'}")

    print("\n=== arm summary ===")
    stats = {}
    for arm, vals in arms.items():
        if not vals:
            continue
        x = np.array([v for _, v in vals])
        stats[arm] = x
        print(f"  {arm:>9}: n={len(x)} mean={x.mean():.4f} median={np.median(x):.4f} "
              f"sd={x.std(ddof=1) if len(x)>1 else float('nan'):.4f} "
              f"range=[{x.min():.4f}, {x.max():.4f}]")

    if len(stats) == 2 and all(len(v) >= 2 for v in stats.values()):
        t, c = stats["treatment"], stats["control"]
        diff = t.mean() - c.mean()
        noise = max(t.std(ddof=1), c.std(ddof=1))
        p = mannwhitney_p(list(t), list(c))
        print("\n=== verdict ===")
        print(f"  mean difference (treatment - control) = {diff:+.4f}")
        print(f"  within-arm sd (noise floor)           =  {noise:.4f}")
        print(f"  exact rank-sum p                      =  {p:.4f}" if p is not None else "")
        if abs(diff) < noise:
            print("  -> effect is SMALLER than the noise floor: not distinguishable.")
        elif p is not None and p < 0.05:
            print("  -> effect exceeds noise and is significant at p<0.05.")
        else:
            print("  -> effect exceeds the noise floor but is not significant at this n.")
    else:
        print("\n  (need >=2 replicates per arm for a verdict)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    main(ap.parse_args().root)
# === END ALG2 MOD (aux-selection) ===
