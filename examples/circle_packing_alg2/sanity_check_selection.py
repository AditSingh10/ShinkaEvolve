#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): end-to-end sanity checks on REAL data. ===
"""Does aux-shaped selection actually do anything, and are the knobs calibrated?

Unit tests prove the bandit is internally correct; this checks it is *meaningful*:
  1. Do the aux distributions P_k pick DIFFERENT parents than the oracle P_0?
     (If P_k ~= P_0 the whole intervention is a no-op.)
  2. How much oracle quality does an aux arm give up per pull? (cost of exploration)
  3. What is the real scale of improvements? -> calibrates kappa in the beta update.
"""
import glob
import os
import sqlite3

import numpy as np

from aux_evals import AUX_SPECS, compute_all
from shinka.database.aux_selection import build_aux_probabilities

RUN = "../circle_packing/results/results_circle_async_small"


def load_archive(run_dir):
    """Valid programs with their aux vector, oracle score and children count."""
    con = sqlite3.connect(os.path.join(run_dir, "programs.sqlite"))
    con.row_factory = sqlite3.Row
    meta = {}
    for r in con.execute("select id, generation, combined_score, correct, parent_id, children_count from programs"):
        meta[r["generation"]] = dict(r)
    rows = []
    for f in glob.glob(os.path.join(run_dir, "gen_*", "results", "extra.npz")):
        gen = int(os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[1])
        m = meta.get(gen)
        if not m or not m["correct"]:
            continue
        try:
            d = np.load(f)
            c, r_ = d["centers"], d["radii"]
        except Exception:
            continue
        rows.append({"gen": gen, "oracle": m["combined_score"],
                     "children": m["children_count"] or 0, "aux": compute_all(c, r_)})
    return rows


def tv_distance(p, q):
    return 0.5 * float(np.abs(p - q).sum())


rows = load_archive(RUN)
print(f"archive sample: {len(rows)} valid programs "
      f"(oracle {min(r['oracle'] for r in rows):.3f} .. {max(r['oracle'] for r in rows):.3f})\n")

oracle = [r["oracle"] for r in rows]
children = [r["children"] for r in rows]
p0 = build_aux_probabilities(oracle, +1, children)      # same math as the oracle sampler

print("== 1. Do aux distributions differ from the oracle distribution? ==")
print(f"{'aux':>9} {'TVdist':>8} {'top1 same?':>11} {'overlap@5':>10}  interpretation")
top0 = set(np.argsort(p0)[-5:])
for name, (_, direction) in AUX_SPECS.items():
    vals = [r["aux"][name] for r in rows]
    pk = build_aux_probabilities(vals, direction, children)
    tv = tv_distance(p0, pk)
    same_top1 = int(np.argmax(pk)) == int(np.argmax(p0))
    ov = len(top0 & set(np.argsort(pk)[-5:])) / 5
    verdict = ("NO-OP (~identical to oracle)" if tv < 0.10 else
               "mild divergence" if tv < 0.35 else "genuinely different parents")
    print(f"{name:>9} {tv:>8.3f} {str(same_top1):>11} {ov:>10.1f}  {verdict}")

print("\n== 2. Oracle quality given up per aux pull ==")
e0 = float(np.dot(p0, oracle))
print(f"  E[oracle | P_0 ] = {e0:.4f}   (best-in-archive = {max(oracle):.4f})")
for name, (_, direction) in AUX_SPECS.items():
    vals = [r["aux"][name] for r in rows]
    pk = build_aux_probabilities(vals, direction, children)
    ek = float(np.dot(pk, oracle))
    print(f"  E[oracle | P_{name:<8}] = {ek:.4f}   delta vs oracle arm = {ek - e0:+.4f}")

print("\n== 3. Real improvement scale (calibrates kappa) ==")
con = sqlite3.connect(os.path.join(RUN, "programs.sqlite"))
con.row_factory = sqlite3.Row
by_id = {r["id"]: r for r in con.execute("select id, combined_score, correct, parent_id from programs")}
imps = []
for r in by_id.values():
    p = by_id.get(r["parent_id"])
    if p is None:
        continue
    imps.append((r["combined_score"] - p["combined_score"]) if r["correct"] else 0.0)
imps = np.array(imps)
if imps.size:
    print(f"  n={imps.size}  mean={imps.mean():+.4f}  median={np.median(imps):+.4f}  "
          f"std={imps.std():.4f}")
    print(f"  |improvement| percentiles: p50={np.percentile(np.abs(imps),50):.4f}  "
          f"p90={np.percentile(np.abs(imps),90):.4f}")
    typ = float(np.percentile(np.abs(imps), 50)) or 1e-3
    print(f"  -> with kappa=50, a typical advantage {typ:.4f} gives "
          f"tanh(50*{typ:.4f})={np.tanh(50*typ):.3f} of the max beta step "
          f"({0.05*np.tanh(50*typ):+.4f} per update)")
# === END ALG2 MOD (aux-selection) ===
