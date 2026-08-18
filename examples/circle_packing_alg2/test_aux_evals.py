#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): sanity test for the structural auxes. ===
"""Validate the three auxes AND their direction priors on real packings.

Loads saved configurations (extra.npz) from previous runs across the score range and
prints aux values sorted by oracle score. If the direction priors are right we expect,
as the oracle score rises:  caging UP, hole DOWN, connect UP.

Usage: python test_aux_evals.py [--run_dir <results dir with gen_*/results/extra.npz>]
"""
import argparse
import glob
import json
import os
import sqlite3

import numpy as np

from aux_evals import AUX_SPECS, compute_all


def load_configs(run_dir):
    """Yield (generation, oracle_score, valid, centers, radii) from a finished run."""
    db = os.path.join(run_dir, "programs.sqlite")
    scores = {}
    if os.path.exists(db):
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        for r in con.execute("select generation, combined_score, correct from programs"):
            # keep the best-scoring record per generation
            g = r["generation"]
            if g not in scores or r["combined_score"] > scores[g][0]:
                scores[g] = (r["combined_score"], bool(r["correct"]))

    out = []
    for f in glob.glob(os.path.join(run_dir, "gen_*", "results", "extra.npz")):
        gen = int(os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[1])
        try:
            d = np.load(f)
            centers, radii = d["centers"], d["radii"]
        except Exception:
            continue
        if centers.shape[0] != len(radii):
            continue
        sc, valid = scores.get(gen, (float("nan"), None))
        out.append((gen, sc, valid, centers, radii))
    return sorted(out, key=lambda t: (np.isnan(t[1]), t[1]))


def main(run_dir):
    rows = load_configs(run_dir)
    if not rows:
        print(f"No packings found under {run_dir}")
        return
    names = list(AUX_SPECS.keys())
    print(f"{'gen':>5} {'oracle':>8} {'valid':>6} | " + " ".join(f"{n:>9}" for n in names))
    print("-" * (23 + 10 * len(names)))
    keep = []
    for gen, sc, valid, centers, radii in rows:
        aux = compute_all(centers, radii)
        keep.append((sc, valid, aux))
        print(f"{gen:>5} {sc:>8.4f} {str(valid):>6} | " +
              " ".join(f"{aux[n]:>9.4f}" for n in names))

    # correlation of each aux with the oracle, over VALID configs only
    v = [(sc, a) for sc, valid, a in keep if valid and not np.isnan(sc)]
    if len(v) >= 3:
        s = np.array([x[0] for x in v])
        print(f"\nSpearman-style rank correlation with oracle (valid only, n={len(v)}):")
        for n in names:
            a = np.array([x[1][n] for x in v])
            if a.std() < 1e-12:
                print(f"  {n:>9}: constant (no signal)")
                continue
            rs = np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(a)))[0, 1]
            prior = AUX_SPECS[n][1]
            agree = "matches prior" if np.sign(rs) == np.sign(prior) else "OPPOSES prior"
            print(f"  {n:>9}: rho={rs:+.3f}  (direction prior {prior:+d}) -> {agree}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        default="../circle_packing/results/results_circle_async_small",
        help="a finished run directory containing gen_*/results/extra.npz",
    )
    a = ap.parse_args()
    main(a.run_dir)
# === END ALG2 MOD (aux-selection) ===
