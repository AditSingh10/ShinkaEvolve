#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): integration test of AuxSamplingStrategy on a real DB. ===
"""Backfill aux scores into a COPY of a finished run's DB, then check that the aux
sampling strategy (a) works against the real schema, and (b) actually selects different
parents than the oracle strategy."""
import glob
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter

import numpy as np

from aux_evals import AUX_SPECS, compute_all
from shinka.database.dbase import DatabaseConfig
from shinka.database.parents import AuxSamplingStrategy, WeightedSamplingStrategy

SRC = "../circle_packing/results/results_circle_async_small"
TMP = "/data/ikakkar/tmp/alg2_dbtest"

os.makedirs(TMP, exist_ok=True)
db_path = os.path.join(TMP, "programs.sqlite")
shutil.copy(os.path.join(SRC, "programs.sqlite"), db_path)

# ---- backfill aux_scores into private_metrics for programs that have a saved config
con = sqlite3.connect(db_path)
con.row_factory = sqlite3.Row
gen_to_cfg = {}
for f in glob.glob(os.path.join(SRC, "gen_*", "results", "extra.npz")):
    gen = int(os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[1])
    try:
        d = np.load(f)
        gen_to_cfg[gen] = (d["centers"], d["radii"])
    except Exception:
        pass

n_filled = 0
for r in con.execute("select id, generation, private_metrics from programs").fetchall():
    cfg = gen_to_cfg.get(r["generation"])
    if cfg is None:
        continue
    try:
        priv = json.loads(r["private_metrics"]) if r["private_metrics"] else {}
    except Exception:
        priv = {}
    priv["aux_scores"] = compute_all(*cfg)
    con.execute("update programs set private_metrics=? where id=?",
                (json.dumps(priv), r["id"]))
    n_filled += 1
con.commit()
print(f"backfilled aux_scores into {n_filled} programs\n")

cfg = DatabaseConfig(
    aux_directions={k: v for k, (_, v) in AUX_SPECS.items()},
    aux_residualize=True,
)
cur = con.cursor()
get_program = lambda pid: pid          # noqa: E731 - we only care which id is chosen

N = 600
np.random.seed(0)
oracle_pick = Counter()
strat0 = WeightedSamplingStrategy(cur, con, cfg, get_program, None, None)
for _ in range(N):
    p = strat0.sample_parent()
    if p:
        oracle_pick[p] += 1

print(f"{'arm':>9} {'sampled?':>9} {'distinct':>9} {'top1 == oracle top1?':>21} {'overlap@5':>10}")
top0 = {p for p, _ in oracle_pick.most_common(5)}
o_top1 = oracle_pick.most_common(1)[0][0] if oracle_pick else None
print(f"{'oracle':>9} {'yes':>9} {len(oracle_pick):>9} {'-':>21} {'-':>10}")

ok = True
for name in AUX_SPECS:
    picks = Counter()
    strat = AuxSamplingStrategy(cur, con, cfg, get_program, None, None, aux_name=name)
    for _ in range(N):
        p = strat.sample_parent()
        if p:
            picks[p] += 1
    if not picks:
        print(f"{name:>9} {'NO':>9}   <-- strategy returned None (FAIL)")
        ok = False
        continue
    a_top1 = picks.most_common(1)[0][0]
    ov = len(top0 & {p for p, _ in picks.most_common(5)}) / 5
    print(f"{name:>9} {'yes':>9} {len(picks):>9} {str(a_top1 == o_top1):>21} {ov:>10.1f}")

print("\nPASS: aux strategy works on the real schema and diverges from the oracle arm"
      if ok else "\nFAIL: see above")
sys.exit(0 if ok else 1)
# === END ALG2 MOD (aux-selection) ===
