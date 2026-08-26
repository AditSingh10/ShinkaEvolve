#!/usr/bin/env python3
# === PILOT MOD (aux-eval): post-hoc analysis. Reads the run DB, emits one CSV row per
# candidate. Built post-hoc (not appended live) to avoid races across concurrent eval
# subprocesses, and so it touches nothing in the loop. ===
"""
Emit the experiment log for one pilot run:
    generation, candidate_id, oracle_score, aux_name, aux_score, valid

oracle_score is the pure recorded combined_score (never the aux). aux_name/aux_score/valid
come from the private_metrics we recorded in evaluate.py. Use across runs to compare
best-oracle-score-so-far vs generation for none/m1/m2/m3.

Usage:
  python build_pilot_csv.py --results_dir results/pilot_m2 [--out pilot_m2.csv]
"""
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path


def build(results_dir: Path, out: Path):
    db = results_dir / "programs.sqlite"
    if not db.exists():
        sys.exit(f"No programs.sqlite under {results_dir}")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, generation, combined_score, correct, private_metrics "
        "FROM programs ORDER BY generation"
    ).fetchall()

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["generation", "candidate_id", "oracle_score", "aux_name", "aux_score", "valid"]
        )
        for r in rows:
            priv = json.loads(r["private_metrics"]) if r["private_metrics"] else {}
            # oracle_score: prefer the explicitly recorded value, fall back to combined_score
            oracle = priv.get("oracle_score", r["combined_score"])
            w.writerow([
                r["generation"],
                r["id"],
                f"{oracle:.6f}" if oracle is not None else "",
                priv.get("aux_name", ""),
                priv.get("aux_score", ""),
                bool(r["correct"]),
            ])
    print(f"Wrote {len(rows)} rows -> {out}")

    # Quick sanity summary printed to stdout.
    best = max((r["combined_score"] for r in rows if r["correct"]), default=None)
    aux_names = {json.loads(r["private_metrics"] or "{}").get("aux_name") for r in rows}
    print(f"  valid programs: {sum(1 for r in rows if r['correct'])}/{len(rows)}")
    print(f"  best valid oracle_score: {best}")
    print(f"  aux_name(s) seen: {aux_names - {None}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rd = Path(a.results_dir)
    out = Path(a.out) if a.out else Path(f"{rd.name}.csv")
    build(rd, out)
# === END PILOT MOD (aux-eval) ===
