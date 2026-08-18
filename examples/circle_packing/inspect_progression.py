#!/usr/bin/env python3
"""
Inspect how the circle-packing run improved over time.

Reconstructs the score -> feedback -> improvement loop from a finished run:
  - traces the lineage of the best VALID program back to the seed,
  - prints, for each step, the score the parent was given as feedback,
  - the LLM's stated reasoning (patch description) and the raw response file,
  - the child's resulting score.

Usage:
  python inspect_progression.py [--results_dir results/results_circle_async_small]
  python inspect_progression.py --gen 134        # deep-dive one generation
  python inspect_progression.py --top 10         # top-N valid solutions table
"""
import argparse
import json
import sqlite3
from pathlib import Path


def load(results_dir: Path):
    con = sqlite3.connect(results_dir / "programs.sqlite")
    con.row_factory = sqlite3.Row
    rows = con.execute("select * from programs").fetchall()
    return {r["id"]: r for r in rows}


def md(row):
    return json.loads(row["metadata"]) if row["metadata"] else {}


def perf_feedback(row):
    """Reproduce the exact 'performance_metrics' block the LLM is shown."""
    pub = json.loads(row["public_metrics"]) if row["public_metrics"] else {}
    s = f"Combined score to maximize: {row['combined_score']:.2f}\n"
    for k, v in pub.items():
        if k == "centers_str":
            continue
        s += f"{k}: {v}; "
    return s.strip().rstrip(";").strip()


def lineage(byid):
    best = max((r for r in byid.values() if r["correct"] == 1),
              key=lambda r: r["combined_score"])
    chain, cur = [], best
    while cur is not None:
        chain.append(cur)
        cur = byid.get(cur["parent_id"])
    return chain[::-1]


def show_top(byid, n):
    valid = sorted((r for r in byid.values() if r["correct"] == 1),
                   key=lambda r: r["combined_score"], reverse=True)[:n]
    print(f"{'rank':<5}{'gen':<5}{'score':<11}{'patch_name':<32}{'model':<22}")
    for i, r in enumerate(valid):
        m = md(r)
        print(f"{i+1:<5}{r['generation']:<5}{r['combined_score']:<11.5f}"
              f"{(m.get('patch_name') or '')[:30]:<32}{(m.get('model_name') or '')[:20]:<22}")


def show_lineage(byid, results_dir):
    chain = lineage(byid)
    print(f"Winning lineage: seed -> gen {chain[-1]['generation']} "
          f"(best valid = {chain[-1]['combined_score']:.5f}), {len(chain)} programs\n")
    prev = None
    for r in chain:
        m = md(r)
        d = "" if prev is None else f"  ({r['combined_score']-prev:+.4f})"
        print(f"gen {r['generation']:<4} score {r['combined_score']:.5f}{d}"
              f"  [{m.get('patch_name','seed')}]  {m.get('model_name','')}")
        prev = r["combined_score"]


def deep_dive(byid, results_dir, gen):
    child = next(r for r in byid.values() if r["generation"] == gen)
    parent = byid.get(child["parent_id"])
    print(f"=== STEP: gen {parent['generation'] if parent else '-'} -> gen {gen} ===\n")
    if parent:
        print(f"PARENT score {parent['combined_score']:.5f} (correct={bool(parent['correct'])})")
        print("Feedback block shown to the LLM:")
        print("    " + perf_feedback(parent).replace("\n", "\n    "))
        ptf = (parent["text_feedback"] or "").strip()
        if ptf:
            print(f"    text_feedback: {ptf[:200]}")
    print()
    resp = (results_dir / f"gen_{gen}" / "attempts" / "novelty_1" /
            "resample_1" / "patch_1" / "llm_response.txt")
    if resp.exists():
        print(f"RAW LLM RESPONSE ({resp}):\n")
        print(resp.read_text()[:2500])
    print(f"\nCHILD score {child['combined_score']:.5f} "
          f"(correct={bool(child['correct'])}) "
          f"delta {child['combined_score']-(parent['combined_score'] if parent else 0):+.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/results_circle_async_small")
    ap.add_argument("--gen", type=int, help="deep-dive a single generation")
    ap.add_argument("--top", type=int, help="show top-N valid solutions")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    byid = load(rd)
    if a.top:
        show_top(byid, a.top)
    elif a.gen is not None:
        deep_dive(byid, rd, a.gen)
    else:
        show_lineage(byid, rd)
