#!/usr/bin/env python3
"""Analysis for the steering ablation: does event-triggered steering help?
Primary endpoint: valid-yield (valid / total proposals), STEER vs baseline, one-sided MWU.
Secondary: runtime-crash rate (one-sided less), ceiling (best-so-far), novelty-rejection rate,
families discovered (steer). Both arms use the updated checker so failure classes are recoverable.
"""
import sqlite3, json, glob, sys
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, "/data/ikakkar/co-evolution")
from steering_replay import load_sequence
from shinka.core.steering import classify_failure

BASE = "/data/ikakkar/co-evolution/ShinkaEvolve/examples/circle_packing_alg2/results"
ARMS = {"baseline": f"{BASE}/pilot_none_abl_base_rep*",
        "steering": f"{BASE}/pilot_none_abl_steer_rep*"}


def run_stats(rd):
    c = sqlite3.connect(f"{rd}/programs.sqlite")
    progs = c.execute("select correct,combined_score from programs").fetchall()
    nov = c.execute("select count(*) from attempt_log where stage='novelty'").fetchone()[0]
    n_prog = len(progs)
    valid = sum(1 for cor, _ in progs if cor == 1)
    total = n_prog + nov
    best = max((s for cor, s in progs if cor == 1 and s is not None), default=0.0)
    # runtime-crash rate among evaluated children (needs gen-dir errors)
    seq = [it for it in load_sequence(rd) if it["kind"] in ("valid", "invalid")]
    rt = sum(1 for it in seq if "runtime" in classify_failure(
        it["kind"] == "valid", it["violations"], it["error"]))
    crash_rate = rt / len(seq) if seq else float("nan")
    return dict(valid_yield=valid / total if total else 0.0, ceiling=best,
                novelty_rate=nov / total if total else 0.0, crash_rate=crash_rate,
                total=total, valid=valid)


def mw(a, b, alt):
    try:
        return mannwhitneyu(np.array(a, float), np.array(b, float), alternative=alt).pvalue
    except Exception:
        return float("nan")


def main():
    data = {}
    for arm, pat in ARMS.items():
        runs = sorted(glob.glob(pat))
        data[arm] = [run_stats(r) for r in runs]
        print(f"{arm}: {len(runs)} runs")
    print("=" * 78)
    print("STEERING ABLATION — does it help?")
    print("=" * 78)
    b, s = data["baseline"], data["steering"]
    if not b or not s:
        print("waiting for runs to finish..."); return
    def col(rows, k): return [r[k] for r in rows]
    print(f"\n{'metric':16s}{'baseline med':>14}{'steering med':>14}{'MWU p':>10}   verdict")
    print("-" * 74)
    rows = [
        ("valid-yield", "valid_yield", "greater", "STEER higher"),
        ("crash-rate", "crash_rate", "less", "STEER lower"),
        ("ceiling", "ceiling", "greater", "STEER higher"),
        ("novelty-rate", "novelty_rate", "two-sided", "differ"),
    ]
    for label, key, alt, hyp in rows:
        bv, sv = col(b, key), col(s, key)
        p = mw(sv, bv, alt) if alt != "two-sided" else mw(sv, bv, "two-sided")
        sig = "**SIGNIFICANT**" if (p < 0.05) else "n.s."
        print(f"{label:16s}{np.median(bv):>14.3f}{np.median(sv):>14.3f}{p:>10.3f}   {hyp}: {sig}")
    print("\nprimary endpoint = valid-yield (one-sided, STEER > baseline).")
    print("=" * 78)


if __name__ == "__main__":
    main()
