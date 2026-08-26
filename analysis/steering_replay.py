#!/usr/bin/env python3
"""OBSERVE-ONLY diagnostic: replay the steering controllers over a real circle-packing run
WITHOUT touching the evolutionary runner. Verifies each subcomponent passes info as expected:
  code -> LLM summary -> embed -> family assignment -> entropy/gamma   (diversity controller)
  violations -> EMA -> lambda -> witnesses                             (validity controller)

Goal: confirm the controllers DIAGNOSE the known collapse on a baseline run before we ever let
them steer. No selection/prompt is changed here.

Usage:
  uv run --project ShinkaEvolve python steering_replay.py <run_dir> [--B 200]
"""
import sqlite3, json, argparse, os, hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

import sys
sys.path.insert(0, "/data/ikakkar/co-evolution/ShinkaEvolve")
from shinka.core.steering import (
    SteeringConfig, FamilyManager, DiversityController, ValidityController, SteeringPolicy,
    classify_failure,
)

GEN = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
EMB = OpenAI(base_url="http://localhost:8001/v1", api_key="local")
SUMMARY_SYS = (
    "You are an expert at classifying optimization algorithms. Given a Python program that packs "
    "circles, output ONLY a short structured summary in exactly this form, one field per line:\n"
    "solver_family:\ninitialization:\nglobal_search:\nlocal_search:\nrepair_strategy:\n"
    "representation:\nkey_mechanism:\n"
    "Be terse (a few words per field). Describe the ALGORITHM, not code style."
)


def summarize(code: str, cache: dict) -> str:
    h = hashlib.md5(code.encode()).hexdigest()
    if h in cache:
        return cache[h]
    r = GEN.chat.completions.create(
        model="qwen32b", temperature=0.0, max_tokens=256,
        messages=[{"role": "system", "content": SUMMARY_SYS},
                  {"role": "user", "content": code[:6000]}])
    s = r.choices[0].message.content.strip()
    cache[h] = s
    return s


def embed(text: str, cache: dict) -> np.ndarray:
    h = hashlib.md5(text.encode()).hexdigest()
    if h in cache:
        return np.array(cache[h])
    v = EMB.embeddings.create(model="qwen-embed", input=text).data[0].embedding
    cache[h] = v
    return np.array(v)


def load_sequence(run_dir: str):
    """Ordered proposal stream: evaluated programs + novelty-rejected attempts, by generation."""
    db = f"{run_dir}/programs.sqlite"
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    seq = []
    for r in c.execute("select id,parent_id,generation,correct,combined_score,code,"
                       "private_metrics from programs"):
        prv = json.loads(r["private_metrics"] or "{}")
        err = None
        if r["correct"] != 1:  # recover the runtime exception string for crashed children
            cj = Path(run_dir) / f"gen_{r['generation']}" / "results" / "correct.json"
            if cj.exists():
                try:
                    err = json.loads(cj.read_text()).get("error")
                except Exception:
                    err = None
        seq.append(dict(gen=r["generation"], kind="valid" if r["correct"] == 1 else "invalid",
                        pid=r["id"], parent=r["parent_id"], score=r["combined_score"] or 0.0,
                        code=r["code"] or "", violations=prv.get("violations"), error=err))
    for r in c.execute("select generation,details from attempt_log where stage='novelty'"):
        seq.append(dict(gen=r["generation"], kind="novelty", pid=None, parent=None,
                        score=0.0, code="", violations=None))
    seq.sort(key=lambda d: (d["gen"], 0 if d["kind"] != "novelty" else 1))
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--B", type=int, default=200)
    args = ap.parse_args()

    cfg = SteeringConfig(enable_diversity_controller=True, enable_validity_controller=True)
    fam = FamilyManager(cfg); div = DiversityController(cfg); val = ValidityController(cfg)
    pol = SteeringPolicy(cfg, fam, div, val)

    cache_path = Path(args.run_dir) / "steering_replay_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    scache, ecache = cache.get("summ", {}), cache.get("emb", {})

    seq = load_sequence(args.run_dir)
    print(f"replaying {len(seq)} proposals from {args.run_dir}  (B={args.B})")
    recs = []
    n_summ = 0
    for b, item in enumerate(seq, 1):
        t = b / args.B
        # validity controller: every EVALUATED child (valid or invalid), never novelty-rejected.
        # classify_failure covers ALL invalid types: runtime crash, malformed, boundary, overlap.
        if item["kind"] in ("valid", "invalid"):
            pf = fam.family_of(item["parent"]) if item["parent"] else None
            failure = classify_failure(item["kind"] == "valid", item["violations"], item["error"])
            val.update_child(failure, pf)
        # diversity: only ADMITTED VALID children get a family
        new_family = False
        if item["kind"] == "valid" and item["code"]:
            summ = summarize(item["code"], scache); n_summ += 1
            e = embed(summ, ecache)
            k_before = fam.num_families()
            fam.assign(item["pid"], e, item["score"], summ)
            new_family = fam.num_families() > k_before
        div.observe(fam.counts(), t)
        # controller checks (hysteresis) after warmup at the check interval
        if b > cfg.controller_warmup and b % cfg.controller_check_interval == 0:
            div.check(); val.check()
        ks = val.active_constraint()
        recs.append(dict(b=b, t=t, kind=item["kind"], score=item["score"],
                         K=fam.num_families(), H=div.H, H_target=div.H_target, gamma=div.gamma,
                         dominant=div.dominant,
                         lam_runtime=val.lam["runtime"], lam_malformed=val.lam["malformed"],
                         lam_overlap=val.lam["overlap"], lam_boundary=val.lam["boundary"],
                         r_runtime=val.r["runtime"], r_overlap=val.r["overlap"],
                         active_k=ks, div_active=div.active, val_active=val.active,
                         mode=pol.current_mode(), new_family=new_family))

    cache_path.write_text(json.dumps({"summ": scache, "emb": ecache}))
    out_jsonl = Path(args.run_dir) / "steering_replay.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r) for r in recs))
    print(f"{n_summ} summaries; wrote {out_jsonl}")

    # ---------------- diagnostic figure ----------------
    R = recs
    bs = [r["b"] for r in R]
    C = {"valid": "#1baf7a", "invalid": "#e34948", "novelty": "#eda100"}
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.subplots_adjust(hspace=0.28, top=0.94, bottom=0.06, left=0.08, right=0.98)

    # A: best-so-far + outcome rug
    best, line = -1, []
    for r in R:
        if r["kind"] == "valid" and r["score"] > best: best = r["score"]
        line.append(best if best > 0 else np.nan)
    ax[0].plot(bs, line, color="black", lw=1.6)
    y0 = np.nanmin(line); y1 = np.nanmax(line); rng = max(y1 - y0, 1e-6)
    for r in R:
        ax[0].vlines(r["b"], y0 - 0.16*rng, y0 - 0.09*rng, color=C[r["kind"]], lw=1.2)
    ax[0].set_ylim(y0 - 0.2*rng, y1 + 0.08*rng)
    ax[0].set_ylabel("best-so-far"); ax[0].set_title("A. Search progress + per-proposal outcome (rug)", loc="left")

    # B: families K + entropy H vs target
    axb = ax[1]
    axb.plot(bs, [r["K"] for r in R], color="#3b6fd6", lw=1.8, label="num families K")
    axb.set_ylabel("K (families)", color="#3b6fd6"); axb.tick_params(axis="y", labelcolor="#3b6fd6")
    axb2 = axb.twinx()
    axb2.plot(bs, [r["H"] for r in R], color="#111", lw=1.6, label="family entropy H")
    axb2.plot(bs, [r["H_target"] for r in R], color="#888", lw=1.4, ls="--", label="desired H*(t)")
    axb2.set_ylabel("entropy"); axb2.set_ylim(-0.05, 1.05)
    axb.set_title("B. Algorithm-family collapse: K stuck low, H below desired", loc="left")
    l1,la1=axb.get_legend_handles_labels(); l2,la2=axb2.get_legend_handles_labels()
    axb.legend(l1+l2, la1+la2, loc="center right", fontsize=8)

    # C: gamma + thresholds + diversity-active shading
    axc = ax[2]
    axc.plot(bs, [r["gamma"] for r in R], color="#b5179e", lw=1.8, label="gamma")
    axc.axhline(cfg.gamma_on, color="#b5179e", ls=":", lw=1, alpha=0.7)
    axc.axhline(cfg.gamma_off, color="#b5179e", ls=":", lw=1, alpha=0.4)
    for r in R:
        if r["div_active"]:
            axc.axvspan(r["b"]-0.5, r["b"]+0.5, color="#b5179e", alpha=0.06)
    axc.set_ylabel("gamma"); axc.set_ylim(-0.02, 1.02)
    axc.set_title("C. Diversity pressure gamma (shaded = DIVERSITY_STEERING would be ON)", loc="left")
    axc.legend(loc="upper right", fontsize=8)

    # D: lambda per failure class + validity-active shading
    axd = ax[3]
    axd.plot(bs, [r["lam_runtime"] for r in R], color="#7b2d8e", lw=2.0, label="lambda runtime (crash)")
    axd.plot(bs, [r["lam_overlap"] for r in R], color="#e34948", lw=1.8, label="lambda overlap")
    axd.plot(bs, [r["lam_boundary"] for r in R], color="#f08a00", lw=1.4, label="lambda boundary")
    axd.plot(bs, [r["lam_malformed"] for r in R], color="#2b8", lw=1.0, alpha=0.7, label="lambda malformed")
    axd.axhline(cfg.lambda_on, color="#333", ls=":", lw=1, alpha=0.6)
    axd.axhline(cfg.lambda_off, color="#333", ls=":", lw=1, alpha=0.3)
    for r in R:
        if r["val_active"]:
            axd.axvspan(r["b"]-0.5, r["b"]+0.5, color="#7b2d8e", alpha=0.05)
    axd.set_ylabel("lambda"); axd.set_ylim(-0.02, 1.02)
    axd.set_title("D. Validity pressure lambda per failure class "
                  "(shaded = VALIDITY_STEERING would be ON)", loc="left")
    axd.set_xlabel("proposal number b"); axd.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle("Observe-only controller replay — do the controllers DIAGNOSE the collapse?",
                 x=0.08, ha="left", fontsize=13, weight="bold")
    out_png = Path(args.run_dir) / "steering_replay.png"
    fig.savefig(out_png, dpi=130); print("wrote", out_png)

    # console summary
    frac_div = np.mean([r["div_active"] for r in R]); frac_val = np.mean([r["val_active"] for r in R])
    print(f"\nSUMMARY: final K={R[-1]['K']}  max gamma={max(r['gamma'] for r in R):.2f}")
    for cls in ("runtime", "overlap", "boundary", "malformed"):
        print(f"   max lambda_{cls:9s} = {max(r.get('lam_'+cls, 0) for r in R):.2f}")
    print(f"   diversity-active {100*frac_div:.0f}% of proposals  |  validity-active {100*frac_val:.0f}%")


if __name__ == "__main__":
    main()
