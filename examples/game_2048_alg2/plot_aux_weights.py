#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): which auxes earn weight? (good vs bad identification) ===
"""Two-panel figure from a treatment run's aux_bandit_history.jsonl:
  (top)    q value of each aux + the oracle arm over the bandit clock -- which structural
           signals the bandit learned to trust (high q) vs discard (low q).
  (bottom) beta = P(sample oracle arm), with floor/init/cap guides.
Usage: python plot_aux_weights.py [results/treatment_rep1]"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/treatment_rep1"

# reference palette: 5 categorical slots (blue/orange/aqua/yellow/magenta) + gray oracle
AUX_COLORS = {
    "empty": "#2a78d6", "monotonic": "#eb6834", "smooth": "#1baf7a",
    "corner": "#eda100", "merge": "#e87ba4",
}
C_MAIN = "#52514e"     # oracle arm -- neutral gray (it is the baseline, not a category)
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#f0eff0"

recs = [json.loads(l) for l in open(f"{RUN}/aux_bandit_history.jsonl")]
u = np.arange(1, len(recs) + 1)
aux_names = [k[2:] for k in recs[-1] if k.startswith("q_") and k != "q_main"]
qm = np.array([r["q_main"] for r in recs])
beta = np.array([r["beta"] for r in recs])
Q = {a: np.array([r.get(f"q_{a}", 0.0) for r in recs]) for a in aux_names}
finalpull = {a: recs[-1].get(f"pulls_{a}", 0) for a in aux_names}

fig, (axq, axb) = plt.subplots(
    2, 1, figsize=(8.0, 6.4), dpi=150, sharex=True,
    gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.12))

# rank auxes by final q so the legend reads good -> bad
order = sorted(aux_names, key=lambda a: Q[a][-1], reverse=True)
axq.plot(u, qm, color=C_MAIN, lw=2.6, zorder=6, solid_capstyle="round")
axq.annotate("oracle", xy=(u[-1], qm[-1]), xytext=(6, 0), textcoords="offset points",
             color=C_MAIN, fontsize=9, va="center", fontweight="bold")
for a in order:
    c = AUX_COLORS.get(a, "#888888")
    axq.plot(u, Q[a], color=c, lw=2.0, zorder=4, solid_capstyle="round")
    axq.annotate(f"{a}", xy=(u[-1], Q[a][-1]), xytext=(6, 0),
                 textcoords="offset points", color=c, fontsize=8.6, va="center",
                 fontweight="bold")

ymax = max(0.4, qm.max(), max(v.max() for v in Q.values())) * 1.15
axq.set_ylim(-0.03, ymax)
axq.set_ylabel("Q value  (EMA reward)", color=INK2, fontsize=10)
axq.set_title(f"2048: which structural auxes did the bandit learn to trust?  ({RUN.split('/')[-1]})",
              color=INK, fontsize=11.5, pad=10, loc="left")
axq.yaxis.set_major_locator(MultipleLocator(0.2))

# legend ranks final q + shows pull counts (good auxes get pulled more)
handles = [plt.Line2D([0], [0], color=C_MAIN, lw=2.6,
                      label=f"oracle  (q={qm[-1]:+.2f})")]
handles += [plt.Line2D([0], [0], color=AUX_COLORS.get(a, "#888"), lw=2.0,
                       label=f"{a}  (q={Q[a][-1]:+.2f}, {finalpull[a]} pulls)") for a in order]
axq.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.2, ncol=2)

# beta panel
axb.axhspan(0.40, 0.90, color=BAND, zorder=0)
for yv, txt, ls in [(0.90, "cap 0.90", ":"), (0.70, "init 0.70", "--"), (0.40, "floor 0.40", ":")]:
    axb.axhline(yv, color=INK2, lw=0.9, ls=ls, alpha=0.55, zorder=1)
    axb.annotate(txt, xy=(u[-1], yv), xytext=(5, 0), textcoords="offset points",
                 color=INK2, fontsize=7.6, va="center")
axb.plot(u, beta, color=INK, lw=2.2, zorder=4, solid_capstyle="round")
axb.set_ylim(0.34, 0.98)
axb.set_ylabel("β = P(sample oracle arm)", color=INK2, fontsize=10)
axb.set_xlabel("Bandit update", color=INK2, fontsize=10)
axb.yaxis.set_major_locator(MultipleLocator(0.2))

for ax in (axq, axb):
    ax.set_xlim(1, len(recs) + max(12, len(recs) * 0.12))
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)

fig.text(0.012, 0.012, "each point = one logged bandit update. Qwen2.5-32B, softmax bandit "
         "+ island-relative reward. Higher final q = aux the bandit found more useful.",
         color=INK2, fontsize=7)
fig.tight_layout(rect=(0, 0.025, 1, 1))
out = "alg2_2048_aux_weights"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", bbox_inches="tight", dpi=170)
print(f"wrote {out}.pdf/.png")
print("final q ranking (good -> bad):")
for a in order:
    print(f"  {a:10s} q={Q[a][-1]:+.3f}  pulls={finalpull[a]}")
print(f"  {'oracle':10s} q={qm[-1]:+.3f}")
# === END ALG2 MOD (aux-selection) ===
