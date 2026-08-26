#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): 2048 best-so-far vs generation, treatment vs control. ===
"""Best valid oracle score reached by each generation, per arm. Individual replicates
(thin) + arm mean (bold) -- honest at n=3."""
import glob
import sqlite3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

C_TREAT, C_CTRL = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
NGEN = 200


def bestsofar_curve(run_dir, ngen=NGEN):
    c = sqlite3.connect(f"{run_dir}/programs.sqlite")
    c.row_factory = sqlite3.Row
    best, by_gen = -1e9, {}
    for r in c.execute("select generation, combined_score, correct from programs "
                       "order by generation, rowid"):
        if r["correct"] and r["combined_score"] is not None and r["combined_score"] > best:
            best = r["combined_score"]
        by_gen[r["generation"]] = best
    curve, last = np.zeros(ngen + 1), 0.0
    for g in range(ngen + 1):
        if g in by_gen:
            last = max(by_gen[g], 0.0)
        curve[g] = last
    return curve


def arm(pattern):
    curves = np.array([bestsofar_curve(d) for d in sorted(glob.glob(pattern))])
    return curves.mean(0), curves


fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
x = np.arange(NGEN + 1)
for pat, color in [("results/treatment_rep*", C_TREAT), ("results/control_rep*", C_CTRL)]:
    mean, curves = arm(pat)
    for cve in curves:
        ax.plot(x, cve, color=color, lw=1.0, alpha=0.40, solid_capstyle="round")
    ax.plot(x, mean, color=color, lw=2.4, zorder=5, solid_capstyle="round")

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], color=C_TREAT, lw=2.4, label="Treatment (5 aux-guided selection)"),
    Line2D([0], [0], color=C_CTRL, lw=2.4, label="Control (oracle-only selection)"),
], loc="lower right", frameon=False, fontsize=8.5)
tm = arm("results/treatment_rep*")[0]; cm = arm("results/control_rep*")[0]
ax.annotate("Treatment", xy=(NGEN, tm[-1]), xytext=(5, 4), textcoords="offset points",
            color=C_TREAT, fontsize=9, va="center", fontweight="bold")
ax.annotate("Control", xy=(NGEN, cm[-1]), xytext=(5, -4), textcoords="offset points",
            color=C_CTRL, fontsize=9, va="center", fontweight="bold")
# tile-value guides (score = max_tile/512 - moves*0.002)
for sc, lab in [(1.0, "512"), (2.0, "1024"), (4.0, "2048")]:
    if sc <= 4.2:
        ax.axhline(sc, color=GRID, lw=1.0, zorder=0)
        ax.annotate(f"~tile {lab}", xy=(2, sc), xytext=(2, sc + 0.03), color=INK2, fontsize=7)

ax.set_xlim(0, NGEN + 22)
ax.set_ylim(-0.05, max(tm.max(), cm.max()) + 0.35)
ax.set_xlabel("Generation", color=INK2, fontsize=10)
ax.set_ylabel("Best valid oracle score so far", color=INK2, fontsize=10)
ax.set_title("2048: aux-guided selection shows no advantage over oracle-only",
             color=INK, fontsize=11.5, pad=10, loc="left")
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.grid(True, axis="y", color=GRID, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
fig.text(0.012, 0.015, "thin = individual replicate (n=3 per arm); bold = mean. "
         "Qwen2.5-32B, softmax bandit + island-relative reward.", color=INK2, fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("alg2_2048_bestsofar.pdf", bbox_inches="tight")
fig.savefig("alg2_2048_bestsofar.png", bbox_inches="tight", dpi=170)
print("wrote alg2_2048_bestsofar.pdf/.png")
for t in (1.0, 2.0):
    gt = int(np.argmax(tm >= t)) if (tm >= t).any() else None
    gc = int(np.argmax(cm >= t)) if (cm >= t).any() else None
    print(f"  mean crosses {t}: treatment gen {gt}, control gen {gc}")
# === END ALG2 MOD (aux-selection) ===
