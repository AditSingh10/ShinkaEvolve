#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): best-so-far vs generation, treatment vs control. ===
"""One figure: best valid oracle score reached by each generation, per arm. Plots the
3 individual replicates (thin) plus the arm mean (bold) -- honest at n=3, since the
control mean is dragged by one stalled run. Shows the headline finding: treatment
reaches the good region earlier, both converge to the same ceiling."""
import glob
import sqlite3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- reference palette (categorical slots 1 & 2 -- validated most-distinct pair) ---
C_TREAT = "#2a78d6"   # blue
C_CTRL = "#eb6834"    # orange
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
NGEN = 200


def bestsofar_curve(run_dir, ngen=NGEN):
    c = sqlite3.connect(f"{run_dir}/programs.sqlite")
    c.row_factory = sqlite3.Row
    best = 0.0
    by_gen = {}
    for r in c.execute(
        "select generation, combined_score, correct from programs order by generation, rowid"
    ):
        if r["correct"] and r["combined_score"] and r["combined_score"] > best:
            best = r["combined_score"]
        by_gen[r["generation"]] = best
    # forward-fill onto a dense 0..ngen axis
    curve, last = np.zeros(ngen + 1), 0.0
    for g in range(ngen + 1):
        if g in by_gen:
            last = by_gen[g]
        curve[g] = last
    return curve


def arm_band(pattern):
    curves = np.array([bestsofar_curve(d) for d in sorted(glob.glob(pattern))])
    return curves.mean(0), curves.min(0), curves.max(0), curves


fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=150)
x = np.arange(NGEN + 1)

# Plot INDIVIDUAL replicates (thin) + the arm mean (bold). At n=3 this shows the real
# data honestly -- all three treatment runs rise early and stay tight, while the control
# runs spread out with one stalling -- instead of a mean+band that the outlier distorts.
for pattern, color in [
    ("results/pilot_none_treatment_rep*", C_TREAT),
    ("results/pilot_none_control_rep*", C_CTRL),
]:
    mean, _, _, curves = arm_band(pattern)
    for cve in curves:
        ax.plot(x, cve, color=color, lw=1.0, alpha=0.40, solid_capstyle="round")
    ax.plot(x, mean, color=color, lw=2.4, solid_capstyle="round", zorder=5)

# proxy handles for a clean 2-entry legend (identity = color)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], color=C_TREAT, lw=2.4, label="Treatment (aux-guided selection)"),
    Line2D([0], [0], color=C_CTRL, lw=2.4, label="Control (oracle-only selection)"),
], loc="lower right", frameon=False, fontsize=8.5)

# direct labels at the right end (mean endpoints)
tm = arm_band("results/pilot_none_treatment_rep*")[0]
cm = arm_band("results/pilot_none_control_rep*")[0]
ax.annotate("Treatment", xy=(NGEN, tm[-1]), xytext=(5, 3), textcoords="offset points",
            color=C_TREAT, fontsize=9, va="center", fontweight="bold")
ax.annotate("Control", xy=(NGEN, cm[-1]), xytext=(5, -2), textcoords="offset points",
            color=C_CTRL, fontsize=9, va="center", fontweight="bold")

# one honest annotation: the early-rise gap (bold mean lines cross 2.3 here)
t23 = int(np.argmax(tm >= 2.3)); c23 = int(np.argmax(cm >= 2.3))
ax.axhline(2.3, color=GRID, lw=1.0, zorder=0)
ax.annotate(f"mean reaches 2.3\nat gen {t23}", xy=(t23, 2.3), xytext=(t23 + 6, 1.45),
            color=C_TREAT, fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", color=C_TREAT, lw=1.1))
ax.annotate(f"gen {c23}", xy=(c23, 2.3), xytext=(c23 + 4, 1.75),
            color=C_CTRL, fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", color=C_CTRL, lw=1.1))

ax.set_xlim(0, NGEN + 24)
ax.set_ylim(0.9, 2.62)
ax.set_xlabel("Generation", color=INK2, fontsize=10)
ax.set_ylabel("Best valid oracle score so far", color=INK2, fontsize=10)
ax.set_title("Aux-guided selection reaches good solutions earlier; both converge",
             color=INK, fontsize=11.5, pad=10, loc="left")
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(0.4))
ax.grid(True, color=GRID, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
fig.text(0.012, 0.015, "thin = individual replicate (n=3 per arm); bold = mean. Qwen2.5-32B, "
         "softmax bandit + island-relative reward (v4).", color=INK2, fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("alg2_bestsofar.pdf", bbox_inches="tight")
fig.savefig("alg2_bestsofar.png", bbox_inches="tight", dpi=170)
print("wrote alg2_bestsofar.pdf and .png")

# quick numeric check printed alongside
tm = arm_band("results/pilot_none_treatment_rep*")[0]
cm = arm_band("results/pilot_none_control_rep*")[0]
for t in (2.0, 2.3, 2.4):
    gt = int(np.argmax(tm >= t)) if (tm >= t).any() else None
    gc = int(np.argmax(cm >= t)) if (cm >= t).any() else None
    print(f"  mean curve crosses {t}: treatment gen {gt}, control gen {gc}")
# === END ALG2 MOD (aux-selection) ===
