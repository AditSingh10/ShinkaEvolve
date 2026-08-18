#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): parent-distribution divergence, aux vs oracle. ===
"""From a treatment run's aux_divergence.jsonl: Jensen-Shannon divergence between each
aux's parent-selection distribution and the oracle's, over the archive, across the search.
High = the aux steers selection somewhere genuinely different from the oracle (real
intervention); ~0 = it re-picks the oracle's own parents (a no-op).
Usage: python plot_aux_divergence.py [results/treatment_rep1]"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/treatment_rep1"
AUX_COLORS = {
    "empty": "#2a78d6", "monotonic": "#eb6834", "smooth": "#1baf7a",
    "corner": "#eda100", "merge": "#e87ba4",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

recs = [json.loads(l) for l in open(f"{RUN}/aux_divergence.jsonl")]
gens = np.array([r["generation"] for r in recs], dtype=float)
aux_names = [k[3:] for k in recs[-1] if k.startswith("js_")]
JS = {a: np.array([r.get(f"js_{a}", np.nan) for r in recs], dtype=float) for a in aux_names}

fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)

# order legend by mean divergence (most-interventional aux first)
order = sorted(aux_names, key=lambda a: np.nanmean(JS[a]), reverse=True)
for a in order:
    c = AUX_COLORS.get(a, "#888888")
    y = JS[a]
    ax.plot(gens, y, color=c, lw=1.3, alpha=0.35, solid_capstyle="round")
    # rolling mean for readability
    k = max(1, len(y) // 25)
    ker = np.ones(k) / k
    ys = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), ker, mode="same")
    ax.plot(gens, ys, color=c, lw=2.4, zorder=5, solid_capstyle="round")
    ax.annotate(a, xy=(gens[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                color=c, fontsize=8.8, va="center", fontweight="bold")

handles = [plt.Line2D([0], [0], color=AUX_COLORS.get(a, "#888"), lw=2.4,
                      label=f"{a}  (mean JS={np.nanmean(JS[a]):.3f})") for a in order]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.4)

ax.set_ylim(0, min(1.0, np.nanmax([np.nanmax(v) for v in JS.values()]) * 1.15 + 0.02))
ax.set_xlim(gens.min(), gens.max() + (gens.max() - gens.min()) * 0.10)
ax.set_xlabel("Generation", color=INK2, fontsize=10)
ax.set_ylabel("JS divergence  (aux vs oracle parent dist, bits)", color=INK2, fontsize=10)
ax.set_title("2048: how differently does each aux steer parent selection vs the oracle?",
             color=INK, fontsize=11.5, pad=10, loc="left")
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(True, color=GRID, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
fig.text(0.012, 0.015, "thin = per-update; bold = rolling mean. 0 = re-picks the oracle's "
         "parents (no-op); higher = genuinely different selection (residualized aux).",
         color=INK2, fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = "alg2_2048_aux_divergence"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", bbox_inches="tight", dpi=170)
print(f"wrote {out}.pdf/.png")
print("mean JS divergence (most interventional -> least):")
for a in order:
    print(f"  {a:10s} mean JS={np.nanmean(JS[a]):.3f}  final={JS[a][-1]:.3f}")
# === END ALG2 MOD (aux-selection) ===
