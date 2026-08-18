#!/usr/bin/env python3
# === ALG2 MOD (aux-selection): per-update bandit trace for treatment rep2. ===
"""Deep-dive figure for the one replicate where an aux persistently beat the oracle.
Two stacked panels sharing the bandit-update clock (128 updates):
  (top) Q values -- main (oracle), connect, hole, caging;
  (bottom) beta = P(sample from oracle arm), with floor/init/cap guides.
Every point is a real logged update (aux_bandit_history.jsonl), not a snapshot.
The story is non-stationary: early aux dip -> oracle recovery -> durable connect
takeover once the oracle saturates."""
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- reference palette: categorical trio (blue/orange/aqua = all-pairs validated) ---
C_MAIN = "#2a78d6"    # oracle arm (slot 1)
C_CONN = "#eb6834"    # connect -- the winner (slot 2)
C_HOLE = "#1baf7a"    # hole (slot 3)
C_CAGE = "#9a9892"    # caging -- flat at 0, muted gray (not a categorical hue)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BAND = "#f0eff0"

recs = [json.loads(l) for l in
        open("results/pilot_none_treatment_rep2/aux_bandit_history.jsonl")]
u = np.arange(1, len(recs) + 1)                      # update index (bandit clock)
qm = np.array([r["q_main"] for r in recs])
qc = np.array([r["q_connect"] for r in recs])
qh = np.array([r["q_hole"] for r in recs])
qg = np.array([r["q_caging"] for r in recs])
beta = np.array([r["beta"] for r in recs])

# key events (verified against the log)
u_peak = int(np.argmax(qm)) + 1                      # Q_main peak (oracle recovery top)
q_peak = qm.max()
u_floor = next(i + 1 for i, r in enumerate(recs) if r["beta"] <= 0.4001)  # first floor hit

fig, (axq, axb) = plt.subplots(
    2, 1, figsize=(7.6, 6.2), dpi=150, sharex=True,
    gridspec_kw=dict(height_ratios=[1.55, 1.0], hspace=0.12))

# ---------------------------------------------------------------- top: Q values
for y, c, name, lw in [(qg, C_CAGE, "caging", 1.4), (qh, C_HOLE, "hole", 1.8),
                       (qc, C_CONN, "connect", 2.6), (qm, C_MAIN, "main (oracle)", 2.6)]:
    axq.plot(u, y, color=c, lw=lw, solid_capstyle="round",
             zorder=4 if name.startswith(("connect", "main")) else 3)

# oracle-recovery peak marker
axq.plot([u_peak], [q_peak], "o", color=C_MAIN, ms=6, zorder=6)
axq.annotate(f"oracle peaks\nQ={q_peak:.2f}\n(upd {u_peak}, gen 85)",
             xy=(u_peak, q_peak), xytext=(u_peak - 33, q_peak - 0.16),
             color=C_MAIN, fontsize=8.3, ha="center", va="top",
             arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=1.1))
# durable takeover: after the collapse, connect stays on top
axq.annotate("connect stays\nabove oracle",
             xy=(116, qc[115]), xytext=(122, 0.60),
             color=C_CONN, fontsize=8.3, ha="left",
             arrowprops=dict(arrowstyle="->", color=C_CONN, lw=1.1))
# direct labels at right edge
for y, c, name in [(qc, C_CONN, "connect"), (qm, C_MAIN, "main"),
                   (qh, C_HOLE, "hole"), (qg, C_CAGE, "caging")]:
    axq.annotate(name, xy=(u[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                 color=c, fontsize=8.6, va="center", fontweight="bold")

axq.set_ylim(-0.03, 0.80)
axq.set_ylabel("Q value  (EMA reward)", color=INK2, fontsize=10)
axq.set_title("Treatment rep 2: the aux 'connect' persistently overtakes the oracle",
              color=INK, fontsize=12, pad=10, loc="left")
axq.yaxis.set_major_locator(MultipleLocator(0.2))

# ------------------------------------------------------------- bottom: beta
axb.axhspan(0.40, 0.90, color=BAND, zorder=0)                 # allowed band
for yv, txt, ls in [(0.90, "cap 0.90", ":"), (0.70, "init 0.70", "--"),
                    (0.40, "floor 0.40", ":")]:
    axb.axhline(yv, color=INK2, lw=0.9, ls=ls, alpha=0.55, zorder=1)
    axb.annotate(txt, xy=(u[-1], yv), xytext=(5, 0), textcoords="offset points",
                 color=INK2, fontsize=7.6, va="center")
axb.plot(u, beta, color=INK, lw=2.2, solid_capstyle="round", zorder=4)
axb.plot([u_floor], [0.40], "o", color=INK, ms=5, zorder=5)
axb.annotate(f"β hits floor\n(upd {u_floor})", xy=(u_floor, 0.40),
             xytext=(u_floor + 4, 0.52), color=INK, fontsize=8.0, ha="left",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
axb.set_ylim(0.34, 0.98)
axb.set_ylabel("β = P(sample oracle arm)", color=INK2, fontsize=10)
axb.set_xlabel("Bandit update", color=INK2, fontsize=10)
axb.yaxis.set_major_locator(MultipleLocator(0.2))

# phase shading spanning both panels (early dip / oracle recovery / aux takeover)
for ax in (axq, axb):
    ax.axvspan(1, 22, color="#f6f5f2", zorder=-1)
    ax.axvspan(72, len(recs), color="#f6f5f2", zorder=-1)
axq.text(11, 0.755, "① early aux dip", color=INK2, fontsize=7.8, ha="center")
axq.text(47, 0.755, "② oracle recovery", color=INK2, fontsize=7.8, ha="center")
axq.text(100, 0.755, "③ aux takeover", color=INK2, fontsize=7.8, ha="center")

for ax in (axq, axb):
    ax.set_xlim(1, len(recs) + 26)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
axb.xaxis.set_major_locator(MultipleLocator(20))

fig.text(0.012, 0.012, "each point = one logged bandit update (n=128; ~gens 3–196). "
         "Qwen2.5-32B, softmax bandit + island-relative reward (v4).",
         color=INK2, fontsize=7)
fig.tight_layout(rect=(0, 0.025, 1, 1))
fig.savefig("alg2_rep2_bandit.pdf", bbox_inches="tight")
fig.savefig("alg2_rep2_bandit.png", bbox_inches="tight", dpi=170)
print("wrote alg2_rep2_bandit.pdf and .png")
print(f"  Q_main peak {q_peak:.3f} @ upd{u_peak}; beta floor first @ upd{u_floor}; "
      f"final Qconn={qc[-1]:.3f} > Qmain={qm[-1]:.3f}, beta={beta[-1]:.3f}")
# === END ALG2 MOD (aux-selection) ===
