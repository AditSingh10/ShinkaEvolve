# === ALG2 MOD (aux-selection): aux-shaped parent selection + two-level bandit. ===
# NEW FILE -- pure addition. Nothing here runs unless a config opts in, so existing
# behaviour is untouched.
#
# Two pieces:
#   1. build_aux_probabilities(): the same weighted-sampling math the oracle sampler uses
#      (sigmoid of median/MAD-standardized score x diversity term), but driven by an aux
#      score and oriented by that aux's direction prior.
#   2. TwoLevelAuxBandit: chooses WHICH distribution to sample the parent from.
#        top level  -- beta = P(main/oracle) vs the aux pool  (bounded, non-stationary)
#        bottom     -- which aux, via an epsilon-floored SOFTMAX over the per-aux Q values
#      Rewards are frontier-relative ratios measured on the ORACLE; oracle never modified.

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np


MAIN_ARM = "main"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def build_aux_probabilities(
    aux_values: Sequence[float],
    direction: int,
    children_counts: Sequence[int],
    lam: float = 10.0,
    oracle_values: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Selection distribution over programs, driven by one aux.

    Mirrors the oracle sampler: performance term = sigmoid(lam * direction *
    (a_i - median)/MAD), diversity term = 1/(1+children). Median/MAD make it
    scale-invariant and outlier-robust; `direction` (+1 maximize / -1 minimize) is the
    aux's supplied prior and is the ONLY thing that says which half to favour.

    RESIDUALIZATION (when `oracle_values` is given): we first remove the component of
    the aux that is linearly predictable from the oracle and select on the residual.
    Measured on a real archive, the raw auxes induce distributions almost identical to
    the oracle's (total-variation ~0.09-0.14, same top-1 parent) because they were chosen
    for correlating with good packings -- so an "aux arm" would re-pick the oracle's own
    parents and the intervention would be a no-op. The residual is the part of the
    structural signal the oracle does NOT already capture, which is precisely the new
    information, and it roughly doubles the divergence.

    If the aux is constant across the pool, every standardized score is 0 -> sigmoid
    0.5 for all -> this degrades gracefully to diversity-only sampling (a dead arm,
    never a crash).
    """
    a = np.asarray(aux_values, dtype=float)
    n_child = np.asarray(children_counts, dtype=float)
    if a.size == 0:
        return np.array([])

    if oracle_values is not None:
        o = np.asarray(oracle_values, dtype=float)
        # need enough points and a non-degenerate oracle spread to regress
        if o.size == a.size and a.size >= 3 and float(np.std(o)) > 1e-12:
            A = np.vstack([o, np.ones_like(o)]).T
            try:
                coef, *_ = np.linalg.lstsq(A, a, rcond=None)
                resid = a - A @ coef
                if np.isfinite(resid).all():
                    a = resid
            except Exception:
                pass  # fall back to the raw aux; never break selection

    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    scale = max(mad, 1e-6)

    perf = _sigmoid(lam * float(direction) * (a - med) / scale)
    div = 1.0 / (1.0 + n_child)
    w = perf * div

    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(a.size, 1.0 / a.size)
    return w / total


class TwoLevelAuxBandit:
    """Two-level, non-stationary bandit over selection distributions.

    We *know* the oracle distribution is good and do *not* yet know if any aux is, so the
    levels encode an asymmetric prior:
      * top: Bernoulli(beta) picks `main` (oracle) vs the aux pool. beta starts high, is
        floored (oracle stays the anchor) and capped (aux pool is never starved).
      * bottom: a SOFTMAX over the per-aux Q values (with an epsilon floor so no aux is
        starved). This replaced a reused AsymmetricUCB: the UCB kept its own second copy
        of each aux's value (separate from q_aux), which was redundant and hard to reason
        about. Now a single set of recency-weighted q_aux drives BOTH the within-pool
        choice (softmax) and beta (advantage). tau is the softmax temperature; it must
        match the scale of q_aux (~[0, 0.6]).

    Reward = a frontier-relative ratio, computed by the runner (0 if the child is
    invalid). Q values are exponential recency-weighted averages (constant step
    eta = 1 - decay), i.e. non-stationary: they track the phase of the search.
    beta moves on the ADVANTAGE max_k(shrunk q_aux) - q_main: auxes gain weight only if
    the best of them beats what the oracle arm was already achieving. That baseline is
    essential -- best-so-far climbs regardless of arm, and both arms share the same
    stochastic LLM-mutation noise, which the difference cancels.
    """

    def __init__(
        self,
        aux_names: List[str],
        # Calibrated for a ~200-generation budget (see test_aux_bandit.py):
        # at beta=0.70 the pool gets ~30% of pulls -> ~20 per aux over 200 gens, and
        # decay=0.80 (eta=0.2, time constant ~5) converges within that many samples.
        beta_init: float = 0.70,
        beta_floor: float = 0.40,
        beta_cap: float = 0.90,
        beta_step: float = 0.05,
        kappa: float = 5.0,
        decay: float = 0.80,
        shrink_n0: float = 5.0,
        # Frontier-relative reward is a ratio in ~[0, 1.2]; this clip is a non-binding
        # guard against a cold-start division blowing up (it only bites at cold start).
        reward_clip: float = 2.0,
        # Within-pool softmax over q_aux. tau ~ scale of q_aux; eps is the exploration
        # floor (every aux keeps >= eps/#aux probability, so none is ever starved).
        tau: float = 0.15,
        epsilon: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.aux_names = list(aux_names)
        self.beta = float(beta_init)
        self.beta_floor = float(beta_floor)
        self.beta_cap = float(beta_cap)
        self.beta_step = float(beta_step)
        self.kappa = float(kappa)
        self.shrink_n0 = float(shrink_n0)
        self.reward_clip = float(reward_clip)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.eta = 1.0 - float(decay)  # EMA step
        self.rng = np.random.default_rng(seed)

        # Single value system: one recency-weighted q per arm, used for BOTH the
        # within-pool softmax and beta. No second bookkeeping.
        self.q_main = 0.0
        self.q_aux: Dict[str, float] = {k: 0.0 for k in self.aux_names}
        self.pulls: Dict[str, int] = {MAIN_ARM: 0, **{k: 0 for k in self.aux_names}}
        self.history: List[dict] = []

    # ---------------------------------------------------------------- selection
    def _softmax_probs(self) -> np.ndarray:
        """Epsilon-floored softmax over the current per-aux Q values."""
        q = np.array([self.q_aux[k] for k in self.aux_names], dtype=float)
        e = np.exp((q - q.max()) / max(self.tau, 1e-9))  # -max: numerical stability
        p = e / e.sum()
        p = (1.0 - self.epsilon) * p + self.epsilon / len(q)  # floor: never starve an aux
        return p / p.sum()

    def select_arm(self) -> str:
        """Return 'main' or one of the aux names."""
        if self.rng.random() < self.beta:
            return MAIN_ARM
        try:
            probs = self._softmax_probs()
            if probs.size != len(self.aux_names) or not np.isfinite(probs).all():
                raise ValueError
        except Exception:
            probs = np.full(len(self.aux_names), 1.0 / len(self.aux_names))
        return str(self.rng.choice(self.aux_names, p=probs))

    # ------------------------------------------------------------------ update
    def update(self, arm: str, improvement: float) -> None:
        """Feed back one oracle-measured improvement for the arm that was used."""
        r = float(improvement)
        if not np.isfinite(r):
            r = 0.0
        r = float(np.clip(r, -self.reward_clip, self.reward_clip))
        self.pulls[arm] = self.pulls.get(arm, 0) + 1

        if arm == MAIN_ARM:
            self.q_main += self.eta * (r - self.q_main)
        else:
            self.q_aux[arm] += self.eta * (r - self.q_aux[arm])

        # beta follows the advantage of the BEST aux over the oracle arm.
        # Using max_k (not the pooled mean) is deliberate: pooling dilutes one good aux
        # with two bad ones, so the pool looks mediocre, beta rises, the pool gets fewer
        # pulls, its estimates go stale -- a positive-feedback death spiral.
        # Each q is shrunk toward 0 by n/(n+n0) so an under-sampled arm cannot pull beta
        # down on noise alone (counteracts the optimism bias of taking a max).
        advantage = self._best_aux_value() - self.q_main
        self.beta = float(
            np.clip(
                self.beta - self.beta_step * np.tanh(self.kappa * advantage),
                self.beta_floor,
                self.beta_cap,
            )
        )

    def _best_aux_value(self) -> float:
        """Highest confidence-shrunk aux value (0 if no aux has been pulled yet)."""
        best = 0.0
        for k, q in self.q_aux.items():
            n = float(self.pulls.get(k, 0))
            best = max(best, q * (n / (n + self.shrink_n0)))
        return best

    # ------------------------------------------------- dynamic arms (bootstrap)
    # For the continuous-bootstrap experiment: the active arm set is a subset of a fixed
    # aux pool and is swapped over the run (drop the worst by q, add fresh pool members).
    # Kept arms retain their learned q (a champion defends); new arms start cold at q=0.
    def active_arms(self) -> List[str]:
        return list(self.aux_names)

    def add_arm(self, name: str) -> None:
        """Activate an aux arm (fresh q=0) if not already present."""
        self.q_aux.setdefault(name, 0.0)
        self.pulls.setdefault(name, 0)
        if name not in self.aux_names:
            self.aux_names.append(name)

    def remove_arm(self, name: str) -> None:
        """Deactivate an aux arm (drops its learned q and pulls)."""
        self.aux_names = [a for a in self.aux_names if a != name]
        self.q_aux.pop(name, None)
        self.pulls.pop(name, None)

    def rank_arms(self) -> List[str]:
        """Active arms sorted best->worst by CONFIDENCE-SHRUNK q (the swap fitness), so a
        thin lucky arm can't survive a swap on noise alone."""
        def key(k: str) -> float:
            n = float(self.pulls.get(k, 0))
            return self.q_aux[k] * (n / (n + self.shrink_n0))
        return sorted(self.aux_names, key=key, reverse=True)

    # ----------------------------------------------------------------- logging
    def snapshot(self, generation: Optional[int] = None, arm: Optional[str] = None,
                 improvement: Optional[float] = None) -> dict:
        snap = {
            "generation": generation,
            "arm": arm,
            "improvement": improvement,
            "beta": self.beta,
            "q_main": self.q_main,
            # `advantage` is the quantity that drives beta: the best aux value after
            # confidence shrinkage, minus the oracle arm.
            "advantage": self._best_aux_value() - self.q_main,
            **{f"q_{k}": v for k, v in self.q_aux.items()},
            **{f"pulls_{k}": v for k, v in self.pulls.items()},
        }
        self.history.append(snap)
        return snap

    def write_history(self, path: str) -> None:
        try:
            with open(path, "w") as f:
                for row in self.history:
                    f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def __repr__(self) -> str:
        q = ", ".join(f"{k}={v:+.4f}" for k, v in self.q_aux.items())
        return (f"TwoLevelAuxBandit(beta={self.beta:.3f}, q_main={self.q_main:+.4f}, "
                f"{q}, pulls={self.pulls})")


# --------------------------------------------------------------------------- #
#  Parent-distribution DIVERGENCE diagnostics (aux vs oracle).                 #
#                                                                             #
#  An aux is only doing real work if, over the SAME candidate pool, it induces #
#  a parent-selection distribution that differs from the oracle's. If the two  #
#  distributions coincide, the aux arm just re-picks the oracle's own parents  #
#  -- a no-op (this is exactly why we residualize). These pure functions make  #
#  that "how different is it, really?" measurable throughout the search.       #
# --------------------------------------------------------------------------- #
def oracle_probabilities(
    oracle_values: Sequence[float],
    children_counts: Sequence[int],
    lam: float = 10.0,
) -> np.ndarray:
    """Replicate WeightedSamplingStrategy's parent distribution over one pool.

    weight_i = sigmoid(lam*(a_i - median)/MAD) * 1/(1+children_i), normalized. This is
    the SAME formula the oracle sampler uses (parents.py), so a divergence against it is
    a faithful "would the aux pick different parents than the oracle?" measurement.
    """
    a = np.asarray(oracle_values, dtype=float)
    n = np.asarray(children_counts, dtype=float)
    if a.size == 0:
        return np.array([])
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    scale = max(mad, 1e-6)
    w = _sigmoid(lam * (a - med) / scale) * (1.0 / (1.0 + n))
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(a.size, 1.0 / a.size)
    return w / total


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in BITS (log2): symmetric, bounded in [0, 1].

    0 => identical parent distributions (aux is a no-op vs the oracle); 1 => disjoint
    support (aux and oracle never favour the same parent). Bounded and symmetric, so it
    reads cleanly as a single 'how divergent' number per aux over the run.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or p.size != q.size:
        return float("nan")
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    """Total-variation distance = 0.5 * sum|p-q|, in [0, 1] (reported alongside JS)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or p.size != q.size:
        return float("nan")
    return float(0.5 * np.abs(p - q).sum())


def aux_oracle_divergences(
    aux_names: Sequence[str],
    aux_directions: Dict[str, int],
    oracle_values: Sequence[float],
    children_counts: Sequence[int],
    aux_matrix: Dict[str, Sequence[float]],
    lam: float = 10.0,
    residualize: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Per-aux JS / TV divergence of its parent distribution vs the oracle's.

    `aux_matrix[name]` is that aux's value for every program in the pool (same order as
    `oracle_values`). Uses the exact selection distributions (residualized aux vs oracle),
    so this is what the samplers would actually do on this pool. Returns
    {aux: {"js": ..., "tv": ...}}; empty if the pool is too small to form a distribution.
    """
    n = len(oracle_values)
    out: Dict[str, Dict[str, float]] = {}
    if n < 2:
        return out
    p_oracle = oracle_probabilities(oracle_values, children_counts, lam=lam)
    for name in aux_names:
        vals = aux_matrix.get(name)
        if vals is None or len(vals) != n:
            continue
        p_aux = build_aux_probabilities(
            vals,
            int(aux_directions.get(name, 1)),
            children_counts,
            lam=lam,
            oracle_values=oracle_values if residualize else None,
        )
        if p_aux.size != n or not np.isfinite(p_aux).all():
            continue
        out[name] = {
            "js": js_divergence(p_aux, p_oracle),
            "tv": total_variation(p_aux, p_oracle),
        }
    return out
# === END ALG2 MOD (aux-selection) ===
