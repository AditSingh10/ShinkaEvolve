"""Event-triggered search steering for AuxEvolve (spec: Event-Triggered Search Steering).

Pure, dependency-light controllers (numpy only). They OBSERVE the search and decide a
steering *mode*; they never touch Q/beta/aux or the objective. Wiring into the async runner
is done separately so that with all `enable_*` flags off this module is a no-op and NORMAL
mode reproduces the 3-aux baseline exactly.

Two orthogonal partitions of the same programs:
  - real islands (island_idx, fixed)  -> unchanged evolutionary substrate (migration, beta/Q)
  - families (family_id, dynamic)      -> this module; drive steering only

Budget: b = every proposal (valid + invalid + novelty-rejected); t = b/B.
Family + validity statistics update only on EVALUATED / ADMITTED children.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

MODE_NORMAL = "NORMAL"
MODE_DIVERSITY = "DIVERSITY_STEERING"
MODE_VALIDITY = "VALIDITY_STEERING"
MODE_COMBINED = "COMBINED_STEERING"

# Problem-agnostic invalid failure classes tracked by the validity controller. A superset across
# problems: circle uses overlap/boundary/runtime/malformed; 2048 uses timeout/runtime; Go/Julia use
# compile/wrong_answer/runtime. Classes that never fire on a given problem simply stay at lambda 0.
CONSTRAINTS = ("runtime", "timeout", "compile", "wrong_answer",
               "malformed", "boundary", "overlap")

_PRETTY = {
    "overlap": "pairwise non-overlap",
    "boundary": "inside-unit-square",
    "runtime": "program executes without error",
    "malformed": "well-formed output (correct shape, finite values)",
    "timeout": "runs within the time limit",
    "compile": "compiles / parses without error",
    "wrong_answer": "produces correct answers",
}


def render_witness(k: str, w: Dict[str, Any]) -> str:
    """One prompt line describing a single failure witness, per class. Handles both
    structured witnesses ({i,j,amount}) and message-only ones ({error}/{note})."""
    if "error" in w and "i" not in w:            # message-only (parsed from checker text)
        return f"- {str(w['error'])[:160]}"
    if k == "overlap":
        return f"- circles {w['i']} and {w['j']} overlap by {w.get('amount', 0.0):.3f}"
    if k == "boundary":
        return f"- circle {w['i']} outside the container by {w.get('amount', 0.0):.3f}"
    if k == "runtime":
        return f"- {str(w.get('error', 'execution error'))[:160]}"
    return f"- {w.get('note', 'malformed output (wrong shape or non-finite values)')}"


# --------------------------------------------------------------------------------------
# Config (all defaults = spec). All enable_* False => baseline no-op.
# --------------------------------------------------------------------------------------
@dataclass
class SteeringConfig:
    # ablation flags (spec §23)
    enable_diversity_controller: bool = False
    enable_validity_controller: bool = False
    enable_family_based_islands: bool = False
    enable_mode_prompt_conditioning: bool = False
    enable_mode_specific_island_sampling: bool = False
    enable_mode_specific_inspiration_sampling: bool = False

    # family manager (§3)
    family_similarity_threshold: float = 0.85
    family_min_members_for_quality: int = 2  # new-family shrinkage guard (§24)

    # diversity (§4, §5)
    H_early: float = 0.90
    H_late: float = 0.40
    gamma_on: float = 0.15
    gamma_off: float = 0.05

    # validity (§6, §7)
    validity_ema_alpha: float = 0.15
    lambda_lr: float = 0.10
    target_violation_rate: float = 0.15
    lambda_on: float = 0.60
    lambda_off: float = 0.25
    violation_rate_on: float = 0.30
    violation_rate_off: float = 0.15
    num_failure_witnesses: int = 3
    witness_buffer_size: int = 12
    constraint_classes: Tuple[str, ...] = CONSTRAINTS  # all invalid failure families

    # controller cadence (§5)
    controller_warmup: int = 20
    controller_check_interval: int = 5
    trigger_confirmation_checks: int = 2
    release_confirmation_checks: int = 2

    # island(=family) sampling weights (§9)
    island_quality_weight_early: float = 0.30
    island_quality_weight_late: float = 0.80
    validity_island_quality_weight: float = 0.50
    combined_div_weight: float = 0.70
    combined_validity_weight: float = 0.30
    island_softmax_temp: float = 0.5
    island_prob_floor: float = 0.02

    # inspiration weights (§11)
    validity_insp_quality_weight: float = 0.5   # 0.5 F + 0.5 R
    combined_insp_quality_weight: float = 0.4   # 0.4 F + 0.3 d + 0.3 R
    combined_insp_distance_weight: float = 0.3
    combined_insp_feasibility_weight: float = 0.3


# --------------------------------------------------------------------------------------
# small numeric helpers (directly unit-tested — spec §26)
# --------------------------------------------------------------------------------------
def normalized_entropy(counts: List[float]) -> float:
    """H in [0,1]; K=1 -> 0 (spec §4)."""
    n = np.asarray([c for c in counts if c > 0], dtype=float)
    K = len(n)
    if K <= 1:
        return 0.0
    p = n / n.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(K))


def target_entropy(t: float, H_early: float, H_late: float) -> float:
    """H*(t) = H_late + (H_early - H_late)(1 - t)  (spec §4)."""
    return float(H_late + (H_early - H_late) * (1.0 - t))


def dominant_fraction(counts: List[float]) -> float:
    n = np.asarray(counts, dtype=float)
    s = n.sum()
    return float(n.max() / s) if s > 0 else 0.0


def classify_failure(correct: bool, violations: Optional[Dict[str, Any]],
                     error_msg: Optional[str] = None) -> Dict[str, list]:
    """Map an EVALUATED child's raw signals to {class: [witnesses]} for the validity controller.
    Valid children -> {} (nothing tripped). Covers every invalid type:
      runtime   (crashed / no output)      witness: the exception string
      malformed (wrong shape / non-finite)  witness: a note
      overlap / boundary (bad geometry)     witnesses: the per-pair / per-circle magnitudes
    """
    if correct:
        return {}
    f: Dict[str, list] = {}
    v = violations or {}
    if v.get("overlap"):
        f["overlap"] = list(v["overlap"])
    if v.get("boundary"):
        f["boundary"] = list(v["boundary"])
    if v.get("malformed"):
        f["malformed"] = [{"note": "malformed output (wrong shape or non-finite values)"}]
    if f:  # structured geometry was captured -> that's the failure, done
        return f
    # No structured violations. Route by the error message (problem-agnostic keyword match) so a
    # checker-rejected result is labeled by its true failure mode, not a generic crash.
    e = str(error_msg or "").strip()
    el = e.lower()
    if "overlap" in el:
        f["overlap"] = [{"error": e}]
    elif "outside" in el or "boundary" in el or "unit square" in el:
        f["boundary"] = [{"error": e}]
    elif "timed out" in el or "timeout" in el or "time limit" in el or "timedout" in el:
        f["timeout"] = [{"error": e}]
    elif any(m in el for m in (          # build/parse failures (Go/Julia/Python syntax)
            "undefined:", "cannot use", "cannot find package", "syntax error",
            "invalid syntax", "was never closed", "unexpected token", "loaderror",
            "undefvarerror", "expected ';'", "expected '}'", "expected declaration",
            "cannot import", "no such module", "compile")):
        f["compile"] = [{"error": e}]
    elif ("incorrect" in el or "mismatch" in el or "does not match" in el or "wrong answer" in el
          or "lower than" in el or ("expected" in el and "got" in el and "unpack" not in el)):
        f["wrong_answer"] = [{"error": e}]
    elif "non-finite" in el or "inhomogeneous" in el or "nan" in el or "shape" in el:
        f["malformed"] = [{"note": e[:140] or "malformed output"}]
    else:
        f["runtime"] = [{"error": e or "execution failed (no valid output produced)"}]
    return f


def _softmax_floor(scores: np.ndarray, temp: float, floor: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return s
    z = (s - s.max()) / max(temp, 1e-6)
    p = np.exp(z)
    p = p / p.sum()
    if floor > 0:  # mix a uniform floor so nothing is unreachable (§9)
        p = (1 - floor * len(p)) * p + floor
        p = np.clip(p, 0, None)
        p = p / p.sum()
    return p


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.ones_like(x) * 0.5
    return (x - lo) / (hi - lo)


# --------------------------------------------------------------------------------------
# FamilyManager (§3, §17C, §24) — pure over embeddings; caller does LLM summary + embed.
# --------------------------------------------------------------------------------------
class FamilyManager:
    def __init__(self, cfg: SteeringConfig):
        self.cfg = cfg
        self.families: List[Dict[str, Any]] = []  # {id, centroid, n, best_score, summary}
        self._member_family: Dict[str, int] = {}  # program_id -> family_id

    def assign(self, program_id: str, embedding: np.ndarray, score: float,
               summary: str = "") -> int:
        e = np.asarray(embedding, dtype=float)
        e = e / (np.linalg.norm(e) + 1e-12)
        if not self.families:
            return self._new_family(program_id, e, score, summary)
        sims = [float(np.dot(e, f["centroid"])) for f in self.families]
        best = int(np.argmax(sims))
        if sims[best] >= self.cfg.family_similarity_threshold:
            f = self.families[best]
            f["centroid"] = (f["centroid"] * f["n"] + e) / (f["n"] + 1)
            f["centroid"] /= (np.linalg.norm(f["centroid"]) + 1e-12)
            f["n"] += 1
            f["best_score"] = max(f["best_score"], score)
            f["members"].append(program_id)
            self._member_family[program_id] = f["id"]
            return f["id"]
        return self._new_family(program_id, e, score, summary)

    def _new_family(self, program_id, e, score, summary) -> int:
        fid = len(self.families)
        self.families.append(dict(id=fid, centroid=e, n=1, best_score=float(score),
                                  summary=summary, members=[program_id]))
        self._member_family[program_id] = fid
        return fid

    def members(self, family_id: int) -> List[str]:
        return list(self.families[family_id].get("members", [])) if 0 <= family_id < len(self.families) else []

    def summary(self, family_id: int) -> str:
        return self.families[family_id].get("summary", "") if 0 <= family_id < len(self.families) else ""

    def family_of(self, program_id: str) -> Optional[int]:
        return self._member_family.get(program_id)

    def counts(self) -> List[int]:
        return [f["n"] for f in self.families]

    def num_families(self) -> int:
        return len(self.families)

    def quality(self) -> np.ndarray:
        """Best score per family, with a shrinkage guard: families below the min-member
        threshold are pulled toward the global mean so one lucky child can't dominate (§24)."""
        if not self.families:
            return np.array([])
        best = np.array([f["best_score"] for f in self.families], dtype=float)
        gmean = float(best.mean())
        out = best.copy()
        for i, f in enumerate(self.families):
            if f["n"] < self.cfg.family_min_members_for_quality:
                out[i] = 0.5 * best[i] + 0.5 * gmean
        return out

    def distance(self, i: int, j: int) -> float:
        """1 - cosine between family-summary centroids."""
        return float(1.0 - np.dot(self.families[i]["centroid"], self.families[j]["centroid"]))

    def distances_from(self, i: int) -> np.ndarray:
        ci = self.families[i]["centroid"]
        return np.array([1.0 - float(np.dot(ci, f["centroid"])) for f in self.families])


# --------------------------------------------------------------------------------------
# DiversityController (§4, §5)
# --------------------------------------------------------------------------------------
class DiversityController:
    def __init__(self, cfg: SteeringConfig):
        self.cfg = cfg
        self.active = False
        self.H = 0.0
        self.H_target = 0.0
        self.gamma = 0.0
        self.dominant = 0.0
        self._on_streak = 0
        self._off_streak = 0

    def observe(self, family_counts: List[int], t: float) -> None:
        """Update cached stats every proposal (does NOT change active state)."""
        self.H = normalized_entropy(family_counts)
        self.H_target = target_entropy(t, self.cfg.H_early, self.cfg.H_late)
        self.gamma = float(np.clip(self.H_target - self.H, 0.0, 1.0))
        self.dominant = dominant_fraction(family_counts)

    def check(self) -> bool:
        """Apply hysteresis at a controller-check step; returns new active state."""
        if self.gamma >= self.cfg.gamma_on:
            self._on_streak += 1
        else:
            self._on_streak = 0
        if self.gamma <= self.cfg.gamma_off:
            self._off_streak += 1
        else:
            self._off_streak = 0
        if not self.active and self._on_streak >= self.cfg.trigger_confirmation_checks:
            self.active = True
        elif self.active and self._off_streak >= self.cfg.release_confirmation_checks:
            self.active = False
        return self.active


# --------------------------------------------------------------------------------------
# ValidityController (§6, §7)
# --------------------------------------------------------------------------------------
class ValidityController:
    """Tracks EVERY invalid failure class (runtime crash, malformed output, boundary, overlap),
    each with its own violation-rate EMA + dual-ascent lambda + witness buffer. The dominant
    unresolved failure (argmax lambda) drives the validity steering prompt block."""

    def __init__(self, cfg: SteeringConfig):
        self.cfg = cfg
        self.classes = tuple(cfg.constraint_classes)
        self.active = False
        self.r: Dict[str, float] = {k: 0.0 for k in self.classes}      # violation-rate EMA
        self.lam: Dict[str, float] = {k: 0.0 for k in self.classes}    # dual pressure
        self.witnesses: Dict[str, deque] = {
            k: deque(maxlen=cfg.witness_buffer_size) for k in self.classes}
        # per (family_id, class) feasibility EMA of children bred FROM that family
        self.family_feasibility: Dict[Tuple[int, str], float] = {}
        self._on_streak = 0
        self._off_streak = 0

    def update_child(self, failure: Dict[str, Any],
                     parent_family_id: Optional[int]) -> None:
        """Every EVALUATED child (valid or invalid). `failure` maps class -> list of witnesses;
        a class present with a non-empty list means the child tripped it. Valid children pass {}.
        Novelty-rejected children never reach the checker and are NOT passed here."""
        if failure is None:
            failure = {}
        a = self.cfg.validity_ema_alpha
        for k in self.classes:
            wl = failure.get(k, []) or []
            x = 1.0 if len(wl) > 0 else 0.0
            self.r[k] = (1 - a) * self.r[k] + a * x
            self.lam[k] = float(np.clip(
                self.lam[k] + self.cfg.lambda_lr * (self.r[k] - self.cfg.target_violation_rate),
                0.0, 1.0))
            if x > 0:
                for w in wl[: self.cfg.num_failure_witnesses]:
                    self.witnesses[k].append(w)
            if parent_family_id is not None:
                key = (parent_family_id, k)
                prev = self.family_feasibility.get(key, 1.0)
                self.family_feasibility[key] = (1 - a) * prev + a * (1.0 - x)

    def active_constraint(self) -> str:
        return max(self.classes, key=lambda k: self.lam[k])

    def has_witnesses(self, k: str) -> bool:
        return len(self.witnesses[k]) > 0

    def top_witnesses(self, k: str) -> List[Any]:
        ws = list(self.witnesses[k])
        ws.sort(key=lambda w: -float(w.get("amount", 0.0)))
        return ws[: self.cfg.num_failure_witnesses]

    def feasibility(self, family_id: int, k: str) -> float:
        return self.family_feasibility.get((family_id, k), 1.0)

    def check(self) -> bool:
        ks = self.active_constraint()
        lam, r = self.lam[ks], self.r[ks]
        on = (lam >= self.cfg.lambda_on and r >= self.cfg.violation_rate_on
              and self.has_witnesses(ks))  # never activate without an observed witness (§24)
        off = lam <= self.cfg.lambda_off and r <= self.cfg.violation_rate_off
        self._on_streak = self._on_streak + 1 if on else 0
        self._off_streak = self._off_streak + 1 if off else 0
        if not self.active and self._on_streak >= self.cfg.trigger_confirmation_checks:
            self.active = True
        elif self.active and self._off_streak >= self.cfg.release_confirmation_checks:
            self.active = False
        return self.active


# --------------------------------------------------------------------------------------
# SteeringPolicy (§8–§16) — combines controller states into a mode + selection/prompt.
# --------------------------------------------------------------------------------------
class SteeringPolicy:
    def __init__(self, cfg: SteeringConfig, fam: FamilyManager,
                 div: DiversityController, val: ValidityController):
        self.cfg = cfg
        self.fam = fam
        self.div = div
        self.val = val

    def current_mode(self) -> str:
        d = self.div.active and self.cfg.enable_diversity_controller
        v = self.val.active and self.cfg.enable_validity_controller
        if d and v:
            return MODE_COMBINED
        if d:
            return MODE_DIVERSITY
        if v:
            return MODE_VALIDITY
        return MODE_NORMAL

    def _quality_weight(self, t: float) -> float:
        e, l = self.cfg.island_quality_weight_early, self.cfg.island_quality_weight_late
        return e + (l - e) * t

    def family_scores(self, mode: str, t: float) -> np.ndarray:
        """Score every family for island(=family) sampling under the active mode (§9)."""
        n = np.array(self.fam.counts(), dtype=float)
        if len(n) == 0:
            return n
        F = _normalize01(self.fam.quality())
        U = _normalize01(1.0 / (1.0 + n))
        if mode == MODE_DIVERSITY:
            a = self._quality_weight(t)
            return a * F + (1 - a) * U
        if mode == MODE_VALIDITY:
            k = self.val.active_constraint()
            R = _normalize01(np.array([self.val.feasibility(f["id"], k) for f in self.fam.families]))
            w = self.cfg.validity_island_quality_weight
            return w * F + (1 - w) * R
        if mode == MODE_COMBINED:
            a = self._quality_weight(t)
            k = self.val.active_constraint()
            R = _normalize01(np.array([self.val.feasibility(f["id"], k) for f in self.fam.families]))
            return (self.cfg.combined_div_weight * (a * F + (1 - a) * U)
                    + self.cfg.combined_validity_weight * R)
        return np.ones(len(n)) / len(n)  # NORMAL: unused

    def sample_family(self, mode: str, t: float, rng: np.random.RandomState) -> int:
        scores = self.family_scores(mode, t)
        p = _softmax_floor(scores, self.cfg.island_softmax_temp, self.cfg.island_prob_floor)
        return int(rng.choice(len(p), p=p))

    def inspiration_family_scores(self, mode: str, parent_family: int, t: float) -> np.ndarray:
        """Score candidate inspiration families (§11). Prefers j != parent_family for
        diversity/combined; caller enforces the j!=i preference."""
        F = _normalize01(self.fam.quality())
        d = _normalize01(self.fam.distances_from(parent_family))
        if mode == MODE_DIVERSITY:
            a = self._quality_weight(t)
            return a * F + (1 - a) * d
        if mode == MODE_VALIDITY:
            k = self.val.active_constraint()
            R = _normalize01(np.array([self.val.feasibility(f["id"], k) for f in self.fam.families]))
            w = self.cfg.validity_insp_quality_weight
            return w * F + (1 - w) * R
        if mode == MODE_COMBINED:
            k = self.val.active_constraint()
            R = _normalize01(np.array([self.val.feasibility(f["id"], k) for f in self.fam.families]))
            return (self.cfg.combined_insp_quality_weight * F
                    + self.cfg.combined_insp_distance_weight * d
                    + self.cfg.combined_insp_feasibility_weight * R)
        return F

    # ---- prompt blocks (§13–§16). Return "" for NORMAL or when conditioning disabled. ----
    def build_prompt_addition(self, mode: str, b: int, B: int, t: float,
                              parent_family_summary: str = "",
                              inspiration_family_summary: str = "") -> str:
        if not self.cfg.enable_mode_prompt_conditioning or mode == MODE_NORMAL:
            return ""
        div = self._diversity_block(b, B, t, parent_family_summary, inspiration_family_summary)
        val = self._validity_block(b, B, t)
        if mode == MODE_DIVERSITY:
            return div
        if mode == MODE_VALIDITY:
            return val
        return self._combined_block(b, B, t, parent_family_summary, inspiration_family_summary)

    def _diversity_block(self, b, B, t, pfam, ifam) -> str:
        return (
            "\n\nSEARCH STEERING: DIVERSITY\n"
            f"\nSEARCH STAGE\nGeneration {b} / {B}\nNormalized stage t = {t:.3f}\n"
            f"\nDIVERSITY STATE\nCurrent algorithm families: {self.fam.num_families()}\n"
            f"Family entropy: {self.div.H:.3f}\n"
            f"Desired entropy for this stage: {self.div.H_target:.3f}\n"
            f"Diversity pressure gamma: {self.div.gamma:.3f}\n"
            f"Dominant family fraction: {self.div.dominant:.1%}\n"
            f"\nPARENT FAMILY\n{pfam}\n\nINSPIRATION FAMILY\n{ifam}\n"
            "\nThe search has become too concentrated in one algorithm family for this "
            "stage of the search.\n\nGenerate a meaningfully different algorithmic child.\n"
            "Do not make only parameter, coefficient, threshold, tolerance, or iteration-count "
            "changes.\nUse the different-family inspiration as evidence of an alternative "
            "algorithmic direction. You may borrow or recombine useful mechanisms, but the child "
            "should remain a coherent algorithm rather than a cosmetic combination of code.\n"
            "Preserve the primary task objective and all hard validity requirements.\n")

    def _validity_block(self, b, B, t) -> str:
        k = self.val.active_constraint()
        pretty = _PRETTY.get(k, k)
        ws = self.val.top_witnesses(k)
        wlines = "\n".join(render_witness(k, w) for w in ws) or "- (aggregate only; no stored witness)"
        return (
            "\n\nSEARCH STEERING: VALIDITY\n"
            f"\nSEARCH STAGE\nGeneration {b} / {B}\nNormalized stage t = {t:.3f}\n"
            f"\nACTIVE VALIDITY PRESSURE\nConstraint: {pretty}\n"
            f"Current lambda: {self.val.lam[k]:.3f}\n"
            f"Recent violation rate: {self.val.r[k]:.1%}\n"
            f"Target violation rate: {self.cfg.target_violation_rate:.1%}\n"
            f"\nREPRESENTATIVE RECENT FAILURES\n{wlines}\n"
            "\nThis failure has occurred repeatedly in recent proposals.\n\nGenerate a child "
            "that reduces the algorithm's tendency to produce this violation while preserving "
            "the primary objective as much as possible.\nDo not merely acknowledge the failure "
            "in comments. The program logic should change in a way that makes future solutions "
            "less likely to exhibit this recurring invalidity.\nThe hard validity checker "
            "remains authoritative.\n")

    def _combined_block(self, b, B, t, pfam, ifam) -> str:
        k = self.val.active_constraint()
        pretty = _PRETTY.get(k, k)
        ws = self.val.top_witnesses(k)
        wlines = "\n".join(render_witness(k, w) for w in ws) or "- (aggregate only)"
        return (
            "\n\nSEARCH STEERING: DIVERSITY + VALIDITY\n"
            f"\nSEARCH STAGE\nGeneration {b} / {B}\nNormalized stage t = {t:.3f}\n"
            f"\nDIVERSITY PRESSURE\nFamily entropy: {self.div.H:.3f}\n"
            f"Desired entropy: {self.div.H_target:.3f}\nGamma: {self.div.gamma:.3f}\n"
            f"Dominant family fraction: {self.div.dominant:.1%}\n"
            "\nThe search is too concentrated in one algorithm family.\n"
            f"\nPARENT\nCurrent family:\n{pfam}\n\nINSPIRATION\nDifferent-family algorithm:\n{ifam}\n"
            f"\nACTIVE VALIDITY PRESSURE\nConstraint: {pretty}\n"
            f"Lambda: {self.val.lam[k]:.3f}\nRecent violation rate: {self.val.r[k]:.1%}\n"
            f"\nREPRESENTATIVE FAILURES\n{wlines}\n"
            "\nMUTATION GOAL\n\nCreate a structurally different child that explores an "
            "alternative algorithmic direction while also reducing the recurring failure.\n"
            "Do not only tune coefficients, tolerances, iteration counts, or other small "
            "parameters.\nUse the alternative-family inspiration as useful evidence, but produce "
            "a coherent new solution.\nPreserve the primary packing objective and hard validity "
            "requirements.\n")
