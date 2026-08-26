# === ALG1 MOD (aux-discovery): propose -> execute -> gate auxiliary objectives. ===
# Algorithm 1 discovers aux functions instead of hand-designing them. An LLM proposes
# candidate aux functions from lineage evidence; each is executed (safely) over the archive;
# survivors must pass TWO divergence filters -- differ from the ORACLE (not a no-op) and be
# NOVEL vs the auxes we already keep (not redundant). Survivors become Algorithm-2 bandit
# arms, where q-learning then up-weights the helpful ones and discards the useless ones.
#
# This module is the pure logic (no LLM client, no DB) so it is unit-testable offline:
#   compile_aux / safe_execute_aux : run untrusted aux code in a restricted namespace
#   build_proposer_prompt          : the design brief shown to the proposer LLM
#   parse_proposed_auxes           : parse the LLM response into candidates
#   gate_candidates                : the two-filter (oracle + novelty) admission test
from __future__ import annotations

import ast
import re
import signal
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .aux_selection import (
    build_aux_probabilities,
    js_divergence,
    oracle_probabilities,
)

# ------------------------------------------------------------------ safe execution
# A deliberately tiny namespace: numeric work only. No imports, I/O, eval/exec, attribute
# tricks. Aux code operates on numpy arrays and returns a float.
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range,
    "float": float, "int": int, "bool": bool, "round": round, "sorted": sorted,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "any": any,
    "all": all, "list": list, "tuple": tuple, "set": set, "dict": dict, "pow": pow,
}
_FORBIDDEN = ("import", "__", "open(", "eval(", "exec(", "compile(", "globals(",
              "locals(", "getattr", "setattr", "delattr", "input(", "os.", "sys.",
              "subprocess", "socket", ".write", "lambda os")


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def compile_aux(code: str, arg_names=("centers", "radii")) -> Callable:
    """Compile aux source into a callable `aux(*arg_names)`. Raises on unsafe/invalid code.

    Static checks first (forbidden tokens, must define `aux` with the right arity), then
    exec in a restricted namespace. `arg_names` lets a different problem use a different
    signature (e.g. ("boards",) for 2048) while keeping the same safety checks. The returned
    callable still runs under a timeout via safe_execute_aux -- compile does not run the body."""
    low = code.lower()
    for tok in _FORBIDDEN:
        if tok in low:
            raise ValueError(f"forbidden token in aux code: {tok!r}")
    tree = ast.parse(code)  # raises SyntaxError on bad code
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    aux_defs = [f for f in funcs if f.name == "aux"]
    if not aux_defs:
        raise ValueError("aux code must define a function named `aux`")
    args = aux_defs[0].args.args
    if len(args) != len(arg_names):
        raise ValueError(f"`aux` must take exactly {tuple(arg_names)}")
    ns: Dict = {"__builtins__": _SAFE_BUILTINS, "np": np, "numpy": np}
    exec(compile(tree, "<aux>", "exec"), ns)  # noqa: S102 -- restricted ns by construction
    fn = ns.get("aux")
    if not callable(fn):
        raise ValueError("`aux` did not compile to a callable")
    return fn


def safe_execute_aux(fn: Callable, centers, radii, timeout_s: float = 1.0) -> float:
    """Run a compiled aux on one artifact, guarded by a timeout. Returns a finite float or
    raises. Timeout uses SIGALRM (best-effort; only in the main thread)."""
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    use_alarm = False
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        use_alarm = True
    except (ValueError, AttributeError):
        pass  # not main thread / unsupported -> run without alarm
    try:
        val = float(fn(centers, radii))
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
    if not np.isfinite(val):
        raise ValueError("aux returned a non-finite value")
    return val


def score_aux_over_archive(fn: Callable, artifacts: Sequence[dict],
                           timeout_s: float = 1.0):
    """Run aux over every archived artifact ({'centers':..,'radii':..}). Returns
    (values, ok, reason): ok=False if it crashes, is non-finite, or is CONSTANT across the
    pool (a dead arm carries no selection signal)."""
    vals = []
    for art in artifacts:
        try:
            vals.append(safe_execute_aux(fn, art["centers"], art["radii"], timeout_s))
        except Exception as e:  # noqa: BLE001
            return None, False, f"execution failed: {e}"
    arr = np.asarray(vals, dtype=float)
    if arr.size < 2:
        return arr, False, "too few programs to evaluate"
    if float(np.std(arr)) < 1e-9:
        return arr, False, "constant across the archive (dead arm)"
    return arr, True, "ok"


# ------------------------------------------------------------------ proposer prompt
_DESIGN_BRIEF = """\
You are proposing AUXILIARY OBJECTIVES to guide an evolutionary search for circle packings
(n=26 unit-square circles; the PRIMARY objective is to MAXIMISE the sum of radii). An aux is
a Python function of a candidate's geometry that scores some STRUCTURAL property; it is used
only to bias which parents the search breeds from -- never shown as the objective.

DESIGN PRINCIPLES (follow all):
1. Do NOT rediscover the primary objective. An aux that is basically the sum of radii (or a
   monotone function of it) is useless -- it just re-picks the same parents. Aim for
   something the objective does NOT already capture.
2. Do NOT rediscover feasibility. Every candidate here is already a VALID packing (no
   overlaps, inside the square). Measuring "how much overlap" or "how valid" is constant
   across all candidates and carries ZERO signal. Do not propose it.
3. Find something LATENT and UNIQUE: local geometry, slack/looseness, contact structure,
   spatial arrangement, size heterogeneity, symmetry-breaking, wasted space, etc. -- a
   property that plausibly PREDICTS which packings can still be improved.
4. Be NOVEL relative to the auxes already in use (listed below). Do not propose something
   whose ranking of packings would essentially duplicate an existing aux.

Each aux receives:
    centers : np.ndarray (26, 2)   circle centres (x, y) in [0,1]^2
    radii   : np.ndarray (26,)     circle radii
and must return a single float. It may use `np` (numpy) only -- no imports, no I/O.

OUTPUT FORMAT -- emit 4 to 8 candidates, each exactly:

### AUX: <short_snake_case_name>
# direction: +1        (or -1; +1 = higher is better, -1 = lower is better)
# rationale: <one sentence: what latent property this captures and why it may help>
def aux(centers, radii):
    ...
    return <float>
"""


def build_proposer_prompt(best_examples: List[dict], worst_examples: List[dict],
                          lineage_deltas: List[dict],
                          existing_auxes: Optional[List[dict]] = None) -> str:
    """Assemble the proposer prompt: design brief + lineage evidence + existing auxes.

    best/worst_examples : [{'score': float, 'centers': [[x,y]..], 'radii': [..]}]
    lineage_deltas      : [{'delta': float, 'parent_score':.., 'child_score':..,
                            'summary': '<short geometry-change description>'}]
    existing_auxes      : [{'name':.., 'rationale':.., 'q': float}]  (avoid duplicating)
    """
    def fmt_pack(e):
        c = np.round(np.asarray(e["centers"], float), 3).tolist()
        r = np.round(np.asarray(e["radii"], float), 3).tolist()
        return f"  score={e['score']:.4f}  radii={r}\n    centers={c}"

    parts = [_DESIGN_BRIEF, "\n=== EVIDENCE FROM THE CURRENT SEARCH ===",
             "\nBEST packings (high sum of radii):"]
    parts += [fmt_pack(e) for e in best_examples]
    parts.append("\nWORST valid packings (low sum of radii):")
    parts += [fmt_pack(e) for e in worst_examples]
    if lineage_deltas:
        parts.append("\nPARENT->CHILD edits (what changed, and the score delta):")
        for d in lineage_deltas:
            parts.append(f"  Δscore={d['delta']:+.4f}  ({d.get('summary','')})")
    if existing_auxes:
        parts.append("\nAUXES ALREADY IN USE (propose something DIFFERENT from these):")
        for a in existing_auxes:
            q = f", q={a['q']:+.3f}" if a.get("q") is not None else ""
            parts.append(f"  - {a['name']}: {a.get('rationale','')}{q}")
    else:
        parts.append("\n(No auxes in use yet -- you are proposing the first batch.)")
    parts.append("\nNow emit your candidate auxes in the exact OUTPUT FORMAT above.")
    return "\n".join(parts)


# ------------------------------------------------------------------ response parsing
_AUX_HEADER = re.compile(r"^###\s*AUX:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)", re.M)
_DIR_RE = re.compile(r"direction:\s*([+-]?\s*1)", re.I)
_RAT_RE = re.compile(r"rationale:\s*(.+)", re.I)


def parse_proposed_auxes(text: str, arg_names=("centers", "radii")) -> List[dict]:
    """Parse an LLM response into [{name, direction, rationale, code}] candidates.

    Splits on '### AUX:' markers; for each block extracts the name, direction, rationale,
    and the aux source (from the first `def aux` to the end of the block). Silently drops
    blocks that don't parse or don't define `aux` with the expected `arg_names` arity."""
    out = []
    matches = list(_AUX_HEADER.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        name = m.group("name")
        dmatch = _DIR_RE.search(block)
        direction = int((dmatch.group(1).replace(" ", "")) if dmatch else "1")
        direction = 1 if direction >= 0 else -1
        rmatch = _RAT_RE.search(block)
        rationale = rmatch.group(1).strip() if rmatch else ""
        # code: strip markdown fences, take from `def aux` onward
        code_region = block[block.find("def aux"):] if "def aux" in block else ""
        code = code_region.replace("```", "").strip()
        if not code:
            continue
        try:
            compile_aux(code, arg_names=arg_names)  # validates safety + shape; discard if bad
        except Exception:
            continue
        out.append({"name": name, "direction": direction,
                    "rationale": rationale, "code": code})
    return out


# ------------------------------------------------------------------ the two-filter gate
def _residual_fraction(aux_values: Sequence[float], oracle_values: Sequence[float]) -> float:
    """Fraction of the aux's variance NOT explained by the oracle (= 1 - R^2 of aux~oracle).

    ~0 means the aux is essentially a linear function of the oracle -- a 'rediscovery' that
    carries no independent signal (and, once residualized, collapses to uniform selection,
    which would spuriously look 'divergent'). ~1 means fully independent of the oracle."""
    a = np.asarray(aux_values, dtype=float)
    o = np.asarray(oracle_values, dtype=float)
    va = float(np.var(a))
    if va < 1e-12:
        return 0.0                      # constant aux -> no signal at all
    if float(np.var(o)) < 1e-12:
        return 1.0                      # oracle flat -> nothing to rediscover
    A = np.vstack([o, np.ones_like(o)]).T
    try:
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        resid = a - A @ coef
        return float(np.var(resid) / va)
    except Exception:
        return 1.0


def gate_candidates(candidate_values: Dict[str, Sequence[float]],
                    candidate_directions: Dict[str, int],
                    oracle_values: Sequence[float],
                    children_counts: Sequence[int],
                    existing_aux_values: Optional[Dict[str, Sequence[float]]] = None,
                    existing_aux_directions: Optional[Dict[str, int]] = None,
                    lam: float = 10.0, residualize: bool = True,
                    tau_info: float = 0.10, tau_oracle: float = 0.10,
                    tau_novelty: float = 0.10) -> Dict[str, dict]:
    """Two-filter admission test. A candidate passes iff:
      (A) ORACLE filter -- it is not a rediscovery of the primary objective:
            residual_fraction (variance left after regressing the oracle out) >= tau_info
            AND its residualized selection distribution diverges from the oracle's
            (JS >= tau_oracle). The residual_fraction guard is essential: an aux that is a
            linear function of the oracle residualizes to ~uniform, which would otherwise
            look 'divergent' and slip through.
      (B) NOVELTY filter -- its distribution is far from every existing aux's
            (min JS >= tau_novelty), so we don't accumulate near-duplicates.
    Returns per-candidate {passed, residual_frac, js_oracle, js_nearest_aux, nearest_aux, reason}."""
    n = len(oracle_values)
    p_oracle = oracle_probabilities(oracle_values, children_counts, lam=lam)
    existing_probs = {}
    if existing_aux_values:
        for name, vals in existing_aux_values.items():
            if len(vals) == n:
                existing_probs[name] = build_aux_probabilities(
                    vals, int((existing_aux_directions or {}).get(name, 1)),
                    children_counts, lam=lam,
                    oracle_values=oracle_values if residualize else None)

    results = {}
    for name, vals in candidate_values.items():
        if len(vals) != n:
            results[name] = {"passed": False, "reason": "value/pool length mismatch"}
            continue
        res_frac = _residual_fraction(vals, oracle_values)
        p = build_aux_probabilities(
            vals, int(candidate_directions.get(name, 1)), children_counts, lam=lam,
            oracle_values=oracle_values if residualize else None)
        js_or = js_divergence(p, p_oracle)
        nearest, js_near = None, float("inf")
        for ename, ep in existing_probs.items():
            j = js_divergence(p, ep)
            if j < js_near:
                js_near, nearest = j, ename

        if res_frac < tau_info:
            passed, reason = False, f"rediscovers oracle (residual_frac={res_frac:.3f} < {tau_info})"
        elif js_or < tau_oracle:
            passed, reason = False, f"selection too close to oracle (JS={js_or:.3f} < {tau_oracle})"
        elif existing_probs and js_near < tau_novelty:
            passed, reason = False, f"redundant with '{nearest}' (JS={js_near:.3f} < {tau_novelty})"
        else:
            passed, reason = True, "ok"
        results[name] = {
            "passed": passed, "residual_frac": res_frac, "js_oracle": js_or,
            "js_nearest_aux": None if js_near == float("inf") else js_near,
            "nearest_aux": nearest, "reason": reason,
        }
    return results
# === END ALG1 MOD (aux-discovery) ===
