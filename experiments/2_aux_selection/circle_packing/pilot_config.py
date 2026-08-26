# === PILOT MOD (aux-eval): which aux evaluator is active this run + feedback text. ===
# Holds the pilot's knobs in ONE place. evaluate.py imports from here.
#
# Selection precedence: the PILOT_AUX environment variable overrides ACTIVE_AUX below.
# This lets us run the 4 conditions without editing files:
#     PILOT_AUX=none python run_evo.py ...
#     PILOT_AUX=m2   python run_evo.py ...
# The eval subprocess inherits the launcher's environment (verified: shinka launches
# `python evaluate.py ...` via subprocess.Popen with no cwd and os.environ.copy()).

import os

# Default when PILOT_AUX is unset. One of: "none", "m1", "m2", "m3".
# "none" => pure baseline (control): no aux feedback at all, oracle only.
ACTIVE_AUX_DEFAULT = "none"

# Only used if Option B (shaped selection) is ever enabled. Keep 0.0 for Option A.
LAMBDA = 0.0

_VALID = {"none", "m1", "m2", "m3"}


def active_aux() -> str:
    """Resolve the active aux: env var PILOT_AUX wins, else ACTIVE_AUX_DEFAULT."""
    aux = os.getenv("PILOT_AUX", ACTIVE_AUX_DEFAULT).strip().lower()
    if aux not in _VALID:
        raise ValueError(
            f"PILOT_AUX={aux!r} is invalid; must be one of {sorted(_VALID)}."
        )
    return aux


def feedback_note(aux: str, aux_score: float) -> str:
    """The terse, factual note appended to the model's feedback when an aux is active.

    Deliberately surfaces the structural PROPERTY (and the config's current value) without
    instructing the model to 'optimize the aux'. The oracle (sum of radii) remains the only
    stated goal. Returns "" for the control so the feedback section stays empty.
    """
    if aux == "none":
        return ""
    if aux == "m1":
        return (
            f"Note: besides maximizing sum of radii, this configuration's total overlap "
            f"depth is {aux_score:.4f} (0 means no overlap). Configurations that are nearly "
            f"feasible but slightly overlapping may become high-scoring with small "
            f"adjustments."
        )
    if aux == "m2":
        return (
            f"Note: {aux_score:.0%} of circles are touching a wall. In dense packings, most "
            f"circles contact the square's boundary."
        )
    if aux == "m3":
        return (
            f"Note: circles average {aux_score:.2f} tangencies each. Dense packings have "
            f"many circles touching neighbors."
        )
    return ""
# === END PILOT MOD (aux-eval) ===
