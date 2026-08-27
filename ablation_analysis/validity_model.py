"""Prompt-only validity state for the ``warmup_validity`` ablation.

This module deliberately has no dependency on family quality, selection, or sampling.
It observes canonical evaluator outcomes and can only render an optional prompt block.
"""
from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CONSTRAINTS = (
    "runtime", "timeout", "compile", "wrong_answer", "malformed", "boundary", "overlap",
)

_PRETTY = {
    "overlap": "pairwise non-overlap",
    "boundary": "inside-unit-square boundary",
    "runtime": "runtime execution",
    "malformed": "well-formed output (correct interface, shape, and finite values)",
    "timeout": "evaluator time limit",
    "compile": "compilation or parsing",
    "wrong_answer": "answer correctness",
}


@dataclass(frozen=True)
class ValidityConfig:
    enabled: bool = True
    validity_ema_alpha: float = 0.15
    lambda_lr: float = 0.10
    target_violation_rate: float = 0.15
    lambda_on: float = 0.60
    lambda_off: float = 0.25
    violation_rate_on: float = 0.30
    violation_rate_off: float = 0.15
    num_failure_witnesses: int = 3
    witness_buffer_size: int = 12
    controller_warmup: int = 20
    controller_check_interval: int = 5
    trigger_confirmation_checks: int = 2
    release_confirmation_checks: int = 2
    constraint_classes: Tuple[str, ...] = CONSTRAINTS

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "ValidityConfig":
        raw = dict(values or {})
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown warmup_validity settings: {sorted(unknown)}")
        if "constraint_classes" in raw:
            raw["constraint_classes"] = tuple(raw["constraint_classes"])
        return cls(**raw)


def _message_witness(message: str) -> Dict[str, Any]:
    return {"error": message[:300] or "evaluation failed"}


def _as_witnesses(value: Any, fallback: str) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"note": str(item)} for item in value]
    if isinstance(value, dict):
        return [value]
    if value:
        return [{"note": fallback}]
    return []


def classify_failure(
    correct: bool,
    violations: Optional[Mapping[str, Any]] = None,
    error_msg: Optional[str] = None,
    timed_out: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Classify one canonical evaluation into problem-agnostic failure classes."""
    if correct:
        return {}
    message = " ".join(str(error_msg or "").strip().split())
    if timed_out:
        return {"timeout": [_message_witness(message or "evaluator timed out")]}

    structured: Dict[str, List[Dict[str, Any]]] = {}
    for key in CONSTRAINTS:
        witnesses = _as_witnesses((violations or {}).get(key), f"{key} violation")
        if witnesses:
            structured[key] = witnesses
    if structured:
        return structured

    lower = message.lower()
    overlap = re.search(
        r"circles?\s+(\d+)\s*(?:&|and)\s*(\d+)\s+overlap.*?dist:\s*"
        r"([-+0-9.eE]+).*?sum radii:\s*([-+0-9.eE]+)",
        message,
        flags=re.IGNORECASE,
    )
    if overlap:
        distance, radius_sum = float(overlap.group(3)), float(overlap.group(4))
        return {"overlap": [{
            "i": int(overlap.group(1)),
            "j": int(overlap.group(2)),
            "amount": max(0.0, radius_sum - distance),
            "error": message[:300],
        }]}
    if "overlap" in lower:
        return {"overlap": [_message_witness(message)]}
    if "outside" in lower or "boundary" in lower or "unit square" in lower:
        return {"boundary": [_message_witness(message)]}
    if any(token in lower for token in ("timed out", "timeout", "time limit", "timedout")):
        return {"timeout": [_message_witness(message)]}
    if any(token in lower for token in (
        "candidate patch failure", "wrong shape", "shape incorrect", "non-finite", "nan",
        "inhomogeneous", "negative radii", "no valid output", "malformed",
    )):
        return {"malformed": [_message_witness(message)]}
    if any(token in lower for token in (
        "syntax error", "invalid syntax", "was never closed", "unexpected token", "loaderror",
        "undefvarerror", "expected ';'", "expected '}'", "expected declaration",
        "cannot import", "no such module", "cannot find package", "compile",
    )):
        return {"compile": [_message_witness(message)]}
    if any(token in lower for token in (
        "incorrect", "mismatch", "does not match", "wrong answer", "lower than",
    )) or ("expected" in lower and "got" in lower and "unpack" not in lower):
        return {"wrong_answer": [_message_witness(message)]}
    return {"runtime": [_message_witness(message or "execution failed")]}


def render_witness(constraint: str, witness: Mapping[str, Any]) -> str:
    if constraint == "overlap" and "i" in witness and "j" in witness:
        return (
            f"- circles {witness['i']} and {witness['j']} overlap by "
            f"{float(witness.get('amount', 0.0)):.4f}"
        )
    text = str(witness.get("error", witness.get("note", "evaluation failed")))
    return f"- {text[:180]}"


class ValidityController:
    """EMA/dual controller whose sole action is an optional mutation-prompt block."""

    def __init__(self, config: ValidityConfig):
        self.config = config
        self.classes = tuple(config.constraint_classes)
        invalid = set(self.classes) - set(CONSTRAINTS)
        if invalid:
            raise ValueError(f"unsupported validity constraints: {sorted(invalid)}")
        self.active = False
        self.r: Dict[str, float] = {key: 0.0 for key in self.classes}
        self.lam: Dict[str, float] = {key: 0.0 for key in self.classes}
        self.max_lam: Dict[str, float] = {key: 0.0 for key in self.classes}
        self.witnesses: Dict[str, deque] = {
            key: deque(maxlen=config.witness_buffer_size) for key in self.classes
        }
        self.failure_counts: Counter = Counter()
        self.episodes: List[Dict[str, Any]] = []
        self._on_streak = 0
        self._off_streak = 0

    def update(self, failure: Optional[Mapping[str, Sequence[Mapping[str, Any]]]]) -> None:
        failure = failure or {}
        alpha = self.config.validity_ema_alpha
        for key in self.classes:
            witnesses = list(failure.get(key, []) or [])
            x = 1.0 if witnesses else 0.0
            self.r[key] = (1.0 - alpha) * self.r[key] + alpha * x
            self.lam[key] = float(np.clip(
                self.lam[key]
                + self.config.lambda_lr * (self.r[key] - self.config.target_violation_rate),
                0.0,
                1.0,
            ))
            self.max_lam[key] = max(self.max_lam[key], self.lam[key])
            if witnesses:
                self.failure_counts[key] += 1
                for witness in witnesses[: self.config.num_failure_witnesses]:
                    self.witnesses[key].append(dict(witness))

    def active_constraint(self) -> str:
        return max(self.classes, key=lambda key: self.lam[key])

    def top_witnesses(self, constraint: Optional[str] = None) -> List[Dict[str, Any]]:
        key = constraint or self.active_constraint()
        witnesses = list(self.witnesses[key])
        witnesses.sort(key=lambda item: -float(item.get("amount", 0.0)))
        return witnesses[: self.config.num_failure_witnesses]

    def should_check(self, proposal_id: int) -> bool:
        return (
            proposal_id > self.config.controller_warmup
            and proposal_id % self.config.controller_check_interval == 0
        )

    def check(self, proposal_id: int) -> Dict[str, Any]:
        result = {
            "controller_check": False,
            "trigger_on_event": False,
            "trigger_off_event": False,
        }
        if not self.should_check(proposal_id):
            return result
        result["controller_check"] = True
        constraint = self.active_constraint()
        on = (
            self.lam[constraint] >= self.config.lambda_on
            and self.r[constraint] >= self.config.violation_rate_on
            and bool(self.witnesses[constraint])
        )
        off = (
            self.lam[constraint] <= self.config.lambda_off
            and self.r[constraint] <= self.config.violation_rate_off
        )
        self._on_streak = self._on_streak + 1 if on else 0
        self._off_streak = self._off_streak + 1 if off else 0
        if not self.active and self._on_streak >= self.config.trigger_confirmation_checks:
            self.active = True
            result["trigger_on_event"] = True
            self.episodes.append({
                "start_proposal": proposal_id,
                "first_prompt_proposal": proposal_id + 1,
                "end_proposal": None,
                "last_prompt_proposal": None,
                "active_constraint": constraint,
            })
        elif self.active and self._off_streak >= self.config.release_confirmation_checks:
            self.active = False
            result["trigger_off_event"] = True
            if self.episodes and self.episodes[-1]["end_proposal"] is None:
                self.episodes[-1]["end_proposal"] = proposal_id
                self.episodes[-1]["last_prompt_proposal"] = proposal_id
        return result

    def prompt_block(self, proposal_id: int, proposal_budget: int) -> str:
        if not self.active:
            return ""
        constraint = self.active_constraint()
        witnesses = self.top_witnesses(constraint)
        lines = "\n".join(render_witness(constraint, item) for item in witnesses)
        return (
            "\n\nSEARCH STEERING: VALIDITY\n"
            f"Proposal: {proposal_id} / {proposal_budget}\n"
            f"Active constraint: {_PRETTY.get(constraint, constraint)}\n"
            f"Current lambda: {self.lam[constraint]:.3f}\n"
            f"Recent violation rate: {self.r[constraint]:.1%}\n"
            f"Target violation rate: {self.config.target_violation_rate:.1%}\n"
            f"Representative recent failures:\n{lines}\n"
            "This failure is recurring. Change the actual program logic so the child is less "
            "likely to exhibit it, while preserving objective quality as much as possible. "
            "Do not merely acknowledge the failure in comments; the canonical evaluator remains "
            "authoritative."
        )
