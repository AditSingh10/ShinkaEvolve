"""Canonical mechanistic summaries and Family Model prompt construction."""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Optional


SUMMARY_FIELDS = (
    "broad_strategy",
    "state_representation",
    "structural_decomposition",
    "core_computational_mechanism",
    "candidate_generation_or_transition",
    "selection_and_update",
    "constraint_handling",
    "refinement_or_optimization",
    "scheduling_or_adaptation",
    "distinctive_mechanistic_details",
    "non_distinguishing_details",
)

SUMMARY_SYSTEM = """Analyze the supplied program's computational mechanism and return exactly
the following eleven fields, in this exact order, with one field per line:
broad_strategy:
state_representation:
structural_decomposition:
core_computational_mechanism:
candidate_generation_or_transition:
selection_and_update:
constraint_handling:
refinement_or_optimization:
scheduling_or_adaptation:
distinctive_mechanistic_details:
non_distinguishing_details:

Write a specific, mechanistic description after every field label. Keep each value on one line.
Describe mechanisms rather than objectives: statements such as "optimize the objective" or
"maximize under constraints" are not mechanisms. A known algorithm name is useful only when
accompanied by its concrete state, transition, update, constraint, and refinement variant.

Family granularity rule: two programs are different algorithm families when their differences
materially change state transitions, search behavior, structural decomposition, update
mechanisms, constraint handling, or refinement. Programs are not different families merely
because of variable names, syntax, harmless refactoring, small constants, numeric thresholds,
parameter tuning, solver parameters, or bug fixes that preserve the computational mechanism.
Put such incidental details only in non_distinguishing_details. The broad_strategy field is for
human orientation and must not substitute for the detailed mechanism fields.

Output only the eleven labeled lines. Do not use Markdown or add commentary."""

WARMUP_INSTRUCTION = """You are in FAMILY-SEEDING WARMUP.

Your goal is to create a genuinely different algorithmic region.

Do not produce another implementation of any existing family mechanism.

A proposal does NOT count as structurally different if it only:
- fixes bugs,
- changes constants,
- tunes parameters,
- changes ring sizes/counts,
- changes thresholds,
- changes loop ordering,
- refactors code,
- changes solver parameters,
- adds a minor heuristic while preserving the same state/transition/update mechanism.

Your proposed program should materially differ from the closest existing family in at least TWO
algorithmically consequential dimensions among:

- state representation
- structural decomposition
- core computational mechanism
- candidate generation / state transition
- selection / update rule
- constraint handling / repair
- refinement / optimization
- scheduling / adaptation

Generic task necessities may remain the same.

Before producing the final program, internally:
1. identify the closest existing family;
2. identify its distinctive mechanism;
3. propose several alternatives;
4. select one whose mechanism differs materially in at least two dimensions;
5. reject your own idea and reconsider if the only differences are parameters, bug fixes,
   constants, or syntax.

Do not output this reasoning.
Return only the required complete candidate program/patch."""


def _clean_value(value: Any) -> str:
    text = " ".join(str(value).strip().split())
    return text or "Unspecified"


def parse_summary(text: str) -> "OrderedDict[str, str]":
    """Parse model output permissively while always returning the fixed schema."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json|text)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    parsed: dict[str, Any] = {}
    try:
        candidate = json.loads(stripped)
        if isinstance(candidate, Mapping):
            parsed = dict(candidate)
    except (json.JSONDecodeError, TypeError):
        current: Optional[str] = None
        values: dict[str, list[str]] = {field: [] for field in SUMMARY_FIELDS}
        field_pattern = re.compile(
            r"^(?:\d+[.)]\s*)?(" + "|".join(map(re.escape, SUMMARY_FIELDS)) + r")\s*:\s*(.*)$",
            flags=re.IGNORECASE,
        )
        canonical_names = {field.lower(): field for field in SUMMARY_FIELDS}
        for raw_line in stripped.splitlines():
            line = raw_line.strip().lstrip("-* ")
            match = field_pattern.match(line)
            if match:
                current = canonical_names[match.group(1).lower()]
                values[current].append(match.group(2))
            elif current and line:
                values[current].append(line)
        parsed = {field: " ".join(values[field]) for field in SUMMARY_FIELDS}
    return OrderedDict((field, _clean_value(parsed.get(field, ""))) for field in SUMMARY_FIELDS)


def serialize_summary(summary: Mapping[str, Any] | str) -> str:
    """Serialize all fields in a stable order for storage, prompts, and embedding."""
    values = parse_summary(summary) if isinstance(summary, str) else OrderedDict(
        (field, _clean_value(summary.get(field, ""))) for field in SUMMARY_FIELDS
    )
    return "\n".join(f"{field}: {values[field]}" for field in SUMMARY_FIELDS)


def family_label(family_id: Optional[int]) -> Optional[str]:
    return f"F{int(family_id) + 1}" if family_id is not None else None


def build_warmup_context(families: Iterable[Any]) -> str:
    blocks = []
    for family in families:
        family_id = family["id"] if isinstance(family, Mapping) else family.id
        summary = family["summary"] if isinstance(family, Mapping) else family.summary
        blocks.append(f"FAMILY {family_label(family_id)}\n{serialize_summary(summary)}")
    existing = "\n\n".join(blocks) if blocks else "None"
    return f"{WARMUP_INSTRUCTION}\n\nEXISTING FAMILY MECHANISMS:\n\n{existing}"


def differing_major_dimensions(left: str, right: str) -> list[str]:
    """Return exact field-level differences for probe reporting only."""
    left_fields, right_fields = parse_summary(left), parse_summary(right)
    major = SUMMARY_FIELDS[1:9]
    return [field for field in major if left_fields[field] != right_fields[field]]
