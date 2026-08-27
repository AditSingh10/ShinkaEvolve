"""Pure state and diagnostics for the two-phase AuxEvolve Family Model."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from representation import family_label


CONDITIONS = (
    "baseline", "observe", "warmup", "parent", "inspiration", "prompt", "full",
    "warmup_validity",
)
WARMUP_CONDITIONS = frozenset(
    ("warmup", "parent", "inspiration", "prompt", "full", "warmup_validity")
)
ONLINE_CONDITIONS = frozenset(("observe", "parent", "inspiration", "prompt", "full"))
FAMILY_CONDITIONS = frozenset(("observe", "warmup", "parent", "inspiration", "prompt", "full"))
PARENT_CONDITIONS = frozenset(("parent", "full"))
DONOR_CONDITIONS = frozenset(("inspiration", "full"))
PROMPT_CONDITIONS = frozenset(("prompt", "full"))


def role_seed(replicate_seed: int, proposal_id: int, role: str) -> int:
    raw = f"{int(replicate_seed)}:{int(proposal_id)}:{role}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def normalized(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / (np.linalg.norm(a) + 1e-12)


def entropy_from_counts(counts: Iterable[int], K: int) -> Dict[str, float]:
    a = np.asarray([x for x in counts if x > 0], dtype=float)
    if not len(a):
        raw = 0.0
    else:
        p = a / a.sum()
        raw = float(-np.sum(p * np.log(p)))
    return {
        "H_raw": raw,
        "H": raw / math.log(K) if K > 1 else 0.0,
        "N_eff": math.exp(raw),
        "top_mass": float(a.max() / a.sum()) if len(a) else 0.0,
    }


class SearchMass:
    def __init__(self, window: int):
        if window < 1:
            raise ValueError("entropy window must be positive")
        self.events = deque(maxlen=window)

    def add(self, family: Optional[int]) -> None:
        if family is not None:
            self.events.append(int(family))

    def metrics(self, K: int) -> Dict[str, float]:
        return entropy_from_counts(Counter(self.events).values(), K)


@dataclass
class Member:
    program_id: str
    score: float
    summary: str
    embedding: np.ndarray


@dataclass
class Family:
    id: int
    centroid: np.ndarray
    members: List[Member] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.members:
            return ""
        return min(
            self.members,
            key=lambda m: 1.0 - float(np.dot(m.embedding, self.centroid)),
        ).summary

    def quality(self) -> float:
        return float(statistics.median(m.score for m in self.members))


class FamilyIndex:
    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = float(similarity_threshold)
        self.families: List[Family] = []
        self.program_family: Dict[str, int] = {}
        self.births = 0
        self.margins: List[float] = []

    def assign(self, program_id: str, score: float, summary: str, embedding: Sequence[float]) -> Dict[str, Any]:
        e = normalized(embedding)
        sims = sorted(
            ((float(np.dot(e, f.centroid)), f.id) for f in self.families), reverse=True
        )
        nearest = sims[0][0] if sims else None
        second = sims[1][0] if len(sims) > 1 else None
        margin = (nearest - second) if second is not None else nearest
        created = not sims or nearest < self.threshold
        if created:
            fid = len(self.families)
            fam = Family(fid, e.copy())
            self.families.append(fam)
            self.births += 1
        else:
            fid = sims[0][1]
            fam = self.families[fid]
        member = Member(program_id, float(score), summary, e)
        fam.members.append(member)
        fam.centroid = normalized(np.mean([m.embedding for m in fam.members], axis=0))
        self.program_family[program_id] = fid
        if margin is not None:
            self.margins.append(float(margin))
        return {
            "family": fid,
            "created": created,
            "nearest_family": sims[0][1] if sims else None,
            "nearest_similarity": nearest,
            "assignment_margin": margin,
        }

    def family_of(self, program_id: Optional[str]) -> Optional[int]:
        return self.program_family.get(program_id) if program_id else None

    def population_entropy(self) -> float:
        return entropy_from_counts((len(f.members) for f in self.families), len(self.families))["H"]

    def probabilities(self, t: float) -> np.ndarray:
        """pi_f(t)=(1-t)/K + t*softmax(median objective)_f."""
        K = len(self.families)
        if not K:
            return np.array([])
        q = np.asarray([f.quality() for f in self.families], dtype=float)
        z = q - q.max()
        soft = np.exp(z) / np.exp(z).sum()
        p = (1.0 - float(np.clip(t, 0, 1))) / K + float(np.clip(t, 0, 1)) * soft
        return p / p.sum()

    def sample_family(self, t: float, rng: np.random.RandomState) -> int:
        p = self.probabilities(t)
        return int(rng.choice(len(p), p=p))

    def member_ids(self, family: int) -> List[str]:
        return [m.program_id for m in self.families[family].members]

    def summary(self, family: Optional[int]) -> Optional[str]:
        return self.families[family].summary if family is not None else None

    def freeze(self) -> None:
        """Semantic marker: callers stop assigning after warmup-only transition."""

    def child_created_births(self, initial_program_seeded: bool) -> int:
        """Count family births caused by children, excluding explicit P0 seeding."""
        return max(0, self.births - int(initial_program_seeded))

    def export(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": f.id,
                "label": family_label(f.id),
                "centroid": f.centroid.tolist(),
                "quality_median": f.quality(),
                "representative_summary": min(
                    f.members, key=lambda m: 1.0 - float(np.dot(m.embedding, f.centroid))
                ).summary,
                "representative_program_id": min(
                    f.members, key=lambda m: 1.0 - float(np.dot(m.embedding, f.centroid))
                ).program_id,
                "members": [m.program_id for m in f.members],
            }
            for f in self.families
        ]


def cosine_diagnostics(index: FamilyIndex) -> Dict[str, Optional[float]]:
    intra, inter, silhouettes = [], [], []
    all_members = [(f.id, m.embedding) for f in index.families for m in f.members]
    for i, (fi, ei) in enumerate(all_members):
        same = [1 - float(np.dot(ei, ej)) for j, (fj, ej) in enumerate(all_members) if j != i and fj == fi]
        other_groups = {
            fj: [1 - float(np.dot(ei, ej)) for j, (fk, ej) in enumerate(all_members) if j != i and fk == fj]
            for fj in range(len(index.families)) if fj != fi
        }
        if same:
            intra.extend(1 - d for d in same)
            if other_groups:
                a, b = float(np.mean(same)), min(float(np.mean(ds)) for ds in other_groups.values() if ds)
                silhouettes.append((b - a) / max(a, b, 1e-12))
        for j, (fj, ej) in enumerate(all_members):
            if j > i and fj != fi:
                inter.append(float(np.dot(ei, ej)))
    return {
        "intra_family_cosine_similarity": float(np.mean(intra)) if intra else None,
        "inter_family_cosine_similarity": float(np.mean(inter)) if inter else None,
        "cosine_silhouette": float(np.mean(silhouettes)) if silhouettes else None,
    }


class EventWriter:
    CORE_FIELDS = (
        "proposal_id", "replicate", "seed", "condition", "phase", "t", "wall_clock_sec",
        "parent_program_id", "donor_program_id", "parent_family", "donor_family",
        "cross_family_donor", "K", "H_search", "H_population", "H_raw", "N_eff",
        "top_family_search_mass", "child_valid", "child_score", "best_score_so_far",
        "child_family", "created_new_family", "nearest_family_similarity", "assignment_margin",
        "generation_seed", "generation_time_sec", "evaluation_time_sec",
        "summarization_time_sec", "embedding_time_sec",
    )
    VALIDITY_FIELDS = (
        "failure_classes", "validity_active", "active_constraint",
        "validity_prompt_injected", "r_by_constraint", "lambda_by_constraint",
        "controller_check", "trigger_on_event", "trigger_off_event",
        "representative_witnesses",
    )
    FIELDS = CORE_FIELDS + VALIDITY_FIELDS

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        missing = set(self.CORE_FIELDS) - set(record)
        if missing:
            raise ValueError(f"event missing fields: {sorted(missing)}")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({k: record.get(k) for k in self.FIELDS}) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
