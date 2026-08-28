#!/usr/bin/env python3
"""Sequential, one-shot runner for the AuxEvolve Circle Packing Family Model."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from family_model import (CONDITIONS, CREATE_ACTION, CREATE_CONDITIONS, DONOR_CONDITIONS,
    EventWriter, FamilyIndex, ONLINE_CONDITIONS, PARENT_CONDITIONS, PROMPT_CONDITIONS,
    SearchMass, WARMUP_CONDITIONS, budget_fraction, cosine_diagnostics, role_seed,
    write_json)
from representation import (SUMMARY_SYSTEM, build_warmup_context, family_label,
    serialize_summary)
from shinka.edit.apply_full import apply_full_patch
from shinka.embed import EmbeddingClient
from shinka.llm.client import get_client_llm
from validity_model import ValidityConfig, ValidityController, classify_failure

TASK = """You are an expert mathematician and computational geometer. Improve the supplied Python
algorithm for packing 26 non-overlapping circles inside a unit square, maximizing the sum of radii.
Return exactly one complete Python code block. Preserve the EVOLVE-BLOCK markers and fixed
run_packing interface. Use only packages already available to the supplied program."""
CREATE_TASK = """You are an expert mathematician and computational geometer. Develop a complete
Python algorithm for packing 26 non-overlapping circles inside a unit square, maximizing the sum
of radii. Return exactly one complete Python code block. Preserve the EVOLVE-BLOCK markers and
fixed run_packing interface. Use only packages already available to the supplied program."""


@dataclass
class Program:
    id: str
    code: str
    score: float
    family: Optional[int] = None
    children: int = 0


@dataclass
class SearchSelection:
    search_action: Optional[str]
    mutation_intent: Optional[str]
    create_probability: Optional[float]
    parent: Optional[Program]
    donor: Optional[Program]
    parent_family: Optional[int]
    donor_family: Optional[int]


def weighted_program_from_pool(pool, rng):
    """Sample by pool-local robust quality and reproduction count."""
    pool = list(pool)
    if not pool:
        raise ValueError("cannot sample from an empty program pool")

    scores = np.asarray([program.score for program in pool], dtype=float)
    median = float(np.median(scores))
    mad = max(float(np.median(np.abs(scores - median))), 1e-6)
    z = 10.0 * (scores - median) / mad

    sigmoid = np.empty_like(z)
    nonnegative = z >= 0
    sigmoid[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
    exp_z = np.exp(z[~nonnegative])
    sigmoid[~nonnegative] = exp_z / (1.0 + exp_z)

    weights = sigmoid * np.asarray(
        [1.0 / (1 + program.children) for program in pool], dtype=float
    )
    probabilities = weights / weights.sum()
    return pool[int(rng.choice(len(pool), p=probabilities))]


class Experiment:
    def __init__(self, args, cfg):
        self.args, self.cfg, self.B, self.b = args, cfg, args.proposals, 0
        self.start = time.monotonic()
        run_name = getattr(args, "run_name", None)
        if not run_name:
            run_name = f"run_{args.condition}_rep{args.replicate}_seed{args.seed}"
        self.run_dir = ROOT / "results" / run_name
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty run: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventWriter(self.run_dir / "events.jsonl")
        self.family = FamilyIndex(args.family_similarity_threshold)
        self.search_mass = SearchMass(args.entropy_window)
        self.programs: Dict[str, Program] = {}
        self.best = None
        self.best_at = 0
        self.warmup_done = args.condition not in WARMUP_CONDITIONS
        self.warmup_end_b = self.warmup_end_sec = 0
        self.warmup_reached_k = False
        self.warmup_valid_children = 0
        self.initial_program_seeded = False
        self.valid_children = 0
        self.invalid_children = 0
        self.family_birth_proposals = []
        self.validity_history = []
        validity_config = ValidityConfig.from_mapping(cfg.get("warmup_validity"))
        self.validity = (
            ValidityController(validity_config)
            if args.condition == "warmup_validity" and validity_config.enabled
            else None
        )
        self.validity_prompt_count = 0
        self.create_attempts = 0
        self.create_valid = 0
        self.create_new_family_successes = 0
        self.last_evaluation = {"error": None, "timed_out": False}
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.gen_client, self.gen_model, _ = get_client_llm(cfg["generation_model"])
        if hasattr(self.gen_client, "with_options"):
            self.gen_client = self.gen_client.with_options(max_retries=0)
        self.embed_client = EmbeddingClient(model_name=cfg["embedding_model"])

    def evaluate(self, path: Path, directory: Path) -> Tuple[bool, Optional[float], float]:
        results = directory / "evaluation"
        results.mkdir(parents=True, exist_ok=True)
        self.last_evaluation = {"error": None, "timed_out": False}
        begin = time.monotonic()
        cmd = [sys.executable, str(ROOT / "evaluator_entry.py"),
               "--program_path", str(path), "--results_dir", str(results)]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True,
                timeout=self.args.evaluator_timeout_sec, check=False)
            (directory / "evaluator.stdout").write_text(done.stdout, encoding="utf-8")
            (directory / "evaluator.stderr").write_text(done.stderr, encoding="utf-8")
            fallback_error = (done.stderr or done.stdout or "").strip()[-1000:]
        except subprocess.TimeoutExpired as exc:
            fallback_error = f"timeout: {exc}"
            (directory / "evaluator.stderr").write_text(fallback_error + "\n", encoding="utf-8")
            self.last_evaluation = {"error": fallback_error, "timed_out": True}
        elapsed = time.monotonic() - begin
        try:
            correctness = json.loads((results / "correct.json").read_text())
            valid = correctness["correct"]
            score = json.loads((results / "metrics.json").read_text()).get("combined_score")
            self.last_evaluation = {
                "error": correctness.get("error"),
                "timed_out": False,
            }
            return bool(valid), float(score) if score is not None else None, elapsed
        except Exception:
            if not self.last_evaluation.get("timed_out"):
                self.last_evaluation = {"error": fallback_error, "timed_out": False}
            return False, None, elapsed

    def observe(self, program: Program, proposal_id: int):
        seed = role_seed(self.args.seed, proposal_id, "summarization")
        begin = time.monotonic()
        response = self.gen_client.chat.completions.create(model=self.gen_model, seed=seed,
            temperature=0.0, max_tokens=768, n=1, messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": program.code[:6000]}])
        summary = serialize_summary(response.choices[0].message.content or "")
        summary_time = time.monotonic() - begin
        begin = time.monotonic()
        embedding, _ = self.embed_client.get_embedding(summary)
        embedding_time = time.monotonic() - begin
        assignment = self.family.assign(program.id, program.score, summary, embedding)
        program.family = assignment["family"]
        assignment["structured_summary"] = summary
        assignment["embedding"] = embedding
        return assignment, summary_time, embedding_time

    def baseline_program(self, rng, exclude=None):
        pool = [p for p in self.programs.values() if p.id != exclude]
        if not pool:
            return self.programs[exclude] if exclude else next(iter(self.programs.values()))
        return weighted_program_from_pool(pool, rng)

    def member_program(self, family, rng):
        ids = self.family.member_ids(family)
        pool = [self.programs[program_id] for program_id in ids]
        return weighted_program_from_pool(pool, rng)

    def select(self, proposal_id, phase, t):
        frng = np.random.RandomState(role_seed(self.args.seed, proposal_id, "family_sampling"))
        prng = np.random.RandomState(role_seed(self.args.seed, proposal_id, "parent_sampling"))
        drng = np.random.RandomState(role_seed(self.args.seed, proposal_id, "donor_sampling"))
        create_probability = None
        if phase == "normal":
            create_probability = 0.0

        if phase == "warmup" and self.family.families:
            parent_family = self.family.sample_uniform_family(frng)
            parent = self.member_program(parent_family, prng)
        elif (phase == "normal" and self.args.condition in CREATE_CONDITIONS
                and self.family.families):
            action_probabilities = self.family.create_probabilities(t)
            create_probability = float(action_probabilities[-1])
            action = self.family.sample_create_action(t, frng)
            if action == CREATE_ACTION:
                return SearchSelection(
                    search_action=CREATE_ACTION,
                    mutation_intent=CREATE_ACTION,
                    create_probability=create_probability,
                    parent=None,
                    donor=None,
                    parent_family=None,
                    donor_family=None,
                )
            parent = self.member_program(action, prng)
        elif (phase == "normal" and self.args.condition in PARENT_CONDITIONS
                and self.family.families):
            parent = self.member_program(
                self.family.sample_annealed_family(t, frng), prng
            )
        else:
            parent = self.baseline_program(prng)
        if phase == "warmup" and self.family.families:
            donor_family = self.family.sample_uniform_family(drng)
            donor = self.member_program(donor_family, drng)
        elif phase == "normal" and self.args.condition in DONOR_CONDITIONS and self.family.families:
            if self.args.condition in CREATE_CONDITIONS:
                donor_family = self.family.sample_create_donor_family(t, drng)
            else:
                donor_family = self.family.sample_annealed_family(t, drng)
            donor = self.member_program(donor_family, drng)
        else:
            donor = self.baseline_program(drng, parent.id)
        parent_family, donor_family = parent.family, donor.family
        mutation_intent = None
        search_action = None
        if phase == "normal" and parent_family is not None:
            search_action = family_label(parent_family)
            if donor_family is not None and self.args.condition in CREATE_CONDITIONS:
                mutation_intent = (
                    "REFINE" if parent_family == donor_family else "COMPOSE"
                )
        return SearchSelection(
            search_action=search_action,
            mutation_intent=mutation_intent,
            create_probability=create_probability,
            parent=parent,
            donor=donor,
            parent_family=parent_family,
            donor_family=donor_family,
        )

    def create_prompt(self) -> str:
        family_context = "\n\n".join(
            f"Family {family_label(family.id)}:\n{family.summary}"
            for family in self.family.families
        )
        return (
            "SEARCH INTENT: CREATE\n\n"
            "CURRENT ALGORITHM FAMILIES\n\n"
            f"{family_context}\n\n"
            "Develop a new algorithmic approach that is meaningfully different from the "
            "approaches already represented above.\n\n"
            "Do not merely modify parameters, constants, thresholds, tolerances, solver "
            "settings, or surface-level implementation details. Change the underlying "
            "computational strategy.\n\n"
            "Novel, hybrid, or task-specific mechanisms are allowed. Continue optimizing "
            "the primary task objective.\n\n"
            "Use the family summaries only to understand what has already been explored; "
            "do not treat them as mechanisms that must be reused.\n\n"
            "Generate one complete candidate now."
        )

    @staticmethod
    def intent_prompt(mutation_intent: Optional[str]) -> str:
        if mutation_intent == "REFINE":
            return (
                "SEARCH INTENT: REFINE\n"
                "The parent and donor belong to the same algorithm family. Use the parent as "
                "the main solution; the donor provides useful implementation choices or "
                "mechanisms from another variation in that family. Preserve the effective core "
                "strategy while making a meaningful algorithmic, implementation, or "
                "optimization improvement. Avoid cosmetic edits. Change parameters, "
                "coefficients, tolerances, or iteration counts only when they support a "
                "substantive improvement."
            )
        if mutation_intent == "COMPOSE":
            return (
                "SEARCH INTENT: COMPOSE\n"
                "The parent and donor belong to different algorithm families. Start from the "
                "parent and identify complementary mechanisms from the donor. Combine, adapt, "
                "or integrate useful mechanisms; do not copy the donor or replace the parent "
                "wholesale. The child may remain in the parent family, move toward the donor "
                "family, or form a new hybrid family. Do not explicitly optimize novelty. "
                "Optimize the primary task objective and avoid superficial parameter-only "
                "changes."
            )
        return ""

    def prompt(self, selection: SearchSelection, phase: str):
        if selection.search_action == CREATE_ACTION:
            return self.create_prompt()
        parent, donor = selection.parent, selection.donor
        if parent is None or donor is None:
            raise ValueError("non-CREATE generation requires a parent and donor")
        context = ""
        if phase == "warmup":
            context = build_warmup_context(self.family.families)
        elif self.args.condition in PROMPT_CONDITIONS:
            intent = (
                self.intent_prompt(selection.mutation_intent)
                if self.args.condition in CREATE_CONDITIONS else ""
            )
            context = (f"{intent}\n\nPARENT FAMILY {family_label(parent.family)}:\n"
                f"{self.family.summary(parent.family)}\n"
                f"DONOR FAMILY {family_label(donor.family)}:\n"
                f"{self.family.summary(donor.family)}\n"
                "Use the family context to make a coherent algorithmic improvement.")
        prompt = (f"{context}\n\nPARENT (score={parent.score:.8f}):\n```python\n{parent.code}\n```\n"
            f"\nDONOR (score={donor.score:.8f}):\n```python\n{donor.code}\n```\n"
            "Generate one improved child now.")
        if self.validity is not None:
            prompt += self.validity.prompt_block(self.b, self.B)
        return prompt

    def generate(self, selection, phase, directory, seed):
        begin = time.monotonic()
        create = selection.search_action == CREATE_ACTION
        response = self.gen_client.chat.completions.create(model=self.gen_model, seed=seed,
            temperature=self.cfg["temperature"], max_tokens=self.cfg["max_tokens"], n=1,
            messages=[{"role": "system", "content": CREATE_TASK if create else TASK},
                      {"role": "user", "content": self.prompt(selection, phase)}])
        raw = response.choices[0].message.content or ""
        (directory / "generation.txt").write_text(raw, encoding="utf-8")
        # CREATE has no semantic parent. The canonical initial program is used only as a
        # neutral technical scaffold for the full-code patch extractor and is never shown
        # in the CREATE prompt.
        scaffold = self.create_scaffold_code if create else selection.parent.code
        code, applied, _, error, _, _ = apply_full_patch(raw, original_str=scaffold,
            patch_dir=directory / "patch", language="python", verbose=False)
        if not applied:
            code = f"raise RuntimeError({('candidate patch failure: ' + str(error))!r})\n"
        return code, time.monotonic() - begin

    def maybe_end_warmup(self):
        if self.warmup_done:
            return
        reached = len(self.family.families) >= self.args.k_min
        if reached or self.b >= self.args.warmup_budget:
            self.warmup_done = True
            self.warmup_end_b = self.b
            self.warmup_end_sec = time.monotonic() - self.start
            self.warmup_reached_k = reached
            if self.args.condition in ("warmup", "warmup_validity"):
                self.family.freeze()

    def run(self):
        initial_path = REPO / "examples/circle_packing/initial.py"
        initial_dir = self.run_dir / "initial"
        initial_dir.mkdir()
        valid, score, _ = self.evaluate(initial_path, initial_dir)
        if not valid or score is None:
            raise RuntimeError("canonical initial Circle Packing program is invalid")
        initial = Program("P0", initial_path.read_text(encoding="utf-8"), score)
        self.create_scaffold_code = initial.code
        self.programs["P0"] = initial
        self.best, self.initial_score = initial, score
        if self.args.condition in WARMUP_CONDITIONS:
            self.observe(initial, 0)  # explicit P0 -> F1, outside proposal budget
            self.initial_program_seeded = True

        for proposal_id in range(1, self.B + 1):
            self.maybe_end_warmup()
            phase = "normal" if self.warmup_done else "warmup"
            t = budget_fraction(self.b, self.B)
            selection = self.select(proposal_id, phase, t)
            parent, donor = selection.parent, selection.donor
            if parent is not None:
                parent.children += 1
            pf, df = selection.parent_family, selection.donor_family
            self.search_mass.add(pf)
            self.b += 1
            if selection.search_action == CREATE_ACTION:
                self.create_attempts += 1
            directory = self.run_dir / f"proposal_{proposal_id:04d}"
            directory.mkdir()
            generation_seed = role_seed(self.args.seed, proposal_id, "candidate_generation")
            validity_prompt_injected = bool(self.validity is not None and self.validity.active)
            if validity_prompt_injected:
                self.validity_prompt_count += 1
            code, generation_time = self.generate(selection, phase, directory, generation_seed)
            child_path = directory / "main.py"
            child_path.write_text(code, encoding="utf-8")
            child_valid, child_score, evaluation_time = self.evaluate(child_path, directory)
            if child_valid:
                self.valid_children += 1
                if selection.search_action == CREATE_ACTION:
                    self.create_valid += 1
            else:
                self.invalid_children += 1
            assignment = {"family": None, "created": None, "nearest_similarity": None,
                          "assignment_margin": None}
            summary_time = embedding_time = 0.0
            if child_valid and child_score is not None:
                child = Program(f"P{proposal_id}", code, child_score)
                self.programs[child.id] = child
                if child.score > self.best.score:
                    self.best, self.best_at = child, proposal_id
                if self.args.condition in ONLINE_CONDITIONS or phase == "warmup":
                    assignment, summary_time, embedding_time = self.observe(child, proposal_id)
                    if phase == "warmup": self.warmup_valid_children += 1
                    if assignment["created"]:
                        self.family_birth_proposals.append(proposal_id)
                        if selection.search_action == CREATE_ACTION:
                            self.create_new_family_successes += 1
            failure = {}
            controller_event = {"controller_check": False, "trigger_on_event": False,
                                "trigger_off_event": False}
            if self.validity is not None:
                failure = classify_failure(
                    child_valid,
                    error_msg=self.last_evaluation.get("error"),
                    timed_out=bool(self.last_evaluation.get("timed_out")),
                )
                self.validity.update(failure)
                controller_event = self.validity.check(proposal_id)
                self.validity_history.append({
                    "proposal_id": proposal_id,
                    "valid": bool(child_valid),
                    "prompt_injected": validity_prompt_injected,
                    "trigger_on_event": controller_event["trigger_on_event"],
                    "trigger_off_event": controller_event["trigger_off_event"],
                })
            self.maybe_end_warmup()
            K, ent = len(self.family.families), self.search_mass.metrics(len(self.family.families))
            self.events.write({"proposal_id": proposal_id, "replicate": self.args.replicate,
                "seed": self.args.seed, "condition": self.args.condition, "phase": phase, "t": t,
                "wall_clock_sec": time.monotonic()-self.start,
                "search_action": selection.search_action,
                "mutation_intent": selection.mutation_intent,
                "create_probability": selection.create_probability,
                "selected_parent_family": (
                    family_label(pf) if pf is not None else None
                ),
                "selected_donor_family": (
                    family_label(df) if df is not None else None
                ),
                "parent_program_id": parent.id if parent is not None else None,
                "donor_program_id": donor.id if donor is not None else None,
                "parent_family": pf, "donor_family": df,
                "cross_family_donor": (pf != df) if pf is not None and df is not None else None,
                "K": K, "H_search": ent["H"], "H_population": self.family.population_entropy(),
                "H_raw": ent["H_raw"], "N_eff": ent["N_eff"],
                "top_family_search_mass": ent["top_mass"], "child_valid": child_valid,
                "child_score": child_score, "best_score_so_far": self.best.score,
                "child_family": assignment["family"], "created_new_family": assignment["created"],
                "nearest_family_similarity": assignment["nearest_similarity"],
                "assignment_margin": assignment["assignment_margin"],
                "generation_seed": generation_seed, "generation_time_sec": generation_time,
                "evaluation_time_sec": evaluation_time, "summarization_time_sec": summary_time,
                "embedding_time_sec": embedding_time,
                "failure_classes": list(failure),
                "validity_active": self.validity.active if self.validity is not None else None,
                "active_constraint": (
                    self.validity.active_constraint()
                    if self.validity is not None and self.validity.active else None
                ),
                "validity_prompt_injected": (
                    validity_prompt_injected if self.validity is not None else None
                ),
                "r_by_constraint": dict(self.validity.r) if self.validity is not None else None,
                "lambda_by_constraint": (
                    dict(self.validity.lam) if self.validity is not None else None
                ),
                "controller_check": (
                    controller_event["controller_check"] if self.validity is not None else None
                ),
                "trigger_on_event": (
                    controller_event["trigger_on_event"] if self.validity is not None else None
                ),
                "trigger_off_event": (
                    controller_event["trigger_off_event"] if self.validity is not None else None
                ),
                "representative_witnesses": (
                    self.validity.top_witnesses()
                    if self.validity is not None and self.validity.active else []
                )})

        ent, margins = self.search_mass.metrics(len(self.family.families)), self.family.margins
        summary = {"condition": self.args.condition, "replicate": self.args.replicate,
            "seed": self.args.seed, "initial_score": self.initial_score,
            "final_best_score": self.best.score, "absolute_improvement": self.best.score-self.initial_score,
            "relative_improvement": (self.best.score-self.initial_score)/self.initial_score,
            "proposal_where_best_found": self.best_at, "final_K": len(self.family.families),
            "final_H_search": ent["H"], "final_N_eff": ent["N_eff"],
            "number_of_family_births": self.family.child_created_births(
                self.initial_program_seeded),
            "family_birth_proposal_ids": self.family_birth_proposals,
            "create_attempts": self.create_attempts,
            "create_valid": self.create_valid,
            "create_new_family_successes": self.create_new_family_successes,
            "create_success_rate": (
                self.create_new_family_successes / self.create_attempts
                if self.create_attempts else None
            ),
            "warmup_proposal_count": self.warmup_end_b,
            "warmup_wall_clock_duration": self.warmup_end_sec,
            "warmup_reached_K_min_before_B_warm": self.warmup_reached_k and self.warmup_end_b < self.args.warmup_budget,
            "valid_children_required_to_reach_K_min": self.warmup_valid_children if self.warmup_reached_k else None,
            "mean_assignment_margin": float(np.mean(margins)) if margins else None,
            "median_assignment_margin": float(np.median(margins)) if margins else None,
            "fraction_assignment_margin_le_zero": float(np.mean(np.asarray(margins)<=0)) if margins else None,
            **cosine_diagnostics(self.family)}
        if self.args.condition == "warmup_validity":
            def window_metrics(records):
                count = len(records)
                valid_count = sum(item["valid"] for item in records)
                return {
                    "proposals": count,
                    "valid": valid_count,
                    "invalid": count - valid_count,
                    "validity_rate": valid_count / count if count else None,
                    "enough_samples_for_interpretation": count >= 10,
                }

            episodes = [dict(episode) for episode in (self.validity.episodes if self.validity else [])]
            first_prompt = min(
                (episode["first_prompt_proposal"] for episode in episodes), default=None
            )
            releases = [
                episode["end_proposal"] for episode in episodes
                if episode["end_proposal"] is not None
            ]
            before = [
                item for item in self.validity_history
                if first_prompt is None or item["proposal_id"] < first_prompt
            ]
            during = [item for item in self.validity_history if item["prompt_injected"]]
            after_release = [
                item for item in self.validity_history
                if releases and item["proposal_id"] > min(releases)
                and not item["prompt_injected"]
            ]
            summary.update({
                "proposal_budget": self.B,
                "valid_proposals": self.valid_children,
                "invalid_proposals": self.invalid_children,
                "validity_rate": self.valid_children / self.B,
                "validity": {
                    "enabled": self.validity is not None,
                    "failure_count_by_class": {
                        key: int(self.validity.failure_counts[key])
                        for key in self.validity.classes
                    } if self.validity else {},
                    "activation_episode_count": len(episodes),
                    "activation_episodes": episodes,
                    "validity_prompt_injection_fraction": self.validity_prompt_count / self.B,
                    "final_r_by_constraint": dict(self.validity.r) if self.validity else {},
                    "final_lambda_by_constraint": dict(self.validity.lam) if self.validity else {},
                    "maximum_lambda_by_constraint": (
                        dict(self.validity.max_lam) if self.validity else {}
                    ),
                    "validity_before_first_steering_activation": window_metrics(before),
                    "validity_during_steering": window_metrics(during),
                    "validity_after_steering_release": window_metrics(after_release),
                },
            })
        exported_families = self.family.export()
        summary["family_representatives"] = [
            {"family": f["id"], "family_label": family_label(f["id"]),
             "summary": f["representative_summary"],
             "program_id": f["representative_program_id"]}
            for f in exported_families
        ]
        write_json(self.run_dir / "families.json", exported_families)
        write_json(self.run_dir / "run_summary.json", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--replicate", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--proposals", type=int, default=200)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--warmup-budget", type=int, default=40)
    p.add_argument("--entropy-window", type=int, default=25)
    p.add_argument("--family-similarity-threshold", type=float, default=0.85)
    p.add_argument("--evaluator-timeout-sec", type=int, default=300)
    p.add_argument("--run-name")
    args = p.parse_args()
    with (ROOT / "base.yaml").open(encoding="utf-8") as stream: cfg = yaml.safe_load(stream)
    Experiment(args, cfg).run()


if __name__ == "__main__": main()
