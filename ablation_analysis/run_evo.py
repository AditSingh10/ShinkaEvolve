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

from family_model import (CONDITIONS, DONOR_CONDITIONS, EventWriter, FamilyIndex,
    ONLINE_CONDITIONS, PARENT_CONDITIONS, PROMPT_CONDITIONS, SearchMass,
    WARMUP_CONDITIONS, cosine_diagnostics, role_seed, write_json)
from representation import (SUMMARY_SYSTEM, build_warmup_context, family_label,
    serialize_summary)
from shinka.edit.apply_full import apply_full_patch
from shinka.embed import EmbeddingClient
from shinka.llm.client import get_client_llm

TASK = """You are an expert mathematician and computational geometer. Improve the supplied Python
algorithm for packing 26 non-overlapping circles inside a unit square, maximizing the sum of radii.
Return exactly one complete Python code block. Preserve the EVOLVE-BLOCK markers and fixed
run_packing interface. Use only packages already available to the supplied program."""
@dataclass
class Program:
    id: str
    code: str
    score: float
    family: Optional[int] = None
    children: int = 0


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
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.gen_client, self.gen_model, _ = get_client_llm(cfg["generation_model"])
        if hasattr(self.gen_client, "with_options"):
            self.gen_client = self.gen_client.with_options(max_retries=0)
        self.embed_client = EmbeddingClient(model_name=cfg["embedding_model"])

    def evaluate(self, path: Path, directory: Path) -> Tuple[bool, Optional[float], float]:
        results = directory / "evaluation"
        results.mkdir(parents=True, exist_ok=True)
        begin = time.monotonic()
        cmd = [sys.executable, str(ROOT / "evaluator_entry.py"),
               "--program_path", str(path), "--results_dir", str(results)]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True,
                timeout=self.args.evaluator_timeout_sec, check=False)
            (directory / "evaluator.stdout").write_text(done.stdout, encoding="utf-8")
            (directory / "evaluator.stderr").write_text(done.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            (directory / "evaluator.stderr").write_text(f"timeout: {exc}\n", encoding="utf-8")
        elapsed = time.monotonic() - begin
        try:
            valid = json.loads((results / "correct.json").read_text())["correct"]
            score = json.loads((results / "metrics.json").read_text()).get("combined_score")
            return bool(valid), float(score) if score is not None else None, elapsed
        except Exception:
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
        scores = np.asarray([p.score for p in pool], dtype=float)
        median = float(np.median(scores))
        mad = max(float(np.median(np.abs(scores - median))), 1e-6)
        z = 10.0 * (scores - median) / mad
        sigmoid = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        weights = sigmoid * np.asarray([1.0 / (1 + p.children) for p in pool])
        return pool[int(rng.choice(len(pool), p=weights / weights.sum()))]

    def member_program(self, family, rng, exclude=None):
        ids = [x for x in self.family.member_ids(family) if x != exclude]
        if not ids:
            return self.baseline_program(rng, exclude)
        pool = sorted((self.programs[x] for x in ids), key=lambda p: p.score, reverse=True)
        weights = np.asarray([(i + 1) ** -1.0 for i in range(len(pool))])
        return pool[int(rng.choice(len(pool), p=weights / weights.sum()))]

    def select(self, proposal_id, phase, t):
        frng = np.random.RandomState(role_seed(self.args.seed, proposal_id, "family_sampling"))
        prng = np.random.RandomState(role_seed(self.args.seed, proposal_id, "parent_sampling"))
        drng = np.random.RandomState(role_seed(self.args.seed, proposal_id, "donor_sampling"))
        if phase == "normal" and self.args.condition in PARENT_CONDITIONS and self.family.families:
            parent = self.member_program(self.family.sample_family(t, frng), prng)
        else:
            parent = self.baseline_program(prng)
        if phase == "normal" and self.args.condition in DONOR_CONDITIONS and self.family.families:
            donor = self.member_program(self.family.sample_family(t, drng), drng, parent.id)
        else:
            donor = self.baseline_program(drng, parent.id)
        return parent, donor

    def prompt(self, parent, donor, phase):
        context = ""
        if phase == "warmup":
            context = build_warmup_context(self.family.families)
        elif self.args.condition in PROMPT_CONDITIONS:
            context = (f"PARENT FAMILY {family_label(parent.family)}:\n"
                f"{self.family.summary(parent.family)}\n"
                f"DONOR FAMILY {family_label(donor.family)}:\n"
                f"{self.family.summary(donor.family)}\n"
                "Use the family context to make a coherent algorithmic improvement.")
        return (f"{context}\n\nPARENT (score={parent.score:.8f}):\n```python\n{parent.code}\n```\n"
            f"\nDONOR (score={donor.score:.8f}):\n```python\n{donor.code}\n```\n"
            "Generate one improved child now.")

    def generate(self, parent, donor, phase, directory, seed):
        begin = time.monotonic()
        response = self.gen_client.chat.completions.create(model=self.gen_model, seed=seed,
            temperature=self.cfg["temperature"], max_tokens=self.cfg["max_tokens"], n=1,
            messages=[{"role": "system", "content": TASK},
                      {"role": "user", "content": self.prompt(parent, donor, phase)}])
        raw = response.choices[0].message.content or ""
        (directory / "generation.txt").write_text(raw, encoding="utf-8")
        code, applied, _, error, _, _ = apply_full_patch(raw, original_str=parent.code,
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
            if self.args.condition == "warmup":
                self.family.freeze()

    def run(self):
        initial_path = REPO / "examples/circle_packing/initial.py"
        initial_dir = self.run_dir / "initial"
        initial_dir.mkdir()
        valid, score, _ = self.evaluate(initial_path, initial_dir)
        if not valid or score is None:
            raise RuntimeError("canonical initial Circle Packing program is invalid")
        initial = Program("P0", initial_path.read_text(encoding="utf-8"), score)
        self.programs["P0"] = initial
        self.best, self.initial_score = initial, score
        if self.args.condition in WARMUP_CONDITIONS:
            self.observe(initial, 0)  # explicit P0 -> F1, outside proposal budget
            self.initial_program_seeded = True

        for proposal_id in range(1, self.B + 1):
            self.maybe_end_warmup()
            phase = "normal" if self.warmup_done else "warmup"
            t = proposal_id / self.B
            parent, donor = self.select(proposal_id, phase, t)
            parent.children += 1
            pf, df = parent.family, donor.family
            self.search_mass.add(pf)
            self.b += 1
            directory = self.run_dir / f"proposal_{proposal_id:04d}"
            directory.mkdir()
            generation_seed = role_seed(self.args.seed, proposal_id, "candidate_generation")
            code, generation_time = self.generate(parent, donor, phase, directory, generation_seed)
            child_path = directory / "main.py"
            child_path.write_text(code, encoding="utf-8")
            child_valid, child_score, evaluation_time = self.evaluate(child_path, directory)
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
            self.maybe_end_warmup()
            K, ent = len(self.family.families), self.search_mass.metrics(len(self.family.families))
            self.events.write({"proposal_id": proposal_id, "replicate": self.args.replicate,
                "seed": self.args.seed, "condition": self.args.condition, "phase": phase, "t": t,
                "wall_clock_sec": time.monotonic()-self.start, "parent_program_id": parent.id,
                "donor_program_id": donor.id, "parent_family": pf, "donor_family": df,
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
                "embedding_time_sec": embedding_time})

        ent, margins = self.search_mass.metrics(len(self.family.families)), self.family.margins
        summary = {"condition": self.args.condition, "replicate": self.args.replicate,
            "seed": self.args.seed, "initial_score": self.initial_score,
            "final_best_score": self.best.score, "absolute_improvement": self.best.score-self.initial_score,
            "relative_improvement": (self.best.score-self.initial_score)/self.initial_score,
            "proposal_where_best_found": self.best_at, "final_K": len(self.family.families),
            "final_H_search": ent["H"], "final_N_eff": ent["N_eff"],
            "number_of_family_births": self.family.child_created_births(
                self.initial_program_seeded),
            "warmup_proposal_count": self.warmup_end_b,
            "warmup_wall_clock_duration": self.warmup_end_sec,
            "warmup_reached_K_min_before_B_warm": self.warmup_reached_k and self.warmup_end_b < self.args.warmup_budget,
            "valid_children_required_to_reach_K_min": self.warmup_valid_children if self.warmup_reached_k else None,
            "mean_assignment_margin": float(np.mean(margins)) if margins else None,
            "median_assignment_margin": float(np.median(margins)) if margins else None,
            "fraction_assignment_margin_le_zero": float(np.mean(np.asarray(margins)<=0)) if margins else None,
            **cosine_diagnostics(self.family)}
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
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--warmup-budget", type=int, default=40)
    p.add_argument("--entropy-window", type=int, default=25)
    p.add_argument("--family-similarity-threshold", type=float, default=0.85)
    p.add_argument("--evaluator-timeout-sec", type=int, default=300)
    args = p.parse_args()
    with (ROOT / "base.yaml").open(encoding="utf-8") as stream: cfg = yaml.safe_load(stream)
    Experiment(args, cfg).run()


if __name__ == "__main__": main()
