#!/usr/bin/env python3
"""Run only the bounded Family Model seeding warmup; never enter normal search."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from family_model import cosine_diagnostics, family_label, role_seed, write_json
from representation import differing_major_dimensions, parse_summary
from run_evo import Experiment, Program, REPO, ROOT


def pairwise_summary_cosines(experiment: Experiment):
    embeddings = [member.embedding for family in experiment.family.families for member in family.members]
    if len(embeddings) < 2:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    array = np.asarray(embeddings, dtype=float)
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    values = (array @ array.T)[np.triu_indices(len(array), k=1)]
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def run_probe(args) -> Path:
    with (ROOT / "base.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    experiment = Experiment(args, config)
    initial_path = REPO / "examples/circle_packing/initial.py"
    initial_dir = experiment.run_dir / "initial"
    initial_dir.mkdir()
    valid, score, _ = experiment.evaluate(initial_path, initial_dir)
    if not valid or score is None:
        raise RuntimeError("canonical initial Circle Packing program is invalid")
    initial = Program("P0", initial_path.read_text(encoding="utf-8"), score)
    experiment.programs[initial.id] = initial
    experiment.best = initial
    experiment.initial_score = score
    initial_assignment, _, _ = experiment.observe(initial, 0)
    experiment.initial_program_seeded = True

    probe_path = experiment.run_dir / "probe_events.jsonl"
    valid_count = 0
    all_hashes, valid_hashes = [], []
    birth_proposals = {}

    for proposal_id in range(1, args.warmup_budget + 1):
        t = proposal_id / args.warmup_budget
        parent, donor = experiment.select(proposal_id, "warmup", t)
        parent.children += 1
        parent_family, donor_family = parent.family, donor.family
        experiment.search_mass.add(parent_family)
        experiment.b += 1

        directory = experiment.run_dir / f"proposal_{proposal_id:04d}"
        directory.mkdir()
        generation_seed = role_seed(args.seed, proposal_id, "candidate_generation")
        code, generation_time = experiment.generate(
            parent, donor, "warmup", directory, generation_seed
        )
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        all_hashes.append(code_hash)
        child_path = directory / "main.py"
        child_path.write_text(code, encoding="utf-8")
        child_valid, child_score, evaluation_time = experiment.evaluate(child_path, directory)

        assignment = {
            "family": None,
            "created": None,
            "nearest_family": None,
            "nearest_similarity": None,
            "assignment_margin": None,
            "structured_summary": None,
        }
        summary_time = embedding_time = 0.0
        major_differences = None
        if child_valid and child_score is not None:
            valid_count += 1
            valid_hashes.append(code_hash)
            child = Program(f"P{proposal_id}", code, child_score)
            experiment.programs[child.id] = child
            if child.score > experiment.best.score:
                experiment.best, experiment.best_at = child, proposal_id
            family_summaries_before = {
                family.id: family.summary for family in experiment.family.families
            }
            assignment, summary_time, embedding_time = experiment.observe(child, proposal_id)
            nearest_family = assignment["nearest_family"]
            if nearest_family is not None:
                major_differences = differing_major_dimensions(
                    assignment["structured_summary"], family_summaries_before[nearest_family]
                )
            if assignment["created"]:
                birth_proposals[family_label(assignment["family"])] = proposal_id

        family_count = len(experiment.family.families)
        entropy = experiment.search_mass.metrics(family_count)
        common_event = {
            "proposal_id": proposal_id,
            "replicate": args.replicate,
            "seed": args.seed,
            "condition": "warmup_probe",
            "phase": "warmup",
            "t": t,
            "wall_clock_sec": time.monotonic() - experiment.start,
            "parent_program_id": parent.id,
            "donor_program_id": donor.id,
            "parent_family": parent_family,
            "donor_family": donor_family,
            "cross_family_donor": (
                parent_family != donor_family
                if parent_family is not None and donor_family is not None else None
            ),
            "K": family_count,
            "H_search": entropy["H"],
            "H_population": experiment.family.population_entropy(),
            "H_raw": entropy["H_raw"],
            "N_eff": entropy["N_eff"],
            "top_family_search_mass": entropy["top_mass"],
            "child_valid": child_valid,
            "child_score": child_score,
            "best_score_so_far": experiment.best.score,
            "child_family": assignment["family"],
            "created_new_family": assignment["created"],
            "nearest_family_similarity": assignment["nearest_similarity"],
            "assignment_margin": assignment["assignment_margin"],
            "generation_seed": generation_seed,
            "generation_time_sec": generation_time,
            "evaluation_time_sec": evaluation_time,
            "summarization_time_sec": summary_time,
            "embedding_time_sec": embedding_time,
        }
        experiment.events.write(common_event)

        fields = parse_summary(assignment["structured_summary"] or "")
        probe_event = {
            **common_event,
            "code_hash": code_hash,
            "assigned_family_label": family_label(assignment["family"]),
            "nearest_family_label": family_label(assignment["nearest_family"]),
            "broad_strategy": (
                fields["broad_strategy"] if assignment["structured_summary"] else None
            ),
            "concise_mechanism": (
                fields["core_computational_mechanism"] + "; "
                + fields["distinctive_mechanistic_details"]
                if assignment["structured_summary"] else None
            ),
            "major_dimensions_different_from_nearest": major_differences,
            "structured_summary": assignment["structured_summary"],
        }
        with probe_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(probe_event) + "\n")
        print(json.dumps({key: probe_event[key] for key in (
            "proposal_id", "child_valid", "child_score", "code_hash",
            "assigned_family_label", "created_new_family", "nearest_family_similarity",
            "broad_strategy", "concise_mechanism",
            "major_dimensions_different_from_nearest",
        )}), flush=True)

        if family_count >= args.k_min:
            break

    exported = experiment.family.export()
    representatives = [
        {
            "family": family["label"],
            "program_id": family["representative_program_id"],
            "summary": family["representative_summary"],
        }
        for family in exported
    ]
    summary = {
        "seed": args.seed,
        "tau_fam": args.family_similarity_threshold,
        "K_min": args.k_min,
        "B_warm": args.warmup_budget,
        "proposals_consumed": experiment.b,
        "valid_proposals": valid_count,
        "unique_code_hashes_all_proposals": len(set(all_hashes)),
        "unique_code_hashes_valid_proposals": len(set(valid_hashes)),
        "child_created_family_births": experiment.family.child_created_births(True),
        "proposal_where_F2_created": birth_proposals.get("F2"),
        "proposal_where_F3_created": birth_proposals.get("F3"),
        "final_K": len(experiment.family.families),
        "reached_K_min": len(experiment.family.families) >= args.k_min,
        "pairwise_summary_cosine": pairwise_summary_cosines(experiment),
        "initial_family": {
            "family": family_label(initial_assignment["family"]),
            "program_id": "P0",
            "summary": initial_assignment["structured_summary"],
        },
        "family_representatives": representatives,
        **cosine_diagnostics(experiment.family),
    }
    write_json(experiment.run_dir / "families.json", exported)
    write_json(experiment.run_dir / "probe_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return experiment.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--warmup-budget", type=int, default=10)
    parser.add_argument("--family-similarity-threshold", type=float, default=0.85)
    parser.add_argument("--entropy-window", type=int, default=5)
    parser.add_argument("--evaluator-timeout-sec", type=int, default=300)
    args = parser.parse_args()
    args.condition = "full"
    args.proposals = args.warmup_budget
    args.run_name = f"warmup_probe_seed{args.seed}"
    run_probe(args)


if __name__ == "__main__":
    main()
