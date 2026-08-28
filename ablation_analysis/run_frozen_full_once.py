#!/usr/bin/env python3
"""Launch the frozen full_annealed run and export paper-ready raw data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from family_model import family_label, write_json
from run_evo import Experiment, REPO, ROOT


RUN_NAME = "run_full_annealed_B200_K3_W40_rep1_seed104729"
FROZEN_FILES = ("representation.py", "family_model.py", "run_evo.py", "base.yaml")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_args() -> argparse.Namespace:
    return argparse.Namespace(
        condition="full_annealed",
        replicate=1,
        seed=104729,
        proposals=200,
        k_min=3,
        warmup_budget=40,
        entropy_window=25,
        family_similarity_threshold=0.85,
        evaluator_timeout_sec=300,
        run_name=RUN_NAME,
    )


def export_post_run(experiment: Experiment) -> None:
    run_dir = experiment.run_dir
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    (run_dir / "initial" / "main.py").write_text(
        experiment.programs["P0"].code, encoding="utf-8"
    )

    raw_families = []
    for family in experiment.family.families:
        raw_families.append({
            "id": family.id,
            "label": family_label(family.id),
            "centroid": family.centroid.tolist(),
            "members": [{
                "program_id": member.program_id,
                "score": member.score,
                "summary": member.summary,
                "embedding": member.embedding.tolist(),
                "code": experiment.programs[member.program_id].code,
            } for member in family.members],
        })
    write_json(run_dir / "family_members_raw.json", {
        "problem": "Circle Packing",
        "embedding_dimension": (
            len(raw_families[0]["centroid"]) if raw_families else None
        ),
        "families": raw_families,
    })

    centroids = [np.asarray(family.centroid, dtype=float) for family in experiment.family.families]
    centroid_columns = [f"Cosine with {family_label(i)} centroid" for i in range(len(centroids))]
    fieldnames = [
        "Problem", "Family ID", "Program ID", "Score", "Summary",
        "Cosine with centroid of its own family", *centroid_columns,
        "Closest other family", "Cosine with closest other family",
        "Best program in family",
    ]
    representative_rows = []
    for family in experiment.family.families:
        representative = max(
            family.members, key=lambda member: float(np.dot(member.embedding, family.centroid))
        )
        similarities = [float(np.dot(representative.embedding, centroid)) for centroid in centroids]
        other_ids = [index for index in range(len(centroids)) if index != family.id]
        closest_other = max(other_ids, key=lambda index: similarities[index]) if other_ids else None
        row = {
            "Problem": "Circle Packing",
            "Family ID": family_label(family.id),
            "Program ID": representative.program_id,
            "Score": representative.score,
            "Summary": representative.summary,
            "Cosine with centroid of its own family": similarities[family.id],
            "Closest other family": family_label(closest_other),
            "Cosine with closest other family": (
                similarities[closest_other] if closest_other is not None else None
            ),
            "Best program in family": max(family.members, key=lambda member: member.score).program_id,
        }
        for index, column in enumerate(centroid_columns):
            row[column] = "" if index == family.id else similarities[index]
        representative_rows.append(row)
    with (run_dir / "family_representatives.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(representative_rows)

    trajectory_fields = (
        "proposal_id", "phase", "t", "K", "H_search", "H_raw", "N_eff",
        "top_family_search_mass", "parent_family", "donor_family",
        "cross_family_donor", "child_valid", "child_score", "best_score_so_far",
    )
    with (run_dir / "search_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows({field: event[field] for field in trajectory_fields} for event in events)

    births = {
        family_label(event["child_family"]): event["proposal_id"]
        for event in events if event["created_new_family"] is True
    }
    search_mass = Counter(
        family_label(event["parent_family"])
        for event in events if event["parent_family"] is not None
    )
    run_summary = json.loads((run_dir / "run_summary.json").read_text())
    analysis = {
        "problem": "Circle Packing",
        "condition": "full_annealed",
        "proposals": len(events),
        "warmup_proposals_used": run_summary["warmup_proposal_count"],
        "warmup_reached_K_min": any(
            event["phase"] == "warmup" and event["K"] >= experiment.args.k_min
            for event in events
        ),
        "family_birth_proposals": births,
        "final_K": len(experiment.family.families),
        "valid_proposals": sum(bool(event["child_valid"]) for event in events),
        "invalid_proposals": sum(not bool(event["child_valid"]) for event in events),
        "best_objective": run_summary["final_best_score"],
        "best_program_id": experiment.best.id,
        "search_mass_per_family": dict(search_mass),
        "members_per_family": {
            family_label(family.id): len(family.members) for family in experiment.family.families
        },
        "H_search_trajectory": [event["H_search"] for event in events],
        "N_eff_trajectory": [event["N_eff"] for event in events],
        "representative_table": representative_rows,
    }
    write_json(run_dir / "full_condition_analysis.json", analysis)


def main() -> None:
    args = fixed_args()
    with (ROOT / "base.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    experiment = Experiment(args, config)
    write_json(experiment.run_dir / "frozen_run_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "configuration": config,
        "source_sha256": {
            name: file_sha256(ROOT / name) for name in FROZEN_FILES
        },
        "initial_program": str(REPO / "examples/circle_packing/initial.py"),
        "initial_program_sha256": file_sha256(
            REPO / "examples/circle_packing/initial.py"
        ),
    })
    experiment.run()
    export_post_run(experiment)


if __name__ == "__main__":
    main()
