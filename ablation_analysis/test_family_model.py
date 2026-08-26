import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from family_model import (
    CONDITIONS, DONOR_CONDITIONS, ONLINE_CONDITIONS, PARENT_CONDITIONS,
    PROMPT_CONDITIONS, WARMUP_CONDITIONS, EventWriter, FamilyIndex, SearchMass,
    family_label, role_seed,
)
from representation import (SUMMARY_FIELDS, build_warmup_context, parse_summary,
    serialize_summary)


def detailed_summary(**overrides):
    values = {
        "broad_strategy": "Segment Tree",
        "state_representation": "A fixed array-backed binary tree of aggregate values",
        "structural_decomposition": "Static recursive decomposition of a fixed index interval",
        "core_computational_mechanism": "Combine child aggregates along root-to-leaf paths",
        "candidate_generation_or_transition": "Point updates rewrite one leaf and its ancestors",
        "selection_and_update": "Queries merge aggregates from covered canonical intervals",
        "constraint_handling": "Valid query and update bounds are assumed",
        "refinement_or_optimization": "None",
        "scheduling_or_adaptation": "None",
        "distinctive_mechanistic_details": "Static topology with eager point updates",
        "non_distinguishing_details": "Array indices and loop syntax",
    }
    values.update(overrides)
    return serialize_summary(values)


class FamilyModelTests(unittest.TestCase):
    def test_exact_seven_condition_factorization(self):
        self.assertEqual(CONDITIONS, (
            "baseline", "observe", "warmup", "parent", "inspiration", "prompt", "full"))
        self.assertEqual(WARMUP_CONDITIONS, {"warmup", "parent", "inspiration", "prompt", "full"})
        self.assertEqual(ONLINE_CONDITIONS, {"observe", "parent", "inspiration", "prompt", "full"})
        self.assertEqual(PARENT_CONDITIONS, {"parent", "full"})
        self.assertEqual(DONOR_CONDITIONS, {"inspiration", "full"})
        self.assertEqual(PROMPT_CONDITIONS, {"prompt", "full"})

    def test_role_seeds_are_stable_and_separated(self):
        self.assertEqual(role_seed(7, 11, "parent_sampling"), role_seed(7, 11, "parent_sampling"))
        self.assertNotEqual(role_seed(7, 11, "parent_sampling"), role_seed(7, 11, "donor_sampling"))
        self.assertNotEqual(role_seed(7, 11, "candidate_generation"), role_seed(7, 12, "candidate_generation"))

    def test_search_entropy_uses_window_occupancy_and_current_K(self):
        mass = SearchMass(4)
        mass.add(0); mass.add(0); mass.add(1)
        got = mass.metrics(K=3)
        raw = -(2/3) * math.log(2/3) - (1/3) * math.log(1/3)
        self.assertAlmostEqual(got["H_raw"], raw)
        self.assertAlmostEqual(got["H"], raw / math.log(3))
        self.assertAlmostEqual(got["N_eff"], math.exp(raw))
        self.assertAlmostEqual(got["top_mass"], 2/3)
        mass.add(2); mass.add(2)
        self.assertEqual(list(mass.events), [0, 1, 2, 2])

    def test_population_entropy_is_separate(self):
        index = FamilyIndex(0.8)
        index.assign("a", 1, "a", [1, 0])
        index.assign("b", 2, "b", [1, 0])
        index.assign("c", 3, "c", [0, 1])
        raw = -(2/3) * math.log(2/3) - (1/3) * math.log(1/3)
        self.assertAlmostEqual(index.population_entropy(), raw / math.log(2))

    def test_family_quality_is_median_and_pi_matches_formula(self):
        index = FamilyIndex(0.8)
        index.assign("a", 1, "a", [1, 0])
        index.assign("b", 9, "b", [1, 0])
        index.assign("c", 3, "c", [1, 0])
        index.assign("d", 2, "d", [0, 1])
        self.assertEqual(index.families[0].quality(), 3)
        np.testing.assert_allclose(index.probabilities(0), [0.5, 0.5])
        expected = np.exp([3, 2]) / np.exp([3, 2]).sum()
        np.testing.assert_allclose(index.probabilities(1), expected)

    def test_human_labels_are_one_based_and_seed_is_not_a_child_birth(self):
        index = FamilyIndex(0.8)
        index.assign("P0", 1, "seed", [1, 0])
        self.assertEqual(family_label(0), "F1")
        self.assertEqual(index.export()[0]["label"], "F1")
        self.assertEqual(index.child_created_births(initial_program_seeded=True), 0)
        index.assign("P1", 2, "different", [0, 1])
        self.assertEqual(index.child_created_births(initial_program_seeded=True), 1)
        self.assertEqual(family_label(1), "F2")

    def test_event_writer_requires_common_schema_and_writes_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            writer = EventWriter(path)
            rec = {field: None for field in writer.FIELDS}
            rec.update(proposal_id=1, condition="baseline")
            writer.write(rec)
            got = json.loads(path.read_text())
            self.assertEqual(set(got), set(writer.FIELDS))
            self.assertIsNone(got["parent_family"])
            with self.assertRaises(ValueError):
                writer.write({"proposal_id": 2})


class RepresentationTests(unittest.TestCase):
    def test_broad_name_does_not_erase_segment_tree_variants(self):
        summaries = {
            "static": detailed_summary(),
            "lazy": detailed_summary(
                state_representation="Array tree plus pending range-update tags per node",
                candidate_generation_or_transition="Range updates attach deferred tags to covered nodes",
                selection_and_update="Push lazy tags before descending and recompute ancestors",
                scheduling_or_adaptation="Deferred propagation is triggered only on descent",
                distinctive_mechanistic_details="Lazy range updates with push-down tags",
            ),
            "dynamic": detailed_summary(
                state_representation="Pointer nodes allocated only for visited value intervals",
                structural_decomposition="Dynamic recursive interval decomposition with sparse topology",
                candidate_generation_or_transition="Updates allocate missing nodes along a path",
                selection_and_update="Mutate aggregates in the sparse nodes on that path",
                scheduling_or_adaptation="Nodes are created on demand",
                distinctive_mechanistic_details="Sparse dynamic allocation over a large domain",
            ),
            "persistent": detailed_summary(
                state_representation="Immutable version roots sharing unchanged tree nodes",
                structural_decomposition="Static interval hierarchy with a separate root per version",
                candidate_generation_or_transition="Path-copy updated nodes to create a new version",
                selection_and_update="Retain old roots and query any version without mutation",
                distinctive_mechanistic_details="Persistent path copying and structural sharing",
            ),
        }
        parsed = {name: parse_summary(text) for name, text in summaries.items()}
        self.assertEqual({item["broad_strategy"] for item in parsed.values()}, {"Segment Tree"})
        signatures = {
            (item["state_representation"], item["structural_decomposition"],
             item["candidate_generation_or_transition"], item["selection_and_update"])
            for item in parsed.values()
        }
        self.assertEqual(len(signatures), 4)

    def test_parameter_only_ring_variants_keep_same_core_mechanism(self):
        common = {
            "broad_strategy": "Deterministic constructive geometric packing",
            "state_representation": "Direct arrays of circle centers and radii",
            "structural_decomposition": "Fixed center circle, inner ring, and outer ring topology",
            "core_computational_mechanism": "Place centers at predetermined ring angles",
            "candidate_generation_or_transition": "Deterministically construct one fixed-ring packing",
            "selection_and_update": "No alternative candidate selection",
            "constraint_handling": "Keep centers fixed and proportionally shrink overlapping radii",
            "refinement_or_optimization": "None",
            "scheduling_or_adaptation": "None",
            "distinctive_mechanistic_details": "Fixed topology and fixed-center radius repair",
        }
        first = parse_summary(serialize_summary({**common,
            "non_distinguishing_details": "Inner radius 0.25 and outer radius 0.50"}))
        second = parse_summary(serialize_summary({**common,
            "non_distinguishing_details": "Inner radius 0.20 and outer radius 0.55"}))
        for field in SUMMARY_FIELDS[1:10]:
            self.assertEqual(first[field], second[field])

    def test_adaptive_placement_exposes_mechanism_changes(self):
        fixed = parse_summary(detailed_summary(
            broad_strategy="Constructive packing",
            structural_decomposition="Fixed concentric-ring topology",
            candidate_generation_or_transition="Construct one packing at predetermined angles",
        ))
        adaptive = parse_summary(detailed_summary(
            broad_strategy="Constructive packing",
            structural_decomposition="Adaptive decomposition of currently open geometric gaps",
            candidate_generation_or_transition="Repeatedly select a gap and insert a new circle from current geometry",
        ))
        self.assertNotEqual(fixed["structural_decomposition"], adaptive["structural_decomposition"])
        self.assertNotEqual(
            fixed["candidate_generation_or_transition"],
            adaptive["candidate_generation_or_transition"],
        )

    def test_warmup_prompt_contains_full_contrast_without_algorithm_menu(self):
        families = [
            {"id": 0, "summary": detailed_summary()},
            {"id": 1, "summary": detailed_summary(
                state_representation="Tree nodes with lazy tags",
                distinctive_mechanistic_details="Deferred range propagation",
            )},
        ]
        prompt = build_warmup_context(families)
        self.assertIn("FAMILY F1", prompt)
        self.assertIn("FAMILY F2", prompt)
        for field in SUMMARY_FIELDS:
            self.assertGreaterEqual(prompt.count(f"{field}:"), 2)
        self.assertIn("fixes bugs", prompt)
        self.assertIn("changes constants", prompt)
        self.assertIn("at least TWO", prompt)
        for forbidden in ("simulated annealing", "genetic algorithm", "SLSQP"):
            self.assertNotIn(forbidden, prompt)


if __name__ == "__main__":
    unittest.main()
