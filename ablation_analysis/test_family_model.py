import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from family_model import (
    ANNEALED_CONDITIONS, CONDITIONS, CREATE_ACTION, CREATE_CONDITIONS, DONOR_CONDITIONS,
    ONLINE_CONDITIONS, PARENT_CONDITIONS, PROMPT_CONDITIONS, WARMUP_CONDITIONS,
    EventWriter, FamilyIndex, SearchMass, budget_fraction, family_label, role_seed,
)
from representation import (SUMMARY_FIELDS, build_warmup_context, parse_summary,
    serialize_summary)
from run_evo import (Experiment, Program, SearchSelection, main as run_main,
    weighted_program_from_pool)


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


class CapturingRng:
    def __init__(self, result=0):
        self.result = result
        self.probabilities = None

    def choice(self, size, p):
        self.probabilities = np.asarray(p, dtype=float)
        return self.result


def sampling_experiment(condition="full_create"):
    experiment = Experiment.__new__(Experiment)
    experiment.args = SimpleNamespace(condition=condition, seed=104729)
    experiment.B = 200
    experiment.b = 0
    experiment.validity = None
    experiment.family = FamilyIndex(0.8)
    programs = (
        (Program("A0", "", 1.0, children=0), [1.0, 0.0]),
        (Program("A1", "", 2.0, children=1), [1.0, 0.0]),
        (Program("A2", "", 4.0, children=3), [1.0, 0.0]),
        (Program("B0", "", 0.5, children=2), [0.0, 1.0]),
        (Program("B1", "", 3.0, children=0), [0.0, 1.0]),
        (Program("B2", "", 5.0, children=1), [0.0, 1.0]),
    )
    experiment.programs = {program.id: program for program, _ in programs}
    for program, embedding in programs:
        assignment = experiment.family.assign(
            program.id, program.score, f"summary {program.id}", embedding
        )
        program.family = assignment["family"]
    return experiment


def probability_index():
    index = FamilyIndex(0.8)
    index.assign("A0", 1.0, "summary A0", [1.0, 0.0])
    index.assign("A1", 3.0, "summary A1", [1.0, 0.0])
    index.assign("B0", 4.0, "summary B0", [0.0, 1.0])
    return index


class FamilyModelTests(unittest.TestCase):
    def test_cli_accepts_explicit_run_name(self):
        argv = [
            "run_evo.py",
            "--condition", "full_create",
            "--replicate", "1",
            "--seed", "104729",
            "--run-name", "run_full_create_B200_K3_W40_rep1_seed104729",
        ]
        with patch("sys.argv", argv), patch("run_evo.Experiment") as experiment:
            run_main()
        args = experiment.call_args.args[0]
        self.assertEqual(args.condition, "full_create")
        self.assertEqual(
            args.run_name, "run_full_create_B200_K3_W40_rep1_seed104729"
        )
        experiment.return_value.run.assert_called_once_with()

    def test_condition_factorization_keeps_warmup_validity_prompt_only(self):
        self.assertEqual(CONDITIONS, (
            "baseline", "observe", "warmup", "parent", "inspiration", "prompt",
            "full_annealed", "full_create", "warmup_validity"))
        self.assertEqual(WARMUP_CONDITIONS, {
            "warmup", "parent", "inspiration", "prompt", "full_annealed",
            "full_create", "warmup_validity"})
        self.assertEqual(ONLINE_CONDITIONS, {
            "observe", "parent", "inspiration", "prompt", "full_annealed", "full_create"})
        self.assertEqual(PARENT_CONDITIONS, {"parent", "full_annealed", "full_create"})
        self.assertEqual(DONOR_CONDITIONS, {"inspiration", "full_annealed", "full_create"})
        self.assertEqual(PROMPT_CONDITIONS, {"prompt", "full_annealed", "full_create"})
        self.assertEqual(ANNEALED_CONDITIONS, {"full_annealed"})
        self.assertEqual(CREATE_CONDITIONS, {"full_create"})
        self.assertNotIn("warmup_validity", PARENT_CONDITIONS | DONOR_CONDITIONS | PROMPT_CONDITIONS)

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

    def test_create_actions_are_not_an_entropy_category(self):
        mass = SearchMass(10)
        for family in (0, None, 1, None, 0, 1):
            mass.add(family)
        self.assertEqual(list(mass.events), [0, 1, 0, 1])
        got = mass.metrics(K=2)
        self.assertAlmostEqual(got["H_raw"], math.log(2))
        self.assertAlmostEqual(got["H"], 1.0)
        self.assertAlmostEqual(got["N_eff"], 2.0)
        self.assertAlmostEqual(got["top_mass"], 0.5)

    def test_budget_fraction_is_completed_proposals_over_total_budget(self):
        self.assertEqual(budget_fraction(0, 200), 0.0)
        self.assertEqual(budget_fraction(40, 200), 0.2)
        self.assertEqual(budget_fraction(200, 200), 1.0)

    def test_population_entropy_is_separate(self):
        index = FamilyIndex(0.8)
        index.assign("a", 1, "a", [1, 0])
        index.assign("b", 2, "b", [1, 0])
        index.assign("c", 3, "c", [0, 1])
        raw = -(2/3) * math.log(2/3) - (1/3) * math.log(1/3)
        self.assertAlmostEqual(index.population_entropy(), raw / math.log(2))

    def test_family_quality_is_median(self):
        index = FamilyIndex(0.8)
        index.assign("a", 1, "a", [1, 0])
        index.assign("b", 9, "b", [1, 0])
        index.assign("c", 3, "c", [1, 0])
        index.assign("d", 2, "d", [0, 1])
        self.assertEqual(index.families[0].quality(), 3)

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


class FamilyActionProbabilityTests(unittest.TestCase):
    def test_action_probabilities_sum_exactly_to_one(self):
        index = probability_index()
        for t in (0.0, 0.1, 0.5, 0.9, 1.0):
            with self.subTest(t=t):
                self.assertEqual(float(index.create_probabilities(t).sum()), 1.0)

    def test_t_zero_is_uniform_over_families_and_create(self):
        np.testing.assert_allclose(
            probability_index().create_probabilities(0.0), [1/3, 1/3, 1/3]
        )

    def test_t_one_is_softmax_over_families_and_zero_create(self):
        probabilities = probability_index().create_probabilities(1.0)
        softmax = np.exp([2.0, 4.0]) / np.exp([2.0, 4.0]).sum()
        np.testing.assert_allclose(probabilities[:-1], softmax)
        self.assertEqual(probabilities[-1], 0.0)

    def test_intermediate_probabilities_match_analytical_formula(self):
        index = probability_index()
        t = 0.4
        softmax = np.exp([2.0, 4.0]) / np.exp([2.0, 4.0]).sum()
        exploration = (1.0 - t) / 3.0
        expected = np.concatenate((exploration + t * softmax, [exploration]))
        np.testing.assert_allclose(index.create_probabilities(t), expected)

    def test_donor_probabilities_are_conditional_existing_family_probabilities(self):
        index = probability_index()
        for t in (0.0, 0.4, 1.0):
            actions = index.create_probabilities(t)
            donors = index.create_donor_probabilities(t)
            self.assertEqual(float(donors.sum()), 1.0)
            np.testing.assert_allclose(donors, actions[:-1] / (1.0 - actions[-1]))


class AnnealedProbabilityTests(unittest.TestCase):
    def test_annealed_probabilities_match_original_policy(self):
        index = probability_index()
        softmax = np.exp([2.0, 4.0]) / np.exp([2.0, 4.0]).sum()
        for t in (0.0, 0.4, 1.0):
            with self.subTest(t=t):
                expected = (1.0 - t) / 2.0 + t * softmax
                probabilities = index.annealed_probabilities(t)
                self.assertEqual(float(probabilities.sum()), 1.0)
                np.testing.assert_allclose(probabilities, expected)

    def test_annealed_policy_has_no_create_component(self):
        index = probability_index()
        rng = CapturingRng(result=1)
        selected = index.sample_annealed_family(0.4, rng)
        self.assertEqual(selected, 1)
        self.assertEqual(len(rng.probabilities), len(index.families))


class FullPolicyIsolationTests(unittest.TestCase):
    def test_warmup_selection_and_prompt_are_identical(self):
        annealed = sampling_experiment("full_annealed")
        create = sampling_experiment("full_create")
        for proposal_id in range(1, 21):
            left = annealed.select(proposal_id, "warmup", proposal_id / 200)
            right = create.select(proposal_id, "warmup", proposal_id / 200)
            self.assertEqual(left.search_action, right.search_action)
            self.assertEqual(left.mutation_intent, right.mutation_intent)
            self.assertEqual(left.parent.id, right.parent.id)
            self.assertEqual(left.donor.id, right.donor.id)
            self.assertEqual(
                annealed.prompt(left, "warmup"),
                create.prompt(right, "warmup"),
            )

    def test_only_full_create_can_sample_create_in_normal_stage(self):
        annealed = sampling_experiment("full_annealed")
        with patch.object(
            annealed.family, "sample_create_action",
            side_effect=AssertionError("annealed policy reached CREATE sampler"),
        ), patch.object(annealed.family, "sample_annealed_family", return_value=0):
            annealed_selection = annealed.select(1, "normal", 0.4)
        self.assertEqual(annealed_selection.search_action, "F1")
        self.assertIsNone(annealed_selection.mutation_intent)
        self.assertEqual(annealed_selection.create_probability, 0.0)

        create = sampling_experiment("full_create")
        with patch.object(
            create.family, "sample_create_action", return_value=CREATE_ACTION
        ):
            create_selection = create.select(1, "normal", 0.4)
        self.assertEqual(create_selection.search_action, CREATE_ACTION)
        self.assertEqual(
            create_selection.create_probability,
            create.family.create_probabilities(0.4)[-1],
        )

    def test_existing_family_routing_differs_only_by_create_intent_prompt(self):
        annealed = sampling_experiment("full_annealed")
        create = sampling_experiment("full_create")
        with patch.object(
            annealed.family, "sample_annealed_family", return_value=0
        ), patch.object(
            create.family, "sample_create_action", return_value=0
        ), patch.object(
            create.family, "sample_create_donor_family", return_value=0
        ):
            annealed_selection = annealed.select(1, "normal", 0.4)
            create_selection = create.select(1, "normal", 0.4)
        self.assertEqual(annealed_selection.parent_family, create_selection.parent_family)
        self.assertEqual(annealed_selection.donor_family, create_selection.donor_family)
        self.assertEqual(annealed_selection.parent.id, create_selection.parent.id)
        self.assertEqual(annealed_selection.donor.id, create_selection.donor.id)
        self.assertIsNone(annealed_selection.mutation_intent)
        self.assertEqual(create_selection.mutation_intent, "REFINE")
        self.assertNotIn("SEARCH INTENT:", annealed.prompt(annealed_selection, "normal"))
        self.assertIn("SEARCH INTENT: REFINE", create.prompt(create_selection, "normal"))


class SearchActionTests(unittest.TestCase):
    def test_create_never_enters_donor_sampling_and_has_no_parent_or_donor(self):
        experiment = sampling_experiment()
        with patch.object(
            experiment.family, "sample_create_action", return_value=CREATE_ACTION
        ), patch.object(
            experiment.family, "sample_create_donor_family",
            side_effect=AssertionError("CREATE sampled donor")
        ), patch.object(
            experiment, "member_program", side_effect=AssertionError("CREATE sampled member")
        ):
            selection = experiment.select(1, "normal", 0.5)
        self.assertEqual(selection.search_action, CREATE_ACTION)
        self.assertEqual(selection.mutation_intent, CREATE_ACTION)
        self.assertIsNone(selection.parent)
        self.assertIsNone(selection.donor)
        self.assertIsNone(selection.parent_family)
        self.assertIsNone(selection.donor_family)

    def test_refine_when_parent_and_donor_families_match(self):
        experiment = sampling_experiment()
        with patch.object(experiment.family, "sample_create_action", return_value=0), patch.object(
            experiment.family, "sample_create_donor_family", return_value=0
        ):
            selection = experiment.select(1, "normal", 0.5)
        self.assertEqual(selection.search_action, "F1")
        self.assertEqual(selection.mutation_intent, "REFINE")
        self.assertEqual(selection.parent_family, selection.donor_family)

    def test_compose_when_parent_and_donor_families_differ(self):
        experiment = sampling_experiment()
        with patch.object(experiment.family, "sample_create_action", return_value=0), patch.object(
            experiment.family, "sample_create_donor_family", return_value=1
        ):
            selection = experiment.select(1, "normal", 0.5)
        self.assertEqual(selection.search_action, "F1")
        self.assertEqual(selection.mutation_intent, "COMPOSE")
        self.assertNotEqual(selection.parent_family, selection.donor_family)

    def test_create_prompt_uses_all_representative_summaries_without_scaffold(self):
        experiment = sampling_experiment()
        experiment.create_scaffold_code = "NEUTRAL CREATE SCAFFOLD"
        selection = SearchSelection(
            CREATE_ACTION, CREATE_ACTION, 0.2, None, None, None, None
        )
        prompt_text = experiment.prompt(selection, "normal")
        self.assertIn("SEARCH INTENT: CREATE", prompt_text)
        self.assertIn("CURRENT ALGORITHM FAMILIES", prompt_text)
        self.assertIn("summary A0", prompt_text)
        self.assertIn("summary B0", prompt_text)
        self.assertNotIn(experiment.create_scaffold_code, prompt_text)
        self.assertNotIn("PARENT (score=", prompt_text)
        self.assertNotIn("DONOR (score=", prompt_text)

    def test_create_prompt_uses_centroid_nearest_member_summary(self):
        experiment = Experiment.__new__(Experiment)
        experiment.family = FamilyIndex(-1.0)
        experiment.family.assign("old-left", 1.0, "OLD LEFT SUMMARY", [1.0, 0.0])
        experiment.family.assign(
            "centroid-nearest", 2.0, "CENTROID NEAREST SUMMARY", [2**-0.5, 2**-0.5]
        )
        experiment.family.assign("old-right", 3.0, "OLD RIGHT SUMMARY", [0.0, 1.0])

        representative = experiment.family.families[0].representative()
        self.assertEqual(representative.program_id, "centroid-nearest")
        self.assertEqual(experiment.family.families[0].summary, representative.summary)
        prompt_text = experiment.create_prompt()
        self.assertIn("CENTROID NEAREST SUMMARY", prompt_text)
        self.assertNotIn("OLD LEFT SUMMARY", prompt_text)
        self.assertNotIn("OLD RIGHT SUMMARY", prompt_text)

    def test_create_generation_uses_unshown_canonical_scaffold(self):
        experiment = sampling_experiment()
        experiment.create_scaffold_code = "NEUTRAL CREATE SCAFFOLD"
        experiment.cfg = {"temperature": 0.7, "max_tokens": 8192}
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="```python\nprint('candidate')\n```")
        )])
        create_call = Mock(return_value=response)
        experiment.gen_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_call))
        )
        experiment.gen_model = "test-model"
        selection = SearchSelection(
            CREATE_ACTION, CREATE_ACTION, 0.2, None, None, None, None
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "run_evo.apply_full_patch",
            return_value=("candidate code", True, None, None, None, None),
        ) as apply_patch_mock:
            experiment.generate(selection, "normal", Path(tmp), seed=7)
        self.assertEqual(
            apply_patch_mock.call_args.kwargs["original_str"],
            experiment.create_scaffold_code,
        )
        messages = create_call.call_args.kwargs["messages"]
        self.assertNotIn(experiment.create_scaffold_code, messages[1]["content"])
        self.assertIn("Develop a complete", messages[0]["content"])

    def test_refine_and_compose_prompts_route_explicit_intents(self):
        experiment = sampling_experiment()
        for donor_family, intent, required_fragments in (
            (0, "REFINE", (
                "same algorithm family",
                "parent as the main solution",
                "Preserve the effective core strategy",
                "meaningful algorithmic, implementation, or optimization improvement",
                "only when they support a substantive improvement",
            )),
            (1, "COMPOSE", (
                "different algorithm families",
                "Start from the parent",
                "complementary mechanisms from the donor",
                "do not copy the donor or replace the parent wholesale",
                "form a new hybrid family",
                "Do not explicitly optimize novelty",
                "Optimize the primary task objective",
                "avoid superficial parameter-only changes",
            )),
        ):
            with self.subTest(intent=intent), patch.object(
                experiment.family, "sample_create_action", return_value=0
            ), patch.object(
                experiment.family, "sample_create_donor_family", return_value=donor_family
            ):
                selection = experiment.select(1, "normal", 0.5)
                prompt_text = experiment.prompt(selection, "normal")
                self.assertEqual(selection.mutation_intent, intent)
                self.assertIn(f"SEARCH INTENT: {intent}", prompt_text)
                for required in required_fragments:
                    self.assertIn(required, prompt_text)

    def test_warmup_samples_families_uniformly_then_uses_weighted_family_pools(self):
        experiment = sampling_experiment()
        pools = []

        def choose_first(pool, rng):
            pool = list(pool)
            pools.append([program.id for program in pool])
            return pool[0]

        with patch.object(
            experiment.family, "sample_uniform_family", side_effect=[1, 0]
        ) as uniform_family, patch.object(
            experiment, "baseline_program",
            side_effect=AssertionError("warmup used global baseline sampling"),
        ), patch(
            "run_evo.weighted_program_from_pool", side_effect=choose_first
        ):
            selection = experiment.select(1, "warmup", 0.0)

        self.assertEqual(uniform_family.call_count, 2)
        self.assertEqual(selection.parent_family, 1)
        self.assertEqual(selection.donor_family, 0)
        self.assertEqual(selection.parent.id, "B0")
        self.assertEqual(selection.donor.id, "A0")
        self.assertEqual(pools, [["B0", "B1", "B2"], ["A0", "A1", "A2"]])

    def test_donor_family_is_sampled_once_and_parent_may_be_donor(self):
        experiment = Experiment.__new__(Experiment)
        experiment.args = SimpleNamespace(condition="full_create", seed=104729)
        experiment.family = FamilyIndex(0.85)
        only = Program("P0", "", 1.0)
        experiment.programs = {only.id: only}
        assignment = experiment.family.assign(
            only.id, only.score, "singleton family", [1.0, 0.0]
        )
        only.family = assignment["family"]

        with patch.object(
            experiment.family, "sample_create_action", return_value=0
        ), patch.object(
            experiment.family, "sample_create_donor_family", return_value=0
        ) as donor_family_sampler:
            selection = experiment.select(1, "normal", 0.5)

        donor_family_sampler.assert_called_once()
        self.assertIs(selection.parent, only)
        self.assertIs(selection.donor, only)
        self.assertEqual(selection.mutation_intent, "REFINE")

    def test_warmup_never_samples_create_and_end_conditions_are_unchanged(self):
        experiment = sampling_experiment()
        with patch.object(
            experiment.family, "sample_create_action",
            side_effect=AssertionError("warmup sampled CREATE")
        ):
            selection = experiment.select(1, "warmup", 0.0)
        self.assertIsNone(selection.search_action)
        self.assertIsNotNone(selection.parent)
        self.assertIsNotNone(selection.donor)

        warmup = Experiment.__new__(Experiment)
        warmup.warmup_done = False
        warmup.family = FamilyIndex(0.8)
        warmup.family.assign("P0", 1.0, "seed", [1.0, 0.0])
        warmup.args = SimpleNamespace(k_min=2, warmup_budget=3, condition="full_create")
        warmup.b = 2
        warmup.start = time.monotonic()
        warmup.maybe_end_warmup()
        self.assertFalse(warmup.warmup_done)
        warmup.family.assign("P1", 1.0, "new", [0.0, 1.0])
        warmup.maybe_end_warmup()
        self.assertTrue(warmup.warmup_done)
        self.assertTrue(warmup.warmup_reached_k)

    def test_one_invalid_create_consumes_exactly_one_proposal(self):
        args = SimpleNamespace(
            condition="full_create",
            replicate=1,
            seed=104729,
            proposals=1,
            k_min=1,
            warmup_budget=40,
            entropy_window=25,
            family_similarity_threshold=0.85,
            evaluator_timeout_sec=300,
        )
        config = {
            "generation_model": "unused",
            "embedding_model": "unused",
            "temperature": 0.7,
            "max_tokens": 8192,
            "warmup_validity": {"enabled": True},
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "run_evo.get_client_llm", return_value=(object(), "unused", None)
        ), patch("run_evo.EmbeddingClient"):
            args.run_name = str(Path(tmp) / "one-create")
            experiment = Experiment(args, config)
            experiment.evaluate = Mock(side_effect=[
                (True, 1.0, 0.0),
                (False, None, 0.0),
            ])

            def observe(program, proposal_id):
                assignment = experiment.family.assign(
                    program.id, program.score, f"summary {program.id}", [1.0, 0.0]
                )
                program.family = assignment["family"]
                return assignment, 0.0, 0.0

            experiment.observe = Mock(side_effect=observe)
            experiment.select = Mock(return_value=SearchSelection(
                CREATE_ACTION, CREATE_ACTION, 0.5, None, None, None, None
            ))
            experiment.generate = Mock(return_value=("invalid candidate", 0.0))
            experiment.run()

            events = [
                json.loads(line)
                for line in (experiment.run_dir / "events.jsonl").read_text().splitlines()
            ]
            summary = json.loads((experiment.run_dir / "run_summary.json").read_text())

        self.assertEqual(experiment.b, 1)
        self.assertEqual(experiment.select.call_args.args[2], 0.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["search_action"], CREATE_ACTION)
        self.assertIsNone(events[0]["parent_family"])
        self.assertIsNone(events[0]["donor_family"])
        self.assertFalse(events[0]["child_valid"])
        self.assertEqual(summary["create_attempts"], 1)
        self.assertEqual(summary["create_valid"], 0)
        self.assertEqual(summary["create_new_family_successes"], 0)
        self.assertEqual(summary["create_success_rate"], 0.0)


class ProgramSamplingTests(unittest.TestCase):
    def test_member_probabilities_match_family_local_score_and_children_formula(self):
        experiment = sampling_experiment()
        rng = CapturingRng()
        experiment.member_program(0, rng)

        scores = np.asarray([1.0, 2.0, 4.0])
        median = np.median(scores)
        mad = max(float(np.median(np.abs(scores - median))), 1e-6)
        sigmoid = 1.0 / (1.0 + np.exp(-10.0 * (scores - median) / mad))
        expected = sigmoid / np.asarray([1.0, 2.0, 4.0])
        expected /= expected.sum()
        np.testing.assert_allclose(rng.probabilities, expected)

    def test_higher_score_has_greater_probability_all_else_equal(self):
        pool = [Program(str(i), "", score) for i, score in enumerate((1.0, 2.0, 3.0))]
        rng = CapturingRng()
        weighted_program_from_pool(pool, rng)
        self.assertGreater(rng.probabilities[1], rng.probabilities[0])
        self.assertGreater(rng.probabilities[2], rng.probabilities[1])

    def test_more_children_lowers_probability_all_else_equal(self):
        pool = (
            Program("unused", "", 2.0, children=0),
            Program("used", "", 2.0, children=3),
        )
        rng = CapturingRng()
        weighted_program_from_pool(pool, rng)
        np.testing.assert_allclose(rng.probabilities, [0.8, 0.2])

    def test_equal_scores_produce_finite_normalized_probabilities(self):
        pool = [Program(str(i), "", 7.0, children=i) for i in range(4)]
        rng = CapturingRng()
        weighted_program_from_pool(pool, rng)
        self.assertTrue(np.all(np.isfinite(rng.probabilities)))
        self.assertAlmostEqual(float(rng.probabilities.sum()), 1.0)

    def test_member_program_never_samples_outside_selected_family(self):
        experiment = sampling_experiment()
        family_ids = set(experiment.family.member_ids(0))
        for seed in range(100):
            selected = experiment.member_program(0, np.random.RandomState(seed))
            self.assertIn(selected.id, family_ids)

    def test_parent_and_donor_both_use_shared_family_pool_weighting(self):
        for condition in ("full_annealed", "full_create"):
            with self.subTest(condition=condition):
                experiment = sampling_experiment(condition)
                pools = []

                def choose_first(pool, rng):
                    pool = list(pool)
                    pools.append([program.id for program in pool])
                    return pool[0]

                family_patches = (
                    (
                        patch.object(experiment.family, "sample_create_action", return_value=0),
                        patch.object(
                            experiment.family, "sample_create_donor_family", return_value=0
                        ),
                    )
                    if condition == "full_create" else
                    (
                        patch.object(
                            experiment.family, "sample_create_donor_family", return_value=0
                        ),
                        patch.object(
                            experiment.family, "sample_annealed_family", return_value=0
                        ),
                    )
                )
                with family_patches[0], family_patches[1], patch(
                    "run_evo.weighted_program_from_pool", side_effect=choose_first
                ):
                    selection = experiment.select(1, "normal", 0.5)

                self.assertEqual(selection.parent.id, "A0")
                self.assertEqual(selection.donor.id, "A0")
                self.assertEqual(
                    pools, [["A0", "A1", "A2"], ["A0", "A1", "A2"]]
                )

    def test_seeded_selection_is_reproducible(self):
        for condition in ("full_annealed", "full_create"):
            with self.subTest(condition=condition):
                left = sampling_experiment(condition)
                right = sampling_experiment(condition)
                left_selections = [
                    (
                        selection.search_action,
                        selection.parent.id if selection.parent else None,
                        selection.donor.id if selection.donor else None,
                    )
                    for i in range(1, 21)
                    for selection in [left.select(i, "normal", i / 20)]
                ]
                right_selections = [
                    (
                        selection.search_action,
                        selection.parent.id if selection.parent else None,
                        selection.donor.id if selection.donor else None,
                    )
                    for i in range(1, 21)
                    for selection in [right.select(i, "normal", i / 20)]
                ]
                self.assertEqual(left_selections, right_selections)


class RepresentationTests(unittest.TestCase):
    def test_summary_schema_is_exactly_the_authoritative_eleven_fields(self):
        self.assertEqual(SUMMARY_FIELDS, (
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
        ))

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
