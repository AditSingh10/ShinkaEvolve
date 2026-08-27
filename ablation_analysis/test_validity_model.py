import copy
import unittest
from types import SimpleNamespace

import numpy as np

from family_model import FamilyIndex
from run_evo import Experiment, Program
from validity_model import ValidityConfig, ValidityController, classify_failure


FAILURE = {"overlap": [{"i": 1, "j": 2, "amount": 0.05}]}


def controller_config(**overrides):
    values = {
        "controller_warmup": 0,
        "controller_check_interval": 1,
        "trigger_confirmation_checks": 2,
        "release_confirmation_checks": 2,
    }
    values.update(overrides)
    return ValidityConfig(**values)


class ValidityStateTests(unittest.TestCase):
    def test_violation_rate_ema_update(self):
        controller = ValidityController(controller_config(
            validity_ema_alpha=0.2, lambda_lr=0.0,
        ))
        controller.update(FAILURE)
        self.assertAlmostEqual(controller.r["overlap"], 0.2)
        controller.update({})
        self.assertAlmostEqual(controller.r["overlap"], 0.16)

    def test_lambda_update_uses_updated_ema(self):
        controller = ValidityController(controller_config(
            validity_ema_alpha=1.0, lambda_lr=0.1, target_violation_rate=0.15,
        ))
        controller.update(FAILURE)
        self.assertAlmostEqual(controller.lam["overlap"], 0.085)

    def test_lambda_is_clipped_to_unit_interval(self):
        controller = ValidityController(controller_config(
            validity_ema_alpha=1.0, lambda_lr=2.0, target_violation_rate=0.15,
        ))
        controller.update(FAILURE)
        self.assertEqual(controller.lam["overlap"], 1.0)
        controller.update({})
        self.assertEqual(controller.lam["overlap"], 0.7)
        controller.update({})
        self.assertAlmostEqual(controller.lam["overlap"], 0.4)
        controller.update({})
        self.assertAlmostEqual(controller.lam["overlap"], 0.1)
        controller.update({})
        self.assertEqual(controller.lam["overlap"], 0.0)

    def test_activation_requires_confirmation_checks(self):
        controller = ValidityController(controller_config(
            validity_ema_alpha=1.0, lambda_lr=1.0, target_violation_rate=0.0,
            lambda_on=0.5, violation_rate_on=0.5,
        ))
        controller.update(FAILURE)
        self.assertFalse(controller.check(1)["trigger_on_event"])
        controller.update(FAILURE)
        event = controller.check(2)
        self.assertTrue(event["trigger_on_event"])
        self.assertTrue(controller.active)

    def test_release_uses_hysteresis_and_confirmation(self):
        controller = ValidityController(controller_config())
        controller.active = True
        controller.episodes.append({
            "start_proposal": 1, "first_prompt_proposal": 2,
            "end_proposal": None, "last_prompt_proposal": None,
            "active_constraint": "overlap",
        })
        controller.lam["overlap"] = 0.2
        controller.r["overlap"] = 0.1
        self.assertFalse(controller.check(1)["trigger_off_event"])
        event = controller.check(2)
        self.assertTrue(event["trigger_off_event"])
        self.assertFalse(controller.active)
        self.assertEqual(controller.episodes[0]["end_proposal"], 2)

    def test_no_activation_without_witness(self):
        controller = ValidityController(controller_config(
            lambda_on=0.5, violation_rate_on=0.5,
        ))
        controller.lam["overlap"] = 1.0
        controller.r["overlap"] = 1.0
        controller.check(1)
        controller.check(2)
        self.assertFalse(controller.active)

    def test_active_constraint_is_maximum_lambda(self):
        controller = ValidityController(controller_config())
        controller.lam["overlap"] = 0.6
        controller.lam["runtime"] = 0.7
        self.assertEqual(controller.active_constraint(), "runtime")

    def test_failure_classification(self):
        cases = (
            (dict(correct=True), set()),
            (dict(correct=False, timed_out=True, error_msg="timeout"), {"timeout"}),
            (dict(correct=False, error_msg="Circles 1 & 3 overlap. Dist: 0.1, Sum Radii: 0.2"), {"overlap"}),
            (dict(correct=False, error_msg="Circle 4 is outside unit square"), {"boundary"}),
            (dict(correct=False, error_msg="invalid syntax at line 2"), {"compile"}),
            (dict(correct=False, error_msg="Centers shape incorrect. Expected (26, 2), got (2, 2)"), {"malformed"}),
            (dict(correct=False, error_msg="answer does not match expected result"), {"wrong_answer"}),
            (dict(correct=False, error_msg="list index out of range"), {"runtime"}),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(set(classify_failure(**arguments)), expected)

    def test_validity_block_empty_when_inactive(self):
        controller = ValidityController(controller_config())
        controller.update(FAILURE)
        self.assertEqual(controller.prompt_block(1, 200), "")

    def test_validity_block_added_when_active(self):
        controller = ValidityController(controller_config())
        controller.update(FAILURE)
        controller.lam["overlap"] = 0.7
        controller.active = True
        block = controller.prompt_block(2, 200)
        self.assertIn("SEARCH STEERING: VALIDITY", block)
        self.assertIn("pairwise non-overlap", block)
        self.assertIn("circles 1 and 2", block)


class PromptOnlyIsolationTests(unittest.TestCase):
    @staticmethod
    def experiment(condition, validity=None):
        experiment = Experiment.__new__(Experiment)
        experiment.args = SimpleNamespace(condition=condition, seed=104729)
        experiment.B = 200
        experiment.b = 23
        experiment.validity = validity
        experiment.family = FamilyIndex(0.85)
        programs = (
            Program("P0", "def run_packing():\n    pass", 1.0, children=0),
            Program("P1", "def run_packing():\n    return 1", 1.2, children=1),
            Program("P2", "def run_packing():\n    return 2", 0.9, children=2),
        )
        experiment.programs = {program.id: program for program in programs}
        for index, program in enumerate(programs):
            assignment = experiment.family.assign(
                program.id, program.score, f"summary {program.id}", [1.0, float(index)],
            )
            program.family = assignment["family"]
        return experiment

    def test_validity_cannot_change_parent_or_donor_sampling(self):
        warmup = self.experiment("warmup")
        validity = ValidityController(controller_config())
        validity.update(FAILURE)
        validity.active = True
        combined = self.experiment("warmup_validity", validity)
        np.testing.assert_allclose(
            warmup.family.probabilities(0.5), combined.family.probabilities(0.5)
        )
        for phase in ("warmup", "normal"):
            for proposal_id in range(1, 31):
                left = warmup.select(proposal_id, phase, proposal_id / 200)
                right = combined.select(proposal_id, phase, proposal_id / 200)
                self.assertEqual((left[0].id, left[1].id), (right[0].id, right[1].id))

    def test_disabled_warmup_validity_reduces_to_warmup_prompt(self):
        warmup = self.experiment("warmup")
        disabled = self.experiment("warmup_validity", validity=None)
        for phase in ("warmup", "normal"):
            self.assertEqual(
                warmup.prompt(warmup.programs["P0"], warmup.programs["P1"], phase),
                disabled.prompt(disabled.programs["P0"], disabled.programs["P1"], phase),
            )

    def test_warmup_and_validity_blocks_coexist(self):
        validity = ValidityController(controller_config())
        validity.update(FAILURE)
        validity.active = True
        experiment = self.experiment("warmup_validity", validity)
        prompt = experiment.prompt(experiment.programs["P0"], experiment.programs["P1"], "warmup")
        self.assertIn("FAMILY-SEEDING WARMUP", prompt)
        self.assertIn("SEARCH STEERING: VALIDITY", prompt)

    def test_smoke_trajectory_accounts_for_each_proposal_and_releases(self):
        controller = ValidityController(controller_config(
            validity_ema_alpha=1.0,
            lambda_lr=1.0,
            target_violation_rate=0.15,
            lambda_on=0.6,
            lambda_off=0.25,
            violation_rate_on=0.3,
            violation_rate_off=0.15,
        ))
        trajectory = []
        for proposal_id in range(1, 9):
            injected = bool(controller.prompt_block(proposal_id, 8))
            controller.update(FAILURE if proposal_id <= 2 else {})
            event = controller.check(proposal_id)
            trajectory.append((
                proposal_id, controller.lam["overlap"], injected,
                event["trigger_on_event"], event["trigger_off_event"],
            ))
        self.assertEqual(len(trajectory), 8)
        self.assertTrue(trajectory[1][3])
        self.assertTrue(trajectory[-1][4])
        self.assertEqual(sum(item[2] for item in trajectory), 6)
        self.assertFalse(controller.active)
        self.assertEqual(controller.episodes[0]["start_proposal"], 2)
        self.assertEqual(controller.episodes[0]["end_proposal"], 8)


if __name__ == "__main__":
    unittest.main()
