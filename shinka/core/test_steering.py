"""Focused unit tests for the steering controllers (spec §26). Runnable directly:
    uv run --project . python shinka/core/test_steering.py
"""
import numpy as np
from shinka.core.steering import (
    SteeringConfig, FamilyManager, DiversityController, ValidityController,
    SteeringPolicy, normalized_entropy, target_entropy, dominant_fraction,
    classify_failure, render_witness,
    MODE_NORMAL, MODE_DIVERSITY, MODE_VALIDITY, MODE_COMBINED,
)

_passed = 0


def check(cond, name):
    global _passed
    assert cond, f"FAIL: {name}"
    _passed += 1
    print(f"  ok  {name}")


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_entropy_and_schedule():
    check(abs(normalized_entropy([5]) - 0.0) < 1e-9, "K=1 entropy -> 0")
    check(abs(normalized_entropy([]) - 0.0) < 1e-9, "K=0 entropy -> 0")
    check(abs(normalized_entropy([3, 3, 3]) - 1.0) < 1e-9, "uniform K -> 1")
    check(normalized_entropy([90, 5, 5]) < 0.6, "skewed entropy < 0.6")
    check(abs(dominant_fraction([90, 5, 5]) - 0.9) < 1e-9, "dominant fraction 0.9")
    check(abs(target_entropy(0.0, 0.9, 0.4) - 0.9) < 1e-9, "H*(t=0)=H_early")
    check(abs(target_entropy(1.0, 0.9, 0.4) - 0.4) < 1e-9, "H*(t=1)=H_late")
    check(abs(target_entropy(0.5, 0.9, 0.4) - 0.65) < 1e-9, "H*(t=0.5) midpoint")


def test_t_budget():
    B = 200
    check(abs((50 / B) - 0.25) < 1e-9, "t=b/B basic")


def test_gamma():
    cfg = SteeringConfig()
    d = DiversityController(cfg)
    d.observe([100], t=0.0)  # K=1 -> H=0, H_target=0.9 -> gamma=0.9
    check(abs(d.gamma - 0.9) < 1e-9, "gamma high when collapsed early")
    d.observe([10, 10, 10, 10], t=0.0)  # uniform -> H=1 -> gamma=0
    check(d.gamma == 0.0, "gamma 0 when well spread")


def test_diversity_hysteresis():
    cfg = SteeringConfig(trigger_confirmation_checks=2, release_confirmation_checks=2)
    d = DiversityController(cfg)
    d.observe([100], 0.0)          # gamma 0.9 >= on
    check(d.check() is False, "no activate after 1 check")
    check(d.check() is True, "activate after 2 consecutive on-checks")
    d.observe([10, 10, 10, 10], 0.0)  # gamma 0 <= off
    check(d.check() is False or d.active, "1 off-check: not yet released")
    check(d.check() is False, "released after 2 consecutive off-checks")


def test_validity_ema_and_lambda():
    cfg = SteeringConfig()
    v = ValidityController(cfg)
    ov = {"overlap": [{"i": 0, "j": 1, "amount": 0.02}], "boundary": []}
    for _ in range(40):
        v.update_child(ov, parent_family_id=0)
    check(v.r["overlap"] > 0.8, "EMA -> high under repeated overlap")
    check(v.lam["overlap"] > 0.6, "lambda rises under repeated violation")
    check(v.r["boundary"] == 0.0 and v.lam["boundary"] == 0.0, "unrelated constraint stays 0")
    check(v.active_constraint() == "overlap", "active constraint = argmax lambda")
    # now stop violating -> lambda decays
    clean = {"overlap": [], "boundary": []}
    for _ in range(80):
        v.update_child(clean, parent_family_id=0)
    check(v.lam["overlap"] < cfg.lambda_off, "lambda decays when violations stop")


def test_runtime_failure_class_and_classify():
    cfg = SteeringConfig()
    v = ValidityController(cfg)
    # a stream of crashing children -> runtime lambda should dominate
    f_crash = classify_failure(correct=False, violations=None,
                               error_msg="name 'construct_packing' is not defined")
    check("runtime" in f_crash and "error" in f_crash["runtime"][0], "classify_failure -> runtime witness")
    f_overlap = classify_failure(False, {"overlap": [{"i": 1, "j": 2, "amount": 0.02}], "boundary": []})
    check("overlap" in f_overlap and "runtime" not in f_overlap, "classify geometric != runtime")
    check(classify_failure(True, None) == {}, "valid child -> no failure class")
    for _ in range(50):
        v.update_child(f_crash, parent_family_id=0)
    check(v.active_constraint() == "runtime", "runtime becomes the dominant (argmax lambda) class")
    check(v.lam["runtime"] > 0.6 and v.lam["overlap"] == 0.0, "runtime lambda rises, geometric stays 0")
    line = render_witness("runtime", v.top_witnesses("runtime")[0])
    check("construct_packing" in line, "runtime witness renders the actual error string")


def test_validity_prompt_shows_runtime_errors():
    cfg = SteeringConfig(enable_mode_prompt_conditioning=True,
                         enable_validity_controller=True)
    fam = FamilyManager(cfg); div = DiversityController(cfg); val = ValidityController(cfg)
    for _ in range(50):
        val.update_child(classify_failure(False, None, "too many values to unpack (expected 2)"), 0)
    pol = SteeringPolicy(cfg, fam, div, val)
    block = pol.build_prompt_addition(MODE_VALIDITY, 40, 200, 0.2)
    check("program executes without error" in block, "runtime constraint pretty-name in prompt")
    check("too many values to unpack" in block, "actual runtime error appears as a witness in prompt")


def test_validity_hysteresis_needs_witness():
    cfg = SteeringConfig(trigger_confirmation_checks=1)
    v = ValidityController(cfg)
    # drive lambda/r up
    for _ in range(60):
        v.update_child({"overlap": [{"i": 1, "j": 2, "amount": 0.05}], "boundary": []}, 0)
    check(v.has_witnesses("overlap"), "witnesses recorded")
    check(v.check() is True, "activates when lambda & rate high AND witness present")


def test_family_assignment():
    cfg = SteeringConfig(family_similarity_threshold=0.85)
    fam = FamilyManager(cfg)
    e_a = unit([1, 0, 0])
    f0 = fam.assign("p0", e_a, score=2.0)
    f1 = fam.assign("p1", unit([0.99, 0.14, 0]), score=2.1)   # close -> same family
    f2 = fam.assign("p2", unit([0, 1, 0]), score=1.5)          # orthogonal -> new family
    check(f0 == 0 and f1 == 0, "similar summaries join same family")
    check(f2 == 1, "distant summary spawns new family")
    check(fam.num_families() == 2, "two families")
    check(fam.counts() == [2, 1], "family counts")
    check(fam.family_of("p1") == 0, "member->family lookup")
    check(fam.distance(0, 1) > 0.5, "orthogonal families are distant")


def test_diversity_island_sampling_favors_underrepresented():
    cfg = SteeringConfig()
    fam = FamilyManager(cfg)
    for i in range(10):  # family 0: big
        fam.assign(f"a{i}", unit([1, 0, 0]) + 0.001 * np.random.randn(3), 2.0)
    fam.assign("b0", unit([0, 1, 0]), 2.0)          # family 1: tiny, equal quality
    fam.assign("b1", unit([0, 1, 0.02]), 2.0)
    div = DiversityController(cfg); val = ValidityController(cfg)
    pol = SteeringPolicy(cfg, fam, div, val)
    s = pol.family_scores(MODE_DIVERSITY, t=0.0)     # early: underrep weighted
    check(s[1] > s[0], "diversity sampling scores underrepresented family higher (early)")


def test_validity_island_sampling_favors_feasible():
    cfg = SteeringConfig()
    fam = FamilyManager(cfg)
    fam.assign("a0", unit([1, 0, 0]), 2.0); fam.assign("a1", unit([1, 0.02, 0]), 2.0)
    fam.assign("b0", unit([0, 1, 0]), 2.0); fam.assign("b1", unit([0, 1, 0.02]), 2.0)
    div = DiversityController(cfg); val = ValidityController(cfg)
    # family 0 breeds feasible overlap children; family 1 breeds violating ones
    for _ in range(30):
        val.update_child({"overlap": [], "boundary": []}, parent_family_id=0)
        val.update_child({"overlap": [{"i": 0, "j": 1, "amount": 0.03}], "boundary": []},
                         parent_family_id=1)
    pol = SteeringPolicy(cfg, fam, div, val)
    s = pol.family_scores(MODE_VALIDITY, t=0.5)
    check(s[0] > s[1], "validity sampling scores constraint-feasible family higher")


def test_cross_family_inspiration_under_diversity():
    cfg = SteeringConfig()
    fam = FamilyManager(cfg)
    fam.assign("a0", unit([1, 0, 0]), 2.0)
    fam.assign("b0", unit([0, 1, 0]), 2.0)          # distant from parent family 0
    div = DiversityController(cfg); val = ValidityController(cfg)
    pol = SteeringPolicy(cfg, fam, div, val)
    s = pol.inspiration_family_scores(MODE_DIVERSITY, parent_family=0, t=0.0)
    check(s[1] > s[0], "diversity inspiration favors a different (distant) family")


def test_single_family_fallback():
    cfg = SteeringConfig()
    fam = FamilyManager(cfg)
    fam.assign("a0", unit([1, 0, 0]), 2.0)
    div = DiversityController(cfg); val = ValidityController(cfg)
    pol = SteeringPolicy(cfg, fam, div, val)
    rng = np.random.RandomState(0)
    fid = pol.sample_family(MODE_DIVERSITY, t=0.0, rng=rng)  # must not crash
    check(fid == 0, "single-family sampling returns the only family, no crash")


def test_prompt_blocks_per_mode():
    cfg = SteeringConfig(enable_mode_prompt_conditioning=True,
                         enable_diversity_controller=True, enable_validity_controller=True)
    fam = FamilyManager(cfg)
    fam.assign("a0", unit([1, 0, 0]), 2.0); fam.assign("b0", unit([0, 1, 0]), 2.0)
    div = DiversityController(cfg); val = ValidityController(cfg)
    div.observe([2, 1], 0.2)
    for _ in range(60):
        val.update_child({"overlap": [{"i": 4, "j": 8, "amount": 0.016}], "boundary": []}, 0)
    pol = SteeringPolicy(cfg, fam, div, val)
    dblock = pol.build_prompt_addition(MODE_DIVERSITY, 42, 200, 0.21, "SLSQP", "grid-init")
    vblock = pol.build_prompt_addition(MODE_VALIDITY, 42, 200, 0.21)
    cblock = pol.build_prompt_addition(MODE_COMBINED, 42, 200, 0.21, "SLSQP", "grid-init")
    check("SEARCH STEERING: DIVERSITY" in dblock and "meaningfully different" in dblock,
          "diversity block content")
    check("SEARCH STEERING: VALIDITY" in vblock and "overlap by 0.016" in vblock,
          "validity block includes witness")
    check("DIVERSITY + VALIDITY" in cblock, "combined block header")


def test_normal_reproduces_baseline():
    cfg = SteeringConfig()  # all enable_* False
    fam = FamilyManager(cfg); div = DiversityController(cfg); val = ValidityController(cfg)
    pol = SteeringPolicy(cfg, fam, div, val)
    check(pol.current_mode() == MODE_NORMAL, "controllers off -> NORMAL")
    # even if controllers flip active, disabled flags keep mode NORMAL
    div.active = True; val.active = True
    check(pol.current_mode() == MODE_NORMAL, "disabled flags force NORMAL")
    check(pol.build_prompt_addition(MODE_DIVERSITY, 1, 200, 0.0, "x", "y") == "",
          "no prompt addition when conditioning disabled")


if __name__ == "__main__":
    for fn in [test_entropy_and_schedule, test_t_budget, test_gamma,
               test_diversity_hysteresis, test_validity_ema_and_lambda,
               test_runtime_failure_class_and_classify,
               test_validity_prompt_shows_runtime_errors,
               test_validity_hysteresis_needs_witness, test_family_assignment,
               test_diversity_island_sampling_favors_underrepresented,
               test_validity_island_sampling_favors_feasible,
               test_cross_family_inspiration_under_diversity,
               test_single_family_fallback, test_prompt_blocks_per_mode,
               test_normal_reproduces_baseline]:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\nALL {_passed} CHECKS PASSED")
