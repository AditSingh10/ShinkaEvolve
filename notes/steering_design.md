# Integration Map — Event-Triggered Search Steering for AuxEvolve (Circle Packing 3-aux)

Spec: `/ua/ikakkar/Claude Code Experiment Spec_ Event-Triggered Search Steering for AuxEvolve.md`
This maps every spec section to the concrete file/function/change. **No code is written until this is signed off.**

## Locked design decisions
- **D1 — Parallel family layer.** The existing 2-island + migration + β/Q evolution is left **bit-identical** to the current 3-aux baseline. A new `FamilyManager` maintains algorithm-families as a *separate* bookkeeping layer over admitted valid programs. Steering re-weights **parent-pool (via a family whitelist), inspiration selection, and the prompt** — it never mutates `island_idx`, migration, or β/Q. `NORMAL` mode reproduces baseline exactly.
- **D2 — LLM algorithm summary.** One local-Qwen call (temp 0) per admitted valid child → structured summary; embed the summary text with `qwen-embed`; cache both keyed by program id / code hash.
- **D3 — Budget.** `b` = monotonic proposal counter over **all** proposals (valid + invalid + novelty-rejected). In this codebase the generation counter already increments once per proposal attempt (circle baseline: 124 programs + 77 novelty-rej = 201 ≈ 200 gens), so `b ≡ generation index` and `B = num_generations`. `t = b/B`. Controller **checks** and `t` key on `b`; **family + validity statistics** update only on evaluated/admitted children.
- **D4 — Async semantics.** Mode is **snapshotted at proposal-build time** onto the proposal; controllers **update at eval-completion**; ON/OFF checks run on completed-eval count. No race; NORMAL unaffected.

## New code (modular, ablation-friendly — spec §25)
New module `shinka/core/steering.py` (or `shinka/database/steering.py`):
| Class | Responsibility | Spec §|
|---|---|---|
| `FamilyManager` | `summarize_program` (LLM, cached), `embed_summary`, `assign_family` (centroid vs `family_similarity_threshold`, else new family), `family_stats` (counts, entropy, dominant frac), `family_distance` | §3, §4, §17C, §24 |
| `DiversityController` | EMA-free per-check: `gamma`, `entropy`, `target_entropy=H_late+(H_early−H_late)(1−t)`, `should_activate/deactivate` (γ_on/γ_off hysteresis, confirmation counts) | §4, §5 |
| `ValidityController` | per-constraint `violation EMA r_k`, `lambda_k` dual-ascent clip[0,1], `active_constraint=argmax λ`, witness buffers, per-family feasibility R_{i,k}; `should_activate/deactivate` (λ_on/λ_off ∧ r_on/r_off hysteresis) | §6, §7 |
| `SteeringPolicy` | `current_mode` (NORMAL/DIVERSITY/VALIDITY/COMBINED), `sample_family`, `sample_inspiration_families`, `build_prompt_addition` | §8–§16 |

All four are **pure/observable**: they read state and return decisions; they hold no evolution state and can be disabled independently.

## Config — new `SteeringConfig` (attached to `DatabaseConfig`, all defaults = spec)
Ablation flags (spec §23): `enable_diversity_controller`, `enable_validity_controller`, `enable_family_based_islands`, `enable_mode_prompt_conditioning`, `enable_mode_specific_island_sampling`, `enable_mode_specific_inspiration_sampling`.
Numeric knobs (spec-provided defaults): `family_similarity_threshold=0.85`, `H_early=0.90`, `H_late=0.40`, `controller_warmup=20`, `controller_check_interval=5`, `gamma_on=0.15/gamma_off=0.05`, `trigger/release_confirmation_checks=2`, `validity_ema_alpha=0.15`, `lambda_lr=0.10`, `target_violation_rate=0.15`, `lambda_on=0.60/off=0.25`, `violation_rate_on=0.30/off=0.15`, `num_failure_witnesses=3`, island quality weights (early 0.30/late 0.80), `validity_island_quality_weight=0.50`, combined split `0.7/0.3`, softmax temps + prob floor.
With **all enable_* = False** the config is a no-op → current baseline. Ablation A = flags off; B = family tracking only; C/D/E/F/G per spec §23.

## Primary endpoint — yield: steering vs no-steering (the whole point)
The experiment's headline comparison is **valid-yield with steering ON vs OFF**, over a fixed proposal budget B.

**Metric (primary):** `valid_yield = #valid ÷ #total_proposals`, where `#total_proposals` counts **every** proposal incl. novelty-rejected + invalid (decision D3). This is exactly the funnel we already compute in `wasted_proposals_langs.py`; the same table + per-proposal timeline is the instrument — no new analysis code, just point it at the steer/no-steer runs.

**Funnel decomposition (secondary, sums to 100%):** invalid-feasibility rate (overlap/boundary) and novelty-rejection rate. These say *how* yield moved (fewer infeasible? fewer duplicates?).

**Arms:**
| arm | flags | question it answers |
|---|---|---|
| **NO-STEER** (= ablation A) | all `enable_*` off → **bit-identical to current 3-aux baseline** | control |
| **STEER** (= ablation E) | diversity + validity both on | does steering raise yield? |
| diversity-only (C) | diversity on | attribute the gain |
| validity-only (D) | validity on | attribute the gain |
| prompt-off (F) / selection-off (G) | per §23 | selection vs prompt attribution |

**Two views of yield:**
1. **Whole-run** valid-yield STEER vs NO-STEER (the headline number).
2. **Windowed** valid-yield: pre-activation window vs during-steering window (§21) — isolates whether yield actually moved *while steering was active* (steering is event-triggered, so the whole-run number dilutes the effect).

**Quality guard:** also compare **best objective** and **best-vs-valid-proposals** (§22 #1/#2). Higher yield only counts if the ceiling holds or improves — steering must not buy feasibility by destroying packing quality.

**Protocol:** same B, n≥5 reps/arm, identical config/model/temp; **run STEER and NO-STEER interleaved under matched GPU load** (the earlier contention confound); one-sided Mann-Whitney (STEER valid-yield > NO-STEER) + effect size, mirroring the prereg study. **NO-STEER is NOT re-run — we reuse the existing baseline control runs** (`circle_packing_alg2/results/pilot_none_control*`), since with all `enable_*` off the code path is bit-identical to those runs. Only the STEER arms are run fresh.

## Section-by-section mapping
| Spec § | What it asks | Where it hooks |
|---|---|---|
| §2, §10 | preserve β/Q/aux | **No change** to `aux_selection.py` / `CombinedParentSelector`. |
| §3 islands→families | family from summary-embedding | **Two orthogonal partitions of the same programs:** real islands (`island_idx∈{0,1}`, fixed, drive migration + β/Q — untouched) and families (`family_id∈{0..K−1}`, dynamic, drive steering). Each valid program carries both tags. Family construction: on admission → LLM summary → embed → cosine vs family centroids → join nearest if ≥0.85 else spawn new family. "Sample island under each mode" = **sample family**, realized as a parent-pool whitelist into the existing β/Q (D1). |
| §3 t | b/B | proposal counter in `async_runner` (D3). |
| §4 γ | family entropy, H*(t), γ | `DiversityController`. |
| §5 | diversity trigger + hysteresis | `DiversityController.should_activate/deactivate`. |
| §6 λ | violation EMA + dual-ascent λ_k | `ValidityController`; **needs structured violations from checker (below)**. |
| §7 | validity trigger + hysteresis | `ValidityController`. |
| §8 | 4-state machine | `SteeringPolicy.current_mode`. |
| §9 island sampling | mode-specific family sampling | `SteeringPolicy.sample_family` → parent-pool **whitelist** threaded into the parent sampler **only under steering** (gated by `enable_mode_specific_island_sampling`). NORMAL → existing `island_sampler`. |
| §10 parent | β/Q within chosen family | reuse `CombinedParentSelector`, constrained to whitelist ids. |
| §11 inspiration | cross-family / feasibility-biased | wrap `CombinedContextSelector.sample_context`; bypass `enforce_island_separation` **only under steering** (gated by `enable_mode_specific_inspiration_sampling`). |
| §12–§16 prompt | append mode blocks | append to per-proposal user message in `query.py`/`PromptSampler`; **no-op when NORMAL** (gated by `enable_mode_prompt_conditioning`). |
| §17A | objective credit | unchanged (β/Q). |
| §17B | validity update incl. invalid children | `ValidityController.update` at eval-completion, for every evaluated child. |
| §17C | family update for valid children | `FamilyManager.assign_family` at admission; log parent→child family transition. |
| §18 | full loop | orchestrated in `async_runner` proposal path (build-time mode snapshot + eval-time update). |
| §19 | do-not-add list | enforced: nothing added to Q/β/objective; no operator taxonomy; no harsher novelty. |
| §20 | novelty as safeguard | **Novelty logic UNCHANGED** (0.99 code-embedding gate + LLM judge; §19 forbids harshening). Note it operates on *full-code* embedding, orthogonal to family assignment's *summary* embedding. Novelty-rejected proposals count into `b` and the novelty-rejection-rate trace but feed neither validity nor family stats. Steering acts upstream (parent pool + cross-family inspiration + prompt) to lower rejections; rejection-rate is a measured **outcome**, tracked before/after each diversity activation. |
| §21 | intervention effectiveness | activation/deactivation records with pre/post stats → `steering_events.jsonl`. |
| §22 | 15 metrics | `steering.jsonl` per-proposal record (below) → plotting script. |
| §24 | edge cases | K=1 → H=0, most-distant-program fallback, allow new family; empty witness → aggregate only, no activate; argmax λ; new-family shrinkage on quality. |
| §26 | tests | `test_steering.py` (list below). |

## Trusted-checker change (required — spec §6, §15)
`examples/circle_packing_alg2/evaluate.py`: replace the first-violation early-return `validate_packing` with a **collect-all** pass emitting into `private_metrics["violations"]`:
```
violations = {
  "overlap":  [{"i":4,"j":8,"amount":0.016}, ...],   # all overlapping pairs, magnitude = (r_i+r_j)-dist
  "boundary": [{"i":9,"amount":0.004}, ...],          # all out-of-square circles, amount outside
}
```
`combined_score` / `oracle_score` / aux logic unchanged. A program is still "valid" iff both lists empty. Constraint families for v1: **overlap, boundary** (spec §6).

## Logging
- `steering.jsonl` (per proposal, spec §18 step 11 + §22): `b, t, mode, island, parent_id, parent_family, inspiration_ids, inspiration_families, beta, selection_arm, Q, H, H_target, gamma, dominant_fraction, family_counts, r_k[], lambda_k[], active_constraint, valid, violations, witnesses, objective, novelty_result, created_new_family`.
- `steering_events.jsonl` (per activation/deactivation, spec §21): mode, b_on, pre-stats, b_off, post-stats.
- `families.jsonl`: family id → summary, centroid, members, best score.

## Tests (`test_steering.py`, spec §26)
t=b/B; normalized entropy incl. K=1→0; H*(t) schedule; γ; diversity ON/OFF hysteresis; validity EMA; λ rise on repeated violation; λ decay on stop; validity hysteresis; argmax active constraint; family assign + new-family creation; diversity island sampling favors underrepresented; validity sampling favors feasible families; cross-family inspiration under diversity; K=1 fallback no-crash; correct prompt block per mode; **NORMAL == baseline** (selection + prompt).

## Staged build (spec §27) with checkpoints
1. structured logging + `b` counter + checker `violations`.
2. `FamilyManager` (+ family stats), observe-only. **Checkpoint: family entropy/dominant-fraction traces on an existing baseline run look sane.**
3. `DiversityController` γ, observe-only. 4. `ValidityController` λ, observe-only. **Checkpoint: γ rises during the known collapse; λ_overlap rises during the red-wall stretch — controllers correctly *diagnose* before steering.**
5. state machine. 6. mode-specific family/parent sampling. 7. mode-specific inspiration. 8. prompt conditioning. 9. ablation flags. 10. tests. **Checkpoint: smoke run (short B) shows a full NORMAL→steer→NORMAL cycle in the logs**, then full 200-proposal run.

## Open follow-ups (not blocking)
- Confirm `b ≡ generation index` empirically at step 1 (expected yes).
- New folder `examples/circle_packing_steer/` forking `shinka_alg2.yaml` (keeps baseline runs intact) — will do unless you prefer in-place.
