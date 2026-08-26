# Pre-registration — Does event-triggered search steering help? (circle packing, 3-aux)

Registered BEFORE looking at results.

## Design
- Base: `shinka_alg2.yaml` (3-aux caging/hole/connect, Qwen2.5-32B, 200 gens, crowding archive).
- **Arm A (baseline):** 3-aux, NO steering (`shinka_alg2.yaml`).
- **Arm E (steering):** 3-aux + ALL controllers (`shinka_steer_full.yaml`): diversity + validity, family
  islands, mode prompt-conditioning, mode-specific island + inspiration sampling. **Spec-default**
  warmup=20, check_interval=5, thresholds unchanged.
- n = 5 reps per arm, **interleaved** on the same GPUs (any drift hits both arms equally), 3 concurrent.
- Both arms use the updated checker → valid-yield AND runtime-crash rate measurable.

## Metrics (fixed now)
| metric | definition |
|---|---|
| **valid-yield** (primary) | valid ÷ total proposals (incl. novelty-rejected) |
| runtime-crash rate | runtime-invalid ÷ evaluated children |
| ceiling | best valid combined_score |
| novelty-rejection rate | novelty-rejected ÷ total proposals |

## Hypotheses & test (one-sided Mann-Whitney U, α=0.05, per metric)
- **H1 (primary):** STEER valid-yield > baseline.
- **H2:** STEER runtime-crash rate < baseline.
- **H3:** STEER ceiling ≥ baseline (steering must not destroy task quality).
- Report medians + MWU p; state plainly whether H1 is supported or rejected. No post-hoc metric swapping.

## Decision rule
Steering "works" iff **H1 supported (p<0.05, STEER higher)** AND H3 not violated (ceiling not
significantly worse). H2 is the mechanism check. n=5 is modest power; a null is "not shown to help,"
not "proven useless."
