# game_2048 — Algorithm 2 (aux-guided parent selection)

Tests whether **aux-guided parent selection** speeds up finding good solutions on 2048,
and — with 5 auxes — whether the bandit can **tell good auxes from bad ones** and whether
each aux actually **steers selection differently from the oracle**.

Same machinery as `circle_packing_alg2`: a two-level non-stationary bandit picks, per
step, whether to select a parent by the oracle or by one of the aux signals. Aux scores
are **private** (never shown to the model) — this is a pure *selection* intervention.
Treatment vs control differ **only** in `aux_directions` (5 auxes vs `{}`).

## The oracle vs the auxes

- **Oracle** (`combined_score`): `max_tile/512 − 0.002·num_moves`. Coarse — it only jumps
  when a higher tile is reached. This sparseness is exactly why dense structural auxes
  might help.
- **5 auxes** (computed from the board trajectory in `aux_evals.py`, all prior `+1`):
  | aux | measures | why it might predict good play |
  |---|---|---|
  | `empty` | mean fraction of empty cells | room to maneuver → survive longer |
  | `monotonic` | rows/cols ordered like a staircase | the classic gradient structure |
  | `smooth` | adjacent occupied tiles have similar values | neighbours can merge |
  | `corner` | max tile sits in a corner | the cornerstone strategy |
  | `merge` | adjacent equal tiles ready to merge | builds big tiles faster |

  All start with a `+1` prior; the experiment learns which ones the bandit actually keeps
  (high `q`, many pulls) and which it discards.

## What gets logged (treatment only)

- `aux_bandit_history.jsonl` — per bandit update: `beta`, `q_main`, `q_<aux>` for each aux,
  `advantage`, `pulls_*`. **Which auxes earn weight = good vs bad identification.**
- `aux_divergence.jsonl` — per update: `js_<aux>` / `tv_<aux>`, the Jensen-Shannon / total-
  variation distance between that aux's parent-selection distribution and the oracle's,
  over the current archive. **~0 = the aux re-picks the oracle's parents (a no-op); higher
  = it genuinely steers selection elsewhere.** (Residualization pushes this up on purpose.)

## Run it

```bash
# vLLM must be serving Qwen on :8000 (gen) and :8001 (embed)
bash run_alg2_replicates.sh 3     # 3 reps/arm, treatment+control interleaved, 200 gens each
```

## Analyze

```bash
python plot_aux_weights.py     results/treatment_rep1   # q per aux + beta (good vs bad)
python plot_aux_divergence.py  results/treatment_rep1   # JS divergence per aux over search
python test_aux_evals.py                                # 23 checks on auxes + divergence math
```
