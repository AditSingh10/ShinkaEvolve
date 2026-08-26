# Circle Packing Family-Model Pilot

## Experiment

- Generation model: `local/qwen32b`; embedding model: `local/qwen-embed`
- Proposal budget: 200
- K_min: 3
- Warmup budget: 40 proposals
- tau_fam: 0.85
- Entropy window: 25 parent selections
- Seed: 104729

## Warmup Result

Warmup was partially successful. It seeded F1 from the initial program and discovered a second, mechanically meaningful population-based family at proposal 13, but it exhausted all 40 warmup proposals without reaching K_min=3.

| Family | Birth proposal | Representative mechanism | Size |
|---|---:|---|---:|
| F1 | 0 (initial seed) | Single-state simulated annealing with cooling and adaptive local perturbations | 26 |
| F2 | 13 | Population genetic search combined with simulated-annealing refinement | 30 |

## Representative Families

Representatives are the actual family members maximizing cosine similarity to their final family centroid; they were not hand-picked.

| Family | Representative | Compact mechanism | Own cosine | F1 centroid | F2 centroid | Closest other | Representative score | Best program / score |
|---|---|---|---:|---:|---:|---|---:|---|
| F1 | P182 | Randomly initialized single-state annealing; temperature- and step-scaled perturbations; Metropolis acceptance; pairwise radius shrinking | 0.971897 | 0.971897 | 0.899675 | F2 (0.899675) | 1.545102 | P170 / 2.422522 |
| F2 | P90 | Elite population selection, crossover, mutation, and temperature-scaled annealing refinement | 0.995726 | 0.896055 | 0.995726 | F1 (0.896055) | 2.232060 | P129 / 2.349860 |

The own-versus-closest-other margins are 0.072222 for F1 and 0.099671 for F2. The representatives are separated at the centroid level, although P39 is a secondary borderline/centroid-drift case: it was assigned to F2 but is closer to the final F1 centroid (0.925103 versus 0.891767).

## Diversity

Final K was 2, with H_search = 0.998846 and N_eff = 1.998400. The final 25-parent search window was split F1 13/25 (52%) and F2 12/25 (48%), so coverage of the two discovered families remained balanced after warmup. This high entropy indicates balance over two families, not discovery of many families.

## Performance

The final best was P170 at proposal 170 with score 2.422522. F2 had the higher median score (2.261697 versus F1's 1.861480), while F1 had the higher best score (2.422522 versus 2.349860). Cross-family donors produced several large mechanism-transfer improvements from weak parents, but only 18/89 cross-family proposals were valid, only 4 of those improved their parent, and none established a new global best. P170 is byte-identical to earlier donor P137, so the final update was a stochastic reevaluation rather than a new implementation.

## Validity

Overall validity was 55/200 (27.5%): 18/40 (45.0%) during warmup and 37/160 (23.125%) afterward. The dominant failures were 82 overlap violations and 50 timeouts, together accounting for 132/145 invalid proposals; interface, Python/runtime, and NumPy/library failures totaled only 13. Low validity is therefore mainly a geometric-feasibility and runtime problem, not a syntax/interface problem.

## Main Takeaways

- Warmup found two meaningful high-level mechanisms: single-state/local search and population-based evolutionary search.
- It did not reach K_min=3, so family seeding was only partially successful.
- Search mass remained balanced over the two families instead of collapsing after warmup.
- Both representatives are closer to their own centroid, with margins of 0.072222 and 0.099671, although F1 is broad and P39 is a borderline assignment.
- The main limitation is low validity from overlaps and timeouts; under-resolved family structure and duplicate stochastic reevaluations are secondary concerns.
