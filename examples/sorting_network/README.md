# Sorting Network Synthesis Example

This example evolves fixed, layered comparator networks. Correctness is
mandatory; valid candidates optimize either depth before comparator count or
comparator count before depth.

Each candidate receives an instance dictionary and deterministic seed:

```python
def run_sorting_network(instance, seed):
    return {
        "network": [
            [(0, 1), (2, 3)],
            [(1, 2)],
            [(0, 1), (2, 3)],
        ]
    }
```

The outer list contains layers. Each nonempty layer contains disjoint
comparators, and each comparator is a canonical pair `(i, j)` satisfying
`0 <= i < j < n_wires`. The candidate is called once per instance, producing
one data-independent network that is then applied to every Boolean input.

The evaluator exhaustively checks all `2 ** n_wires` Boolean vectors and
recomputes depth, comparator count, failed-input count, counterexamples, and
score. Candidate metadata is ignored, and mutation of the input instance is
rejected. Invalid or incorrect networks receive score zero.

Run the baseline evaluator from this directory:

```bash
python evaluate.py --program_path initial.py --results_dir results/manual
```

Run evolution with:

```bash
python run_evo.py
```

The baseline is an odd-even transposition network. No SAT solver or other
external optimization dependency is required. Evolution can improve it with
bitonic or odd-even merge constructions, pruning, dependency-safe compaction,
counterexample-guided repair, bounded search, and hybrid strategies.
