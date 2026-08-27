# Autoresearch Example

This example evolves lightweight byte-level next-token predictors. Candidates
minimize evaluator-computed validation cross-entropy and bits-per-byte under a
small deterministic CPU budget. No GPU, dataset download, or ML framework is
required.

Each candidate receives training bytes, independent validation contexts, the
fixed 256-byte vocabulary, a seed, and resource limits. Validation target bytes
are retained by the evaluator:

```python
def run_autoresearch(instance, seed):
    return {
        "probabilities": [
            [1.0 / 256] * 256
            for _ in instance["validation_contexts"]
        ]
    }
```

Candidates may instead return `log_probabilities`. There must be exactly one
finite, normalized 256-value row per validation context. Probability rows must
be nonnegative and sum to one; log-probability rows must have log-sum-exp zero.
Candidate metadata is ignored.

Candidate calls run in disposable child processes. The evaluator enforces the
per-instance timeout, detects candidate-visible input mutation, and computes
loss and score only after predictions return to the parent process. Invalid
outputs, exceptions, timeouts, mutation, and zero probability on a target all
receive score zero.

Run the baseline evaluator from this directory:

```bash
python evaluate.py --program_path initial.py --results_dir results/manual
```

Run evolution with:

```bash
python run_evo.py
```

The baseline uses longest-suffix byte counts with unigram fallback. Evolution
can explore interpolated n-grams, PPM/context trees, suffix matching, explicit
pattern detection, tiny CPU-trained models, batching and optimizer strategies,
and statistical-neural hybrids.
