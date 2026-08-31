# Autoresearch

This example connects AuxEvolve to Andrej Karpathy's official Autoresearch
benchmark. AuxEvolve evolves the complete upstream `train.py`; each valid
candidate trains and evaluates a language model on one NVIDIA GPU. The research
objective is the best validation bits per byte (`val_bpb`) achievable under the
official five-minute training budget. Lower `val_bpb` is better.

The adapter pins upstream commit
`228791fb499afffb54b46200aca536f79142f117`. The adapter uses an external
checkout and verifies its revision and protected runtime files. This keeps
upstream provenance clear and runtime state outside the AuxEvolve checkout.
The pinned upstream README labels the project MIT, but that tree does not
contain a standalone `LICENSE` file.

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- one NVIDIA GPU supported by the pinned upstream environment (upstream
  documents H100 testing)
- enough disk space for the upstream environment, dataset, and tokenizer

Run the commands below from the AuxEvolve repository root. Choose an absolute
location outside this repository for the upstream checkout:

```bash
export AUTORESEARCH_ROOT=/absolute/path/to/autoresearch-upstream
```

## One-time setup

You run setup as a separate step. Candidate evaluation does not clone source,
install dependencies, or download data.

```bash
python examples/autoresearch/setup_upstream.py clone --root "$AUTORESEARCH_ROOT"
python examples/autoresearch/setup_upstream.py sync --root "$AUTORESEARCH_ROOT"
python examples/autoresearch/setup_upstream.py prepare --root "$AUTORESEARCH_ROOT"
export CUDA_VISIBLE_DEVICES=0
python examples/autoresearch/setup_upstream.py kernel --root "$AUTORESEARCH_ROOT"
python examples/autoresearch/setup_upstream.py check --root "$AUTORESEARCH_ROOT"
```

These commands clone the official repository, detach `HEAD` at the pinned
commit, install exactly the locked environment with `uv sync --frozen`, and run
the pinned `prepare.py`. The `kernel` step downloads the Flash Attention kernel
variant for the assigned GPU. The final check is read-only: it verifies the
commit, protected-file hashes, virtual-environment interpreter, runtime imports,
cached GPU kernel, and expected dataset/tokenizer artifacts. Its probe uses
`uv run --frozen --no-sync` with `HF_HUB_OFFLINE=1`, so it cannot repair an
incomplete setup or access the Hub.

The pinned `prepare.py` uses `~/.cache/autoresearch` for data and tokenizer
artifacts.

## Establish the baseline

Assign exactly one GPU, then run the untouched pinned `train.py`:

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/autoresearch/setup_upstream.py baseline --root "$AUTORESEARCH_ROOT"
```

This upstream training run takes about five minutes plus startup, compilation,
final evaluation, and cleanup. Record its raw `val_bpb` before starting
evolution.

To exercise the AuxEvolve evaluator on that same baseline:

```bash
python examples/autoresearch/evaluate.py \
  --program_path "$AUTORESEARCH_ROOT/train.py" \
  --upstream-root "$AUTORESEARCH_ROOT" \
  --results_dir /absolute/path/to/autoresearch-eval-results \
  --timeout-seconds 600
```

The evaluator writes `metrics.json`, `correct.json`, `candidate_stdout.log`,
and `candidate_stderr.log`. It stores raw `val_bpb` under `public` in
`metrics.json`. Because AuxEvolve maximizes fitness, the internal score is

```text
combined_score = 1 / (1 + val_bpb)
```

This monotonic transformation changes only score direction. Report and compare
raw `val_bpb` in experiments and papers.

## Run evolution

With the prepared checkout and one assigned GPU:

```bash
export AUTORESEARCH_ROOT=/absolute/path/to/autoresearch-upstream
export CUDA_VISIBLE_DEVICES=0
python examples/autoresearch/run_evo.py
```

The runner seeds evolution from the pinned upstream `train.py` and permits one
evaluation job at a time. Local AuxEvolve jobs do not allocate GPUs. To use
multiple GPUs, run one configured runner process for each assigned GPU or use a
GPU-aware scheduler; do not expose multiple comma-separated devices to a single
runner.

## What is fixed and what can evolve

The candidate may change all of `train.py`, including model architecture,
attention, optimizers, normalization, schedules, batching, efficiency changes,
and training strategy. The initial candidate is the official pinned file.

The following remain fixed:

- upstream commit and locked dependency environment
- the pinned seed `train.py`, plus `prepare.py`, `.python-version`,
  `pyproject.toml`, and `uv.lock` in the external checkout
- upstream data preparation, tokenizer, validation data, `evaluate_bpb`, and
  the 300-second training budget
- one visible GPU per runner

The evaluator hashes protected files before and after each candidate. It also
requires a zero exit status, a reported training time from 300 through 330
seconds, an evaluator-measured wall-clock runtime at least as long as the
reported training time, and one well-formed, finite, nonnegative official
`val_bpb` summary. The 30-second upper tolerance permits the official loop's
final step to cross the five-minute boundary without allowing another training
budget. Crashes, OOMs, timeouts, malformed or missing metrics, non-finite
metrics, prerequisite failures, and protected-file changes produce
`correct=false` and score zero. A 600-second inner evaluator timeout leaves room
for compilation and final evaluation; the local job has a 12-minute outer
timeout as a second bound. Both timeout layers terminate the original process
group and repeatedly discover detached descendants while the parent remains
observable.

## Trust boundary

This cooperative research benchmark provides no security sandbox. A candidate
is arbitrary Python running with the evaluator's user and GPU permissions. The
adapter rejects an instant process that claims 300 training seconds, but it
does not independently load a checkpoint and recompute model quality. A
candidate that consumes the required wall-clock time can still fabricate
`val_bpb`, matching the trust assumption of the official editable-`train.py`
workflow. Candidate execution forces Hugging Face Hub offline mode, but this
does not block other network clients. Run untrusted candidates inside an
isolated machine or container with restricted credentials and network access.
Host-level process traversal is best-effort: a candidate can detach a child,
redirect its file descriptors, and exit before the evaluator observes that
child. Use a cgroup, PID namespace, container, or dedicated machine when the
operator must guarantee cleanup before reusing the GPU.

## Reproducible comparisons

For a direct baseline-versus-AuxEvolve comparison, keep the pinned commit,
starting `train.py`, prepared dataset/tokenizer, GPU model, 300-second training
budget, and total research budget fixed. Hardware matters: the fixed wall-clock
objective optimizes throughput and model quality together, so results from
different GPU types are not comparable. Preserve the candidate logs and public
metrics for every run.

Ordinary repository tests use fake checkouts and short subprocesses; they do not
require upstream source, network access, PyTorch, a GPU, or a five-minute run.
Run the baseline/evolution integration step on compatible hardware.
