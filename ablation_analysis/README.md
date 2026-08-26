# AuxEvolve Family-Model Ablation

## Goal

This ablation tests the AuxEvolve Family Model independently from validity steering and other auxiliary mechanisms. It isolates semantic family discovery, family-conditioned sampling, and the explicit family-seeding warmup.

## Setup

- Task: Circle Packing (26 circles in a unit square)
- Generation and summarization: Qwen2.5-32B-Instruct
- Family-summary embeddings: Qwen3-Embedding-0.6B
- No validity steering, repair/retry, novelty rejection, or auxiliary selection
- Proposal accounting: one generated candidate is one proposal

Local OpenAI-compatible generation and embedding endpoints are configured in `base.yaml`. `run_evo.py` runs any ablation condition; `run_ablation.sh` launches paired conditions and seeds; `run_frozen_full_once.py` reproduces the frozen B=200 pilot configuration.

## Family Model

Valid programs are converted into detailed mechanistic summaries and embedded. Each program is assigned online to its nearest family centroid when its cosine similarity passes `tau_fam`; otherwise it seeds a new family. Warmup explicitly requests structurally different mechanisms until `K_min` is reached or the warmup budget is exhausted.

Family quality is the median objective of admitted members. During normal search, the family distribution transitions from uniform coverage toward quality-weighted sampling:

$$
\pi_f(t)=(1-t)\frac{1}{K}+t\,\mathrm{Softmax}(Q)_f.
$$

Parent-family and donor-family draws are independent. Search-mass entropy and effective family count are diagnostics only and never alter sampling or phase transitions.

The seven conditions isolate warmup, online observation, parent-family sampling, donor-family sampling, family context in prompts, and their full combination. Generated run artifacts are written below `results/` and are intentionally git-ignored.

## Pilot Result

The completed Circle Packing pilot used B=200, `K_min=3`, a 40-proposal warmup, `tau_fam=0.85`, a 25-selection entropy window, and seed 104729. F2 was discovered at proposal 13, but the run finished at K=2 and did not reach `K_min`. Final search diagnostics were $H_{\text{search}}\approx0.999$ and $N_{\text{eff}}\approx1.998$, with recent family search mass approximately 52%/48%. The best score was 2.4225 and validity was 27.5%; overlap violations and timeouts dominated invalid proposals.

The pilot suggests that the family sampler maintained balanced search mass once multiple families existed, while family discovery/clustering remains under-resolved and validity is a major bottleneck.

## Checks

From `ablation_analysis/`:

```bash
python3 -m unittest -v test_family_model.py
python3 -m py_compile family_model.py representation.py run_evo.py run_warmup_probe.py run_frozen_full_once.py evaluator_entry.py
bash -n run_ablation.sh
```
