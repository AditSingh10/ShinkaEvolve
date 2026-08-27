# AuxEvolve Family-Model Ablation

## Goal

The original seven conditions test the AuxEvolve Family Model independently from validity steering and other auxiliary mechanisms. They isolate semantic family discovery, family-conditioned sampling, and the explicit family-seeding warmup. An eighth `warmup_validity` condition adds a prompt-only validity controller while leaving the Family Model policy unchanged.

## Setup

- Task: Circle Packing (26 circles in a unit square)
- Generation and summarization: Qwen2.5-32B-Instruct
- Family-summary embeddings: Qwen3-Embedding-0.6B
- No repair/retry, novelty rejection, or auxiliary selection; validity steering is enabled only for `warmup_validity`
- Proposal accounting: one generated candidate is one proposal

Local OpenAI-compatible generation and embedding endpoints are configured in `base.yaml`. `run_evo.py` runs any ablation condition; `run_ablation.sh` launches the seven Family Model conditions and paired seeds; `run_frozen_full_once.py` reproduces the frozen B=200 Family Model pilot configuration.

## Family Model

Valid programs are converted into detailed mechanistic summaries and embedded. Each program is assigned online to its nearest family centroid when its cosine similarity passes `tau_fam`; otherwise it seeds a new family. Warmup explicitly requests structurally different mechanisms until `K_min` is reached or the warmup budget is exhausted.

Family quality is the median objective of admitted members. During normal search, the family distribution transitions from uniform coverage toward quality-weighted sampling:

$$
\pi_f(t)=(1-t)\frac{1}{K}+t\,\mathrm{Softmax}(Q)_f.
$$

Parent-family and donor-family draws are independent. Search-mass entropy and effective family count are diagnostics only and never alter sampling or phase transitions.

The seven Family Model conditions isolate warmup, online observation, parent-family sampling, donor-family sampling, family context in prompts, and their full combination. `warmup_validity` follows the warmup-only selection and family-routing path and can change only the generation prompt. Generated run artifacts are written below `results/` and are intentionally git-ignored.

## Family-Model Pilot Result

The completed Circle Packing pilot used B=200, `K_min=3`, a 40-proposal warmup, `tau_fam=0.85`, a 25-selection entropy window, and seed 104729. F2 was discovered at proposal 13, but the run finished at K=2 and did not reach `K_min`. Final search diagnostics were $H_{\text{search}}\approx0.999$ and $N_{\text{eff}}\approx1.998$, with recent family search mass approximately 52%/48%. The best score was 2.4225 and validity was 27.5%; overlap violations and timeouts dominated invalid proposals.

The pilot suggests that the family sampler maintained balanced search mass once multiple families existed, while family discovery/clustering remains under-resolved and validity is a major bottleneck.

## Prompt-Only Validity Pilot

`warmup_validity` maintains an EMA violation rate and a clipped dual weight for runtime, timeout, compile, wrong-answer, malformed-output, boundary, and overlap failures. At periodic checks it activates only after confirmed threshold crossings, injects recent evaluator witnesses into the prompt, and releases with hysteresis. It has no dependency on family probabilities, parent/donor selection, score weighting, family quality, or admission.

The completed seed-104729 Circle Packing pilot used B=200, `K_min=3`, a 40-proposal warmup, and `tau_fam=0.85`. It produced 150 valid and 50 invalid proposals (75% validity), reached K=2 with F2 born at proposal 5, and found a best score of 2.4100617793 at proposal 189. One overlap episode activated at proposal 35, injected prompts for proposals 36-145, and then released. Overlap represented 35/50 invalid proposals; its incidence was 12/20 before activation, 18/110 during steering, and 0/20 immediately after release. This is temporal evidence only: no exact matched B=200 warmup-only run exists, so causal validity, objective, and family-discovery effects remain unestimated. Generated event, result, and analysis files are intentionally excluded from version control.

## Checks

From `ablation_analysis/`:

```bash
python3 -m unittest -v test_family_model.py
python3 -m unittest -v test_validity_model.py
python3 -m py_compile family_model.py validity_model.py representation.py run_evo.py run_warmup_probe.py run_frozen_full_once.py evaluator_entry.py
bash -n run_ablation.sh
```
