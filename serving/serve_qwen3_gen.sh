#!/usr/bin/env bash
# Serve Qwen3-32B (BF16, THINKING enabled) for ShinkaEvolve.
# Qwen3 is a reasoning model: it emits <think>...</think> then the answer. With
# --reasoning-parser qwen3, vLLM puts the reasoning in message.reasoning_content and the
# FINAL answer in message.content, so ShinkaEvolve (which reads content) sees only the answer.
# OpenAI-compatible endpoint at http://localhost:8000/v1  (served model name: qwen32b)
# BF16 ~65GB -> tensor-parallel 4 on GPUs 0,1,3,4 (GPU 2 is the embedding server).
set -euo pipefail

export HF_HOME=/data/ikakkar/hf_cache
export CUDA_VISIBLE_DEVICES=0,1,3,4
# RTX A5000 (GA102) has broken/disabled GPU P2P over PCIe, which deadlocks NCCL during
# tensor-parallel init ("No available shared memory broadcast block"). Force NCCL to use
# shared-memory/staging instead of P2P. Required for TP>1 on these cards.
export NCCL_P2P_DISABLE=1

source /data/ikakkar/venvs/vllm/bin/activate

exec vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen32b \
  --tensor-parallel-size 4 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce \
  --enforce-eager
  # BF16 32B is ~16GB weights/GPU on 22GB A5000s -> little headroom. --enforce-eager skips
  # CUDA-graph capture (the OOM culprit).
  # CONTEXT BUDGET: max_model_len must cover prompt + max_tokens. At 16384 with
  # max_tokens=12288 only 4096 remained for the prompt; once evolved programs grew past
  # that, EVERY request failed with a 400 and the loop retried forever (~9h lost).
  # The prompt holds THREE programs (parent + 2 inspirations). The largest program seen
  # across past runs is 33KB = 11,661 Qwen tokens, so the worst case is ~35k prompt.
  # 40960 is the model's architectural cap (max_position_embeddings); 40960-12288 = 28,672
  # tokens of prompt headroom, i.e. programs up to ~9.5k tokens (~27KB) each.
  # Note max_model_len does NOT reserve KV per request (vLLM allocates blocks on demand),
  # so raising it barely affects concurrency; it must only fit inside the KV pool
  # (~256 KiB/token, pool ~16GB => ~61k tokens, so 49152 is safe).
