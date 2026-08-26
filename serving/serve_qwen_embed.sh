#!/usr/bin/env bash
# Serve a local embedding model for ShinkaEvolve code-similarity dedup.
# OpenAI-compatible embeddings endpoint at http://localhost:8001/v1 (model name: qwen-embed)
# Uses GPU 2 only (tiny model). Weights cached under /data/ikakkar/hf_cache.
set -euo pipefail

export HF_HOME=/data/ikakkar/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=2

source /data/ikakkar/venvs/vllm/bin/activate

# vLLM 0.25.1 selects embedding/pooling mode via `--runner pooling`.
exec vllm serve Qwen/Qwen3-Embedding-0.6B \
  --served-model-name qwen-embed \
  --runner pooling \
  --port 8001 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.30
