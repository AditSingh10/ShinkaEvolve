#!/usr/bin/env bash
# Serve Qwen2.5-32B-Instruct (general, AWQ 4-bit) for ShinkaEvolve.
# General model chosen because the pipeline does more than code (novelty judging,
# textual-feedback reasoning, optional meta-analysis), not only program mutation.
# OpenAI-compatible endpoint at http://localhost:8000/v1  (served model name: qwen32b)
# Uses GPUs 0,1 (tensor-parallel 2). Weights cached under /data/ikakkar/hf_cache.
set -euo pipefail

export HF_HOME=/data/ikakkar/hf_cache
export CUDA_VISIBLE_DEVICES=0,1

source /data/ikakkar/venvs/vllm/bin/activate

exec vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --served-model-name qwen32b \
  --tensor-parallel-size 2 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
  # quantization (AWQ) is auto-detected from the model config; vLLM picks the Marlin kernel on Ampere.
