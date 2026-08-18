#!/usr/bin/env bash
# Flip the generation model on port 8000 between Qwen2.5 (fast, functionality) and
# Qwen3 (reasoning, performance). Both register as `qwen32b`, so shinka_qwen.yaml is unchanged.
# The embedding server on GPU 2 / port 8001 is left untouched.
#
# Usage: bash switch_model.sh [2.5|3]
set -uo pipefail
cd "$(dirname "$0")"
TARGET="${1:-}"
[[ "$TARGET" == "2.5" || "$TARGET" == "3" ]] || { echo "usage: switch_model.sh [2.5|3]"; exit 1; }

echo ">> stopping current gen server (embed on GPU 2 is preserved)..."
# Kill any compute process on the generation GPUs (everything EXCEPT GPU 2 = embed).
for p in $(nvidia-smi -i 0,1,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null || true
done
# Also kill the parent vllm-serve / EngineCore procs for either gen model.
pkill -9 -f "vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ" 2>/dev/null || true
pkill -9 -f "vllm serve Qwen/Qwen3-32B" 2>/dev/null || true
sleep 5

if [[ "$TARGET" == "2.5" ]]; then SCRIPT=serve_qwen_gen.sh; else SCRIPT=serve_qwen3_gen.sh; fi
echo ">> launching $SCRIPT (detached)..."
setsid nohup bash "$SCRIPT" > gen_server.log 2>&1 < /dev/null &

echo ">> waiting for port 8000..."
t=0
until curl -s -m3 localhost:8000/v1/models >/dev/null 2>&1; do
  sleep 10; t=$((t+10))
  if grep -qiE "out of memory|Engine core init.*failed|Worker failed" gen_server.log 2>/dev/null; then
    echo "!! startup error -- see gen_server.log"; exit 1
  fi
  [ $t -ge 300 ] && { echo "!! timeout after ${t}s -- see gen_server.log"; exit 1; }
done
echo ">> UP after ~${t}s. Serving Qwen$TARGET as 'qwen32b' on :8000."
