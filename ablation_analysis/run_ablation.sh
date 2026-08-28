#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
PROPOSALS="${PROPOSALS:-200}"
K_MIN="${K_MIN:-5}"
WARMUP_BUDGET="${WARMUP_BUDGET:-40}"
ENTROPY_WINDOW="${ENTROPY_WINDOW:-25}"
CONDITIONS=(baseline observe warmup parent inspiration prompt full_annealed full_create)
# Paired replicate seeds: every condition receives the same seed for a replicate.
SEEDS=(${SEEDS:-104729 130363 155921 181081 206369})

mkdir -p "$HERE/logs"
for port in 8000 8001; do
  curl --noproxy '*' --fail --silent --max-time 5 "http://127.0.0.1:${port}/v1/models" >/dev/null || {
    echo "Model endpoint unavailable on port ${port}" >&2; exit 1;
  }
done

run_one() {
  local condition="$1" replicate="$2" seed="$3"
  echo ">>> ${condition}/rep${replicate}/seed${seed}"
  PYTHONPATH="$REPO:$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$HERE/run_evo.py" \
    --condition "$condition" --replicate "$replicate" --seed "$seed" \
    --proposals "$PROPOSALS" --k-min "$K_MIN" --warmup-budget "$WARMUP_BUDGET" \
    --entropy-window "$ENTROPY_WINDOW" \
    >"$HERE/logs/${condition}_rep${replicate}_seed${seed}.log" 2>&1
  echo "<<< ${condition}/rep${replicate}/seed${seed}"
}

running=0
for index in "${!SEEDS[@]}"; do
  replicate=$((index + 1)); seed="${SEEDS[$index]}"
  for condition in "${CONDITIONS[@]}"; do
    run_one "$condition" "$replicate" "$seed" & running=$((running + 1))
    if ((running >= MAX_CONCURRENT)); then wait -n; running=$((running - 1)); fi
  done
done
wait
