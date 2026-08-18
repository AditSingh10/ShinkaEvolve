#!/usr/bin/env bash
# Matched ablation: does event-triggered steering help?  (spec §22 endpoint)
#   A = baseline  : shinka_alg2.yaml       (3-aux, NO steering)
#   E = steering  : shinka_steer_full.yaml (3-aux + all controllers, spec-default warmup=20)
# Same everything else (200 gens, Qwen2.5, aux caging/hole/connect), interleaved so any GPU
# drift hits both arms equally. n=5 reps/arm. Both use the updated checker -> valid-yield AND
# runtime-crash rate are measurable.
set -uo pipefail
source /data/ikakkar/co-evolution/data_env.sh
BASE=/data/ikakkar/co-evolution/ShinkaEvolve/examples/circle_packing_alg2
cd "$BASE"; mkdir -p logs_ablation
UV="uv run --project /data/ikakkar/co-evolution/ShinkaEvolve python"
MAXCC=3   # concurrent runs (all share the one qwen server; fair to both arms)

for pt in 8000 8001; do curl -s -m5 "http://localhost:$pt/v1/models" >/dev/null 2>&1 || { echo "FATAL no server :$pt"; exit 1; }; done
echo "STEER ABLATION started $(date) — 5 reps x {baseline, steering}, max $MAXCC concurrent"

run_one(){ # $1 config  $2 tag
  echo ">>> START $2 $(date +%H:%M:%S)"
  env PILOT_AUX=none $UV run_evo.py --config_path "$1" --run_tag "$2" > "logs_ablation/$2.log" 2>&1
  echo "<<< DONE  $2 $(date +%H:%M:%S)"
}

jobs=()
for r in 1 2 3 4 5; do
  jobs+=("shinka_alg2.yaml|abl_base_rep$r")
  jobs+=("shinka_steer_full.yaml|abl_steer_rep$r")
done

running=0
for j in "${jobs[@]}"; do
  cfg="${j%%|*}"; tag="${j##*|}"
  run_one "$cfg" "$tag" &
  running=$((running+1))
  if [ "$running" -ge "$MAXCC" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "==== STEER ABLATION COMPLETE $(date) ===="
