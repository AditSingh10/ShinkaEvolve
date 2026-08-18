#!/usr/bin/env bash
set -uo pipefail
source /data/ikakkar/co-evolution/data_env.sh
cd /data/ikakkar/co-evolution/ShinkaEvolve/examples/game_2048_alg2; mkdir -p logs_ablation
UV="uv run --project /data/ikakkar/co-evolution/ShinkaEvolve python"; MAXCC=3
echo "2048 ABLATION started $(date)"
run_one(){ echo ">>> $2 $(date +%H:%M:%S)"; env PILOT_AUX=none $UV run_evo.py --config_path "$1" --run_tag "$2" > "logs_ablation/$2.log" 2>&1; echo "<<< DONE $2 $(date +%H:%M:%S)"; }
jobs=(); for r in 1 2 3 4 5; do jobs+=("shinka_alg2.yaml|abl_base_rep$r" "shinka_steer_full.yaml|abl_steer_rep$r"); done
running=0; for j in "${jobs[@]}"; do run_one "${j%%|*}" "${j##*|}" & running=$((running+1)); [ "$running" -ge "$MAXCC" ] && { wait -n; running=$((running-1)); }; done; wait
echo "==== 2048 ABLATION COMPLETE $(date) ===="
