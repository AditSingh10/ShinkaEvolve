#!/usr/bin/env bash
set -uo pipefail
source /data/ikakkar/co-evolution/data_env.sh
export PATH=/data/ikakkar/toolchains/julia-1.10.5/bin:$PATH JULIA_DEPOT_PATH=/data/ikakkar/.julia
cd /data/ikakkar/co-evolution/ShinkaEvolve/examples/julia_prime_counting; mkdir -p logs_ablation
UV="uv run --project /data/ikakkar/co-evolution/ShinkaEvolve python"; MAXCC=3
echo "JULIA ABLATION started $(date)"
run_one(){ echo ">>> $1 $(date +%H:%M:%S)"; $UV run_evo_local.py $2 --run_tag "$1" > "logs_ablation/$1.log" 2>&1; echo "<<< DONE $1 $(date +%H:%M:%S)"; }
jobs=(); for r in 1 2 3 4 5; do jobs+=("abl_base_rep$r|" "abl_steer_rep$r|--steer"); done
running=0; for j in "${jobs[@]}"; do run_one "${j%%|*}" "${j##*|}" & running=$((running+1)); [ "$running" -ge "$MAXCC" ] && { wait -n; running=$((running-1)); }; done; wait
echo "==== JULIA ABLATION COMPLETE $(date) ===="
