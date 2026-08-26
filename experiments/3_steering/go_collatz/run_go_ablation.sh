#!/usr/bin/env bash
set -uo pipefail
source /data/ikakkar/co-evolution/data_env.sh
export PATH=/data/ikakkar/toolchains/go/bin:$PATH GOCACHE=/data/ikakkar/.cache/go-build GOPATH=/data/ikakkar/go GOMODCACHE=/data/ikakkar/go/pkg/mod
cd /data/ikakkar/co-evolution/ShinkaEvolve/examples/go_collatz_steps; mkdir -p logs_ablation
UV="uv run --project /data/ikakkar/co-evolution/ShinkaEvolve python"; MAXCC=3
echo "GO ABLATION started $(date)"
run_one(){ echo ">>> $1 $(date +%H:%M:%S)"; $UV run_evo_local.py $2 --run_tag "$1" > "logs_ablation/$1.log" 2>&1; echo "<<< DONE $1 $(date +%H:%M:%S)"; }
jobs=(); for r in 1 2 3 4 5; do jobs+=("abl_base_rep$r|" "abl_steer_rep$r|--steer"); done
running=0; for j in "${jobs[@]}"; do run_one "${j%%|*}" "${j##*|}" & running=$((running+1)); [ "$running" -ge "$MAXCC" ] && { wait -n; running=$((running-1)); }; done; wait
echo "==== GO ABLATION COMPLETE $(date) ===="
