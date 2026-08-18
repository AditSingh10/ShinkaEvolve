#!/usr/bin/env bash
# Validity-ONLY steering experiment (diversity OFF) for circle packing, in its own folder.
# Hypothesis (from the ablation forensics): validity steering alone -> higher valid-yield
# WITHOUT the ceiling collapse that diversity/COMBINED caused.
# Compare against the existing matched baselines: circle_packing_alg2/results/pilot_none_abl_base_rep*
set -uo pipefail
source /data/ikakkar/co-evolution/data_env.sh
DIR=/data/ikakkar/co-evolution/ShinkaEvolve/examples/circle_packing_steer_valonly
cd "$DIR"; mkdir -p logs
UV="uv run --project /data/ikakkar/co-evolution/ShinkaEvolve python"
MAXCC=3
for pt in 8000 8001; do curl -s -m5 "http://localhost:$pt/v1/models" >/dev/null 2>&1 || { echo "FATAL no server :$pt"; exit 1; }; done
echo "VALIDITY-ONLY experiment started $(date) — 5 reps, max $MAXCC concurrent"

run_one(){ # $1 tag
  echo ">>> START $1 $(date +%H:%M:%S)"
  env PILOT_AUX=none $UV run_evo.py --config_path shinka_valonly.yaml --run_tag "$1" > "logs/$1.log" 2>&1
  echo "<<< DONE  $1 $(date +%H:%M:%S)"
}
running=0
for r in 1 2 3 4 5; do
  run_one "valonly_rep$r" &
  running=$((running+1))
  if [ "$running" -ge "$MAXCC" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "==== VALIDITY-ONLY EXPERIMENT COMPLETE $(date) ===="
