#!/usr/bin/env bash
# ALG2 game_2048 replicate study: N runs per arm, treatment (aux-guided selection) vs
# control (oracle-only). Arms INTERLEAVED (t1,c1,t2,c2,...) so stopping early still leaves
# balanced pairs. Self-cds; intended for tmux/detached use.
#
# Usage: bash run_alg2_replicates.sh [N_REPS]      (default 3)
set -uo pipefail
cd "$(dirname "$0")"
source /data/ikakkar/venvs/shinka/bin/activate
REPS="${1:-3}"

# ---- preflight: vLLM servers up + context budget ---------------------------------
for p in 8000 8001; do
  curl -s -m5 "http://localhost:$p/v1/models" >/dev/null 2>&1 \
    || { echo "FATAL: no vLLM server on :$p"; exit 1; }
done
python - <<'PY' || exit 1
import json, sys, urllib.request, yaml
mml = json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))["data"][0]["max_model_len"]
mt = yaml.safe_load(open("shinka_alg2.yaml"))["evo_config"]["llm_kwargs"]["max_tokens"]
print(f"context budget: max_model_len={mml} max_tokens={mt} -> {mml-mt} for prompt")
sys.exit(0 if mml-mt >= 8000 else 1)
PY
echo "preflight OK | $REPS replicates per arm | started $(date)"

run_one () {                     # $1 = config, $2 = tag
  echo ""
  echo "======== $2  ($(date +%H:%M:%S)) ========"
  rm -rf "results/$2"
  python run_evo.py --config_path "$1" --run_tag "$2"
  echo "-------- $2 done exit=$? $(date +%H:%M:%S) --------"
}

for i in $(seq 1 "$REPS"); do
  run_one shinka_alg2.yaml         "treatment_rep$i"
  run_one shinka_alg2_control.yaml "control_rep$i"
done

echo ""
echo "======== ALL REPLICATES DONE $(date) ========"
