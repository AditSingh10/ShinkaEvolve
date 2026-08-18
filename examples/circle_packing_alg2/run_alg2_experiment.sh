#!/usr/bin/env bash
# ALG2 main experiment: treatment (aux-guided selection) then control (oracle-only),
# 200 generations each on the local Qwen3 server. ~8h per arm, ~15-16h total.
# Self-cds, so it can be launched from anywhere. Intended to run inside tmux.
set -uo pipefail
cd "$(dirname "$0")"
source /data/ikakkar/venvs/shinka/bin/activate
export PILOT_AUX=none          # ALG2 uses SELECTION, not feedback notes

# ---- preflight: fail fast rather than waste hours ------------------------------
for p in 8000 8001; do
  curl -s -m5 "http://localhost:$p/v1/models" >/dev/null 2>&1 \
    || { echo "FATAL: no vLLM server on :$p  (run serving/switch_model.sh 3)"; exit 1; }
done
MODEL=$(curl -s -m5 http://localhost:8000/v1/models | python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")

# ---- context-budget preflight ---------------------------------------------------
# A previous run wasted ~9h because max_model_len(16384) - max_tokens(12288) left only
# 4096 tokens for the prompt; once evolved programs grew past that, EVERY request 400-ed
# and the loop just retried forever. Fail fast instead.
python - <<'PY' || exit 1
import json, sys, urllib.request, yaml
mml = json.load(urllib.request.urlopen("http://localhost:8000/v1/models"))["data"][0]["max_model_len"]
mt = yaml.safe_load(open("shinka_alg2.yaml"))["evo_config"]["llm_kwargs"]["max_tokens"]
head = mml - mt
print(f"context budget: max_model_len={mml}  max_tokens={mt}  -> {head} tokens for the prompt")
if head < 8000:
    print(f"FATAL: only {head} tokens left for prompts; evolved programs will overflow it.\n"
          f"       Raise --max-model-len in serving/serve_qwen3_gen.sh or lower max_tokens.")
    sys.exit(1)
PY

echo "preflight OK | gen model=$MODEL | started $(date)"
echo "NOTE: expects Qwen3 to be the served model (serving/switch_model.sh 3)."

run_arm () {                    # $1 = config, $2 = tag
  local cfg="$1" tag="$2"
  echo ""
  echo "================ ARM: $tag  ($(date)) ================"
  rm -rf "results/pilot_none_${tag}"
  python run_evo.py --config_path "$cfg" --run_tag "$tag"
  echo "---- $tag finished with exit $? at $(date) ----"
}

run_arm shinka_alg2.yaml         treatment
run_arm shinka_alg2_control.yaml control

echo ""
echo "================ ALL DONE $(date) ================"
echo "results/pilot_none_treatment  (aux-guided selection)"
echo "results/pilot_none_control    (oracle-only selection)"
echo "bandit trace: results/pilot_none_treatment/aux_bandit_history.jsonl"
