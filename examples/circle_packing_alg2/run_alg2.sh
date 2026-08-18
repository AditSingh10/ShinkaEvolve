#!/usr/bin/env bash
# ALG2 aux-guided parent selection run. Self-cds so it always runs in the right dir.
# Usage: bash run_alg2.sh [smoke|full] [tag]
set -uo pipefail
cd "$(dirname "$0")"
source /data/ikakkar/venvs/shinka/bin/activate
MODE="${1:-smoke}"
TAG="${2:-}"
export PILOT_AUX=none          # ALG2 uses selection, not feedback notes
ARGS=(--config_path shinka_alg2.yaml)
[ "$MODE" = "smoke" ] && ARGS+=(--smoke)
[ -n "$TAG" ] && ARGS+=(--run_tag "$TAG")
SUB="pilot_none${TAG:+_$TAG}"
[ "$MODE" = "smoke" ] && SUB="smoke_none${TAG:+_$TAG}"
rm -rf "results/${SUB}"
echo "ALG2 run: mode=$MODE tag=${TAG:-none} -> results/${SUB} (in $(pwd))"
python run_evo.py "${ARGS[@]}"
echo "ALG2 EXIT $?"
