#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:?usage: watch_train_then_benchmark.sh TRAIN_PID CHECKPOINT_DIR [REPORT_STEM]}"
CHECKPOINT_DIR="${2:?usage: watch_train_then_benchmark.sh TRAIN_PID CHECKPOINT_DIR [REPORT_STEM]}"
REPORT_STEM="${3:-results/suitesparse_eval_hdrive20_train_excluded_robust_v1}"

REPORT_JSON="${REPORT_STEM}.json"
REPORT_MD="${REPORT_STEM}.md"
REPORT_CSV="${REPORT_STEM}.csv"
REPORT_LOG="${REPORT_STEM}.log"
PARTIAL_JSONL="${REPORT_STEM}.partial.jsonl"

mkdir -p results

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 60
done

conda run --no-capture-output -n pyg_env python scripts/evaluate_neuro_ilu_suitesparse.py \
  --checkpoint "${CHECKPOINT_DIR}/model/best_val.pt" \
  --selected-json results/selected_eval_matrices_hdrive_train_excluded.json \
  --max-iter 2000 \
  --rtol 1e-8 \
  --spilu-drop-tol 1e-4 \
  --spilu-fill-factor 10.0 \
  --output "${REPORT_JSON}" \
  --partial-jsonl "${PARTIAL_JSONL}" > "${REPORT_LOG}" 2>&1

conda run --no-capture-output -n pyg_env python scripts/export_suitesparse_benchmark.py \
  --input "${REPORT_JSON}" \
  --markdown-out "${REPORT_MD}" \
  --csv-out "${REPORT_CSV}"
