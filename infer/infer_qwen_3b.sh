#!/usr/bin/env bash
set -e

# ================= Dataset =================
DATASET="${1:-csqa}"

CUDA_VISIBLE_DEVICES=6

# ================= Model =================
MODEL_DIR="/ssd/common/LLMs/Qwen2.5-3B-Instruct"
MODEL_TAG="$(basename "${MODEL_DIR%/}")"   # Qwen2.5-1.5B-Instruct

# ================= Data =================
INPUT_JSONL="./data/${DATASET}_4options.jsonl"

OUT_DIR="./data"
mkdir -p "${OUT_DIR}"

OUT_WRONG="${OUT_DIR}/${DATASET}_wrong_${MODEL_TAG}.jsonl"
OUT_CORRECT="${OUT_DIR}/${DATASET}_correct_${MODEL_TAG}.jsonl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

echo "================ Infer & Split ================="
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_DIR  : ${MODEL_DIR}"
echo "INPUT      : ${INPUT_JSONL}"
echo "OUT_WRONG  : ${OUT_WRONG}"
echo "OUT_CORRECT: ${OUT_CORRECT}"
echo "================================================"

python infer/infer.py \
  --model_name "${MODEL_DIR}" \
  --jsonl "${INPUT_JSONL}" \
  --out_wrong_jsonl "${OUT_WRONG}" \
  --out_correct_jsonl "${OUT_CORRECT}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}"

echo "Infer & split done."
