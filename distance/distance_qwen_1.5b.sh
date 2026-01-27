#!/usr/bin/env bash
set -e

# ================= Dataset =================
DATASET="${1:-csqa}"

# ================= GPU =================
CUDA_VISIBLE_DEVICES=6
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# ================= Model =================
MODEL_NAME="/ssd/common/LLMs/Qwen2.5-1.5B-Instruct"
MODEL_TAG="$(basename "${MODEL_NAME%/}")"

# ================= Input data (from zero_ckpts) =================
BASE_DIR="./zero_ckpts/${DATASET}_${MODEL_TAG}_lr1e-7_usefreeze0/jsonl"

WRONG_JSONL="${BASE_DIR}/infer_wrong_ep25.jsonl"
CORRECT_JSONL="${BASE_DIR}/infer_correct_ep25.jsonl"

# ================= Output directory =================
OUT_DIR="./distance_results/${DATASET}_${MODEL_TAG}"
mkdir -p "${OUT_DIR}"

OUT_CSV="${OUT_DIR}/grad_dot_split_stats.csv"
OUT_JSONL="${OUT_DIR}/correct_with_grad_dot.jsonl"

# ================= Print configuration =================
echo "================ Distance Analysis ================="
echo "GPU               : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME        : ${MODEL_NAME}"
echo "MODEL_TAG         : ${MODEL_TAG}"
echo "WRONG_JSONL       : ${WRONG_JSONL}"
echo "CORRECT_JSONL     : ${CORRECT_JSONL}"
echo "OUT_CSV           : ${OUT_CSV}"
echo "OUT_JSONL         : ${OUT_JSONL}"
echo "OUT_DIR           : ${OUT_DIR}"
echo "==================================================="

# ================= Start analysis =================
python distance/distance.py \
  --model_name "${MODEL_NAME}" \
  --wrong_jsonl "${WRONG_JSONL}" \
  --correct_jsonl "${CORRECT_JSONL}" \
  --out_csv "${OUT_CSV}" \
  --out_jsonl "${OUT_JSONL}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}"
