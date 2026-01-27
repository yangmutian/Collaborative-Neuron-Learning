#!/usr/bin/env bash
set -e

# ================= Dataset =================
DATASET="${1:-csqa}"

# ================= GPU =================
CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# ================= Model =================
MODEL_NAME="/ssd/common/LLMs/Qwen2.5-1.5B-Instruct"
MODEL_TAG="$(basename "${MODEL_NAME%/}")"

# ================= Data =================
WRONG_JSONL="./data/${DATASET}_wrong_${MODEL_TAG}.jsonl"
CORRECT1_JSONL="./data/${DATASET}_correct1_${MODEL_TAG}.jsonl"
CORRECT2_JSONL="./data/${DATASET}_correct2_${MODEL_TAG}.jsonl"

# ================= Training hyperparameters =================
LR=1e-7
EPOCHS=25

# ================= Output directory =================
OUT_DIR="./ood/${DATASET}_wrong_correct_sgd_${MODEL_TAG}_lr${LR}"
mkdir -p "${OUT_DIR}"

# ================= Print configuration =================
echo "================ Experiment ================="
echo "GPU            : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME     : ${MODEL_NAME}"
echo "WRONG_JSONL    : ${WRONG_JSONL}"
echo "CORRECT1_JSONL : ${CORRECT1_JSONL}"
echo "CORRECT2_JSONL : ${CORRECT2_JSONL}"
echo "LR             : ${LR}"
echo "EPOCHS         : ${EPOCHS}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "============================================"

# ================= Start =================
python sft_traditional/sft_traditional.py \
  --lr ${LR} \
  --epochs ${EPOCHS} \
  --model_name ${MODEL_NAME} \
  --wrong_jsonl ${WRONG_JSONL} \
  --correct1_jsonl ${CORRECT1_JSONL} \
  --correct2_jsonl ${CORRECT2_JSONL} \
  --out_dir ${OUT_DIR} \
  --cuda_visible_devices ${CUDA_VISIBLE_DEVICES}
