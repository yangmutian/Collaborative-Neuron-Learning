#!/usr/bin/env bash
set -e

# ================= Dataset =================
DATASET="${1:-csqa}"

# ================= GPU =================
CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# ================= Model =================
MODEL_NAME="/ssd/common/LLMs/Llama-3.2-3B-Instruct"
MODEL_TAG="$(basename "${MODEL_NAME%/}")"   # Qwen2.5-3B-Instruct

# ================= Data =================
WRONG_JSONL="./data/${DATASET}_wrong_${MODEL_TAG}.jsonl"
CORRECT_JSONL="./data/${DATASET}_correct_${MODEL_TAG}.jsonl"

# ================= Training hyperparameters =================
LR=2e-8
EPOCHS=25
USE_FREEZE=1

# ================= Output directory =================
OUT_DIR="./zero_ckpts/${DATASET}_${MODEL_TAG}_lr${LR}_usefreeze${USE_FREEZE}"
mkdir -p "${OUT_DIR}"

# ================= Print configuration =================
echo "================ Experiment ================="
echo "GPU               : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME        : ${MODEL_NAME}"
echo "MODEL_TAG         : ${MODEL_TAG}"
echo "WRONG_JSONL       : ${WRONG_JSONL}"
echo "CORRECT_JSONL     : ${CORRECT_JSONL}"
echo "LR                : ${LR}"
echo "EPOCHS            : ${EPOCHS}"
echo "USE_FREEZE        : ${USE_FREEZE}"
echo "OUT_DIR           : ${OUT_DIR}"
echo "============================================"

# ================= Start training =================
python sft/sft.py \
  --lr ${LR} \
  --epochs ${EPOCHS} \
  --use_freeze ${USE_FREEZE} \
  --model_name ${MODEL_NAME} \
  --wrong_jsonl ${WRONG_JSONL} \
  --correct_jsonl ${CORRECT_JSONL} \
  --out_dir ${OUT_DIR} \
  --cuda_visible_devices ${CUDA_VISIBLE_DEVICES}
