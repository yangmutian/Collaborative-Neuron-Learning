#!/usr/bin/env bash
set -e

# ================= Dataset =================
DATASET="${1:-csqa}"

# ================= GPU =================
CUDA_VISIBLE_DEVICES=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# ================= Model =================
MODEL_NAME="/ssd/common/LLMs/Qwen2.5-1.5B-Instruct"
MODEL_TAG="$(basename "${MODEL_NAME%/}")"   # Qwen2.5-1.5B-Instruct

# ================= Data =================
WRONG1_JSONL="./data/${DATASET}_wrong1_${MODEL_TAG}.jsonl"   # for training
WRONG2_JSONL="./data/${DATASET}_wrong2_${MODEL_TAG}.jsonl"   # for inference
CORRECT_JSONL="./data/${DATASET}_correct_${MODEL_TAG}.jsonl"

# ================= Training hyperparameters =================
LR=1e-7
EPOCHS=25
USE_FREEZE=1    # 0: vanilla SGD, 1: CNL / gradient alignment

# ================= Output directory =================
OUT_DIR="./ood/${DATASET}_wrong_${MODEL_TAG}_lr${LR}_usefreeze${USE_FREEZE}"
mkdir -p "${OUT_DIR}"

# ================= Print configuration =================
echo "================ Experiment ================="
echo "GPU               : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME        : ${MODEL_NAME}"
echo "MODEL_TAG         : ${MODEL_TAG}"
echo "WRONG1_JSONL      : ${WRONG1_JSONL} (train)"
echo "WRONG2_JSONL      : ${WRONG2_JSONL} (infer)"
echo "CORRECT_JSONL     : ${CORRECT_JSONL}"
echo "LR                : ${LR}"
echo "EPOCHS            : ${EPOCHS}"
echo "USE_FREEZE        : ${USE_FREEZE}"
echo "OUT_DIR           : ${OUT_DIR}"
echo "============================================"

# ================= Start training =================
python sft_ood_wrong/sft_ood_wrong.py \
  --lr ${LR} \
  --epochs ${EPOCHS} \
  --use_freeze ${USE_FREEZE} \
  --model_name ${MODEL_NAME} \
  --wrong1_jsonl ${WRONG1_JSONL} \
  --wrong2_jsonl ${WRONG2_JSONL} \
  --correct_jsonl ${CORRECT_JSONL} \
  --out_dir ${OUT_DIR} \
  --cuda_visible_devices ${CUDA_VISIBLE_DEVICES}
