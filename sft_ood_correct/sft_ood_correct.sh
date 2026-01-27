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
CORRECT1_JSONL="./data/${DATASET}_correct1_${MODEL_TAG}.jsonl"   # for training
CORRECT2_JSONL="./data/${DATASET}_correct2_${MODEL_TAG}.jsonl"   # for inference
WRONG_JSONL="./data/${DATASET}_wrong_${MODEL_TAG}.jsonl"         # reference / mean-grad

# ================= Training hyperparameters =================
LR=1e-7
EPOCHS=25
USE_FREEZE=1   # 0: vanilla SGD, 1: CNL / gradient alignment

# ================= Output directory =================
OUT_DIR="./ood/${DATASET}_correct_${MODEL_TAG}_lr${LR}_usefreeze${USE_FREEZE}"
mkdir -p "${OUT_DIR}"

# ================= Print configuration =================
echo "================ Experiment ================="
echo "GPU               : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME        : ${MODEL_NAME}"
echo "MODEL_TAG         : ${MODEL_TAG}"
echo "CORRECT1_JSONL    : ${CORRECT1_JSONL} (train)"
echo "CORRECT2_JSONL    : ${CORRECT2_JSONL} (infer)"
echo "WRONG_JSONL       : ${WRONG_JSONL} (reference)"
echo "LR                : ${LR}"
echo "EPOCHS            : ${EPOCHS}"
echo "USE_FREEZE        : ${USE_FREEZE}"
echo "OUT_DIR           : ${OUT_DIR}"
echo "============================================"

# ================= Start training =================
python sft_ood_correct/sft_ood_correct.py \
  --lr ${LR} \
  --epochs ${EPOCHS} \
  --use_freeze ${USE_FREEZE} \
  --model_name ${MODEL_NAME} \
  --correct1_jsonl ${CORRECT1_JSONL} \
  --correct2_jsonl ${CORRECT2_JSONL} \
  --wrong_jsonl ${WRONG_JSONL} \
  --out_dir ${OUT_DIR} \
  --cuda_visible_devices ${CUDA_VISIBLE_DEVICES}
