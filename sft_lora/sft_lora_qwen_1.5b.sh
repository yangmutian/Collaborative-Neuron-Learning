#!/usr/bin/env bash
set -e

# ================= Dataset =================
DATASET="${1:-csqa}"

# ================= GPU =================
CUDA_VISIBLE_DEVICES=7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# ================= Model =================
MODEL_NAME="/ssd/common/LLMs/Qwen2.5-1.5B-Instruct"
MODEL_TAG="$(basename "${MODEL_NAME%/}")"

# ================= Data =================
WRONG_JSONL="./data/${DATASET}_wrong_${MODEL_TAG}.jsonl"
CORRECT_JSONL="./data/${DATASET}_correct_${MODEL_TAG}.jsonl"

# ================= Training hyperparameters =================
LR=1e-5
EPOCHS=25

# ================= LoRA hyperparameters =================
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# ================= Output directory =================
OUT_DIR="./lora_ckpts/${DATASET}_${MODEL_TAG}_lr${LR}_r${LORA_R}_sgd"
mkdir -p "${OUT_DIR}"

# ================= Print configuration =================
echo "================ LoRA Experiment ================="
echo "GPU               : ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME        : ${MODEL_NAME}"
echo "MODEL_TAG         : ${MODEL_TAG}"
echo "WRONG_JSONL       : ${WRONG_JSONL}"
echo "CORRECT_JSONL     : ${CORRECT_JSONL}"
echo "LR                : ${LR}"
echo "EPOCHS            : ${EPOCHS}"
echo "LORA_R            : ${LORA_R}"
echo "LORA_ALPHA        : ${LORA_ALPHA}"
echo "LORA_DROPOUT      : ${LORA_DROPOUT}"
echo "OUT_DIR           : ${OUT_DIR}"
echo "=================================================="

# ================= Start training =================
python sft_lora/sft_lora_sgd.py \
  --lr ${LR} \
  --epochs ${EPOCHS} \
  --model_name ${MODEL_NAME} \
  --wrong_jsonl ${WRONG_JSONL} \
  --correct_jsonl ${CORRECT_JSONL} \
  --out_dir ${OUT_DIR} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --cuda_visible_devices ${CUDA_VISIBLE_DEVICES}

