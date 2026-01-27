# -*- coding: utf-8 -*-
import os, json, csv, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--wrong_jsonl", type=str, required=True)
    p.add_argument("--correct_jsonl", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    # LoRA hyperparameters
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    return p.parse_args()

args = parse_args()

# ======================= Configuration =======================
if args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = args.model_name
WRONG_JSONL = args.wrong_jsonl
CORRECT_JSONL = args.correct_jsonl
LR = args.lr
EPOCHS = args.epochs

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_DIR = os.path.join(OUT_DIR, "jsonl")
os.makedirs(JSONL_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")

print("DEVICE:", DEVICE)
print("MODEL:", MODEL_NAME)
print("OUT_DIR:", OUT_DIR)
print("JSONL_DIR:", JSONL_DIR)
print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

# ======================= IO =======================
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def append_csv(path, row, header):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ======================= Basic =======================
def build_inputs(tok, q):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True
    )
    return tok(prompt, return_tensors="pt").to(DEVICE)

def label_token_id(tok, label):
    return tok(label, add_special_tokens=False,
               return_tensors="pt").input_ids[0, -1].to(DEVICE)

def next_token_loss(model, tok, q, label):
    inp = build_inputs(tok, q)
    logits = model(**inp).logits[:, -1, :]
    tgt = label_token_id(tok, label).unsqueeze(0)
    return F.cross_entropy(logits, tgt)

@torch.no_grad()
def predict_abcd(model, tok, q, cand_ids):
    inp = build_inputs(tok, q)
    logits = model(**inp).logits[:, -1, :][0]
    return "ABCD"[int(torch.argmax(logits[cand_ids]))]

# ======================= Train (LoRA, wrong only) =======================
def train_lora(model, tok, wrong_rows, optimizer, desc):
    model.train()
    n = len(wrong_rows)
    loss_sum = 0.0

    for ex in tqdm(wrong_rows, desc=desc):
        loss = next_token_loss(model, tok, ex["question"], ex["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.detach())
        del loss

    return loss_sum / n

# ======================= Infer =======================
@torch.no_grad()
def infer_and_dump(model, tok, data, cand_ids, path, desc):
    model.eval()
    ok = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in tqdm(data, desc=desc):
            pred = predict_abcd(model, tok, ex["question"], cand_ids)
            ok += (pred == ex["label"])
            f.write(json.dumps({
                "label": ex["label"],
                "predict_label": pred,
                "question": ex["question"]
            }, ensure_ascii=False) + "\n")
    model.train()
    return ok

# ======================= Main =======================
def main():
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    wrong_data = read_jsonl(WRONG_JSONL)
    correct_data = read_jsonl(CORRECT_JSONL)

    cand_ids = [
        tok(c, add_special_tokens=False,
            return_tensors="pt").input_ids[0, -1].item()
        for c in "ABCD"
    ]

    # Optimizer (only LoRA parameters are trainable)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    header = [
        "epoch",
        "train_avg_loss",
        "wrong_to_correct",
        "correct_to_wrong",
    ]

    for ep in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {ep} =====")

        train_loss = train_lora(
            model, tok, wrong_data, optimizer,
            desc=f"train wrong (LoRA) ep{ep}"
        )

        w_ok = infer_and_dump(
            model, tok, wrong_data, cand_ids,
            os.path.join(JSONL_DIR, f"infer_wrong_ep{ep}.jsonl"),
            f"infer wrong ep{ep}"
        )
        c_ok = infer_and_dump(
            model, tok, correct_data, cand_ids,
            os.path.join(JSONL_DIR, f"infer_correct_ep{ep}.jsonl"),
            f"infer correct ep{ep}"
        )

        append_csv(SUMMARY_CSV, {
            "epoch": ep,
            "train_avg_loss": train_loss,
            "wrong_to_correct": w_ok,
            "correct_to_wrong": len(correct_data) - c_ok,
        }, header)

    print("Done.")

if __name__ == "__main__":
    main()

