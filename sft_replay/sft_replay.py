# -*- coding: utf-8 -*-
import os, json, csv, argparse
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--model_name", type=str, required=True)

    # training data
    p.add_argument("--wrong_jsonl", type=str, required=True)
    p.add_argument("--correct1_jsonl", type=str, required=True)

    # evaluation data
    p.add_argument("--correct2_jsonl", type=str, required=True)

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    return p.parse_args()

args = parse_args()

# ======================= Configuration =======================
if args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = args.model_name
WRONG_JSONL = args.wrong_jsonl
CORRECT1_JSONL = args.correct1_jsonl
CORRECT2_JSONL = args.correct2_jsonl

LR = args.lr
EPOCHS = args.epochs

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_DIR = os.path.join(OUT_DIR, "jsonl")
os.makedirs(JSONL_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")

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

# ======================= train: pure SGD =======================
def train_sgd(model, tok, train_rows, lr, desc):
    model.train()
    loss_sum = 0.0
    n = len(train_rows)

    for ex in tqdm(train_rows, desc=desc):
        loss = next_token_loss(model, tok, ex["question"], ex["label"])
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)

        loss_sum += float(loss.detach())
        del loss

    return loss_sum / n

# ======================= infer =======================
@torch.no_grad()
def infer_and_dump(model, tok, data, cand_ids, path, desc):
    model.eval()
    ok = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in tqdm(data, desc=desc):
            pred = predict_abcd(model, tok, ex["question"], cand_ids)
            ok += (pred == ex["label"])
            f.write(json.dumps({
                "question": ex["question"],
                "label": ex["label"],
                "predict_label": pred
            }, ensure_ascii=False) + "\n")
    model.train()
    return ok

# ======================= main =======================
def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    wrong_data = read_jsonl(WRONG_JSONL)
    correct1_data = read_jsonl(CORRECT1_JSONL)
    correct2_data = read_jsonl(CORRECT2_JSONL)

    # ✅ training set = wrong + correct1
    train_data = wrong_data + correct1_data

    random.shuffle(train_data)

    cand_ids = [
        tok(c, add_special_tokens=False,
            return_tensors="pt").input_ids[0, -1].item()
        for c in "ABCD"
    ]

    header = [
        "epoch",
        "train_avg_loss",
        "wrong_to_correct",
        "correct2_to_wrong",
    ]

    for ep in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {ep} =====")

        train_loss = train_sgd(
            model, tok, train_data, LR,
            desc=f"train wrong+correct1 ep{ep}"
        )

        w_ok = infer_and_dump(
            model, tok, wrong_data, cand_ids,
            os.path.join(JSONL_DIR, f"infer_wrong_ep{ep}.jsonl"),
            f"infer wrong ep{ep}"
        )
        c2_ok = infer_and_dump(
            model, tok, correct2_data, cand_ids,
            os.path.join(JSONL_DIR, f"infer_correct2_ep{ep}.jsonl"),
            f"infer correct2 ep{ep}"
        )

        append_csv(SUMMARY_CSV, {
            "epoch": ep,
            "train_avg_loss": train_loss,
            "wrong_to_correct": w_ok,
            "correct2_to_wrong": len(correct2_data) - c2_ok,
        }, header)

    print("Done.")

if __name__ == "__main__":
    main()
