# -*- coding: utf-8 -*-
import os
import json
import csv
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--wrong_jsonl", type=str, required=True)
    p.add_argument("--correct_jsonl", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    return p.parse_args()

# ======================= Utils =======================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

# ======================= Prompt / Loss =======================
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

# ======================= Gradient =======================
def mean_grad_correct(model, tok, rows, named_params, desc):
    model.train()
    model.zero_grad(set_to_none=True)
    n = len(rows)
    assert n > 0

    for r in tqdm(rows, desc=desc):
        loss = next_token_loss(model, tok, r["question"], r["label"])
        loss.backward()
        del loss

    grad = {}
    for name, p in named_params:
        if p.grad is not None:
            p.grad.div_(n)
            grad[name] = p.grad.detach().clone()
    return grad

def grad_one_sample(model, tok, row, named_params):
    model.zero_grad(set_to_none=True)
    loss = next_token_loss(model, tok, row["question"], row["label"])
    loss.backward()
    del loss

    grad = {}
    for name, p in named_params:
        if p.grad is not None:
            grad[name] = p.grad.detach()
    return grad

def grad_dot(g, g_ref):
    s = 0.0
    for k in g:
        if k in g_ref:
            s += torch.sum(g[k] * g_ref[k]).item()
    return s

# ======================= Main =======================
def main():
    global DEVICE
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)

    named_params = list(model.named_parameters())

    wrong_rows = load_jsonl(args.wrong_jsonl)
    correct_rows = load_jsonl(args.correct_jsonl)

    wrong_only = [r for r in wrong_rows if r["label"] != r["predict_label"]]

    mean_wrong_grad = mean_grad_correct(model, tok, wrong_only, named_params, desc="mean wrong grad")

    results = []

    # Statistics for CSV summary
    dot_neg_total = 0
    dot_neg_wrong = 0
    dot_pos_total = 0
    dot_pos_wrong = 0

    # Calculate grad_dot for each correct sample
    for r in tqdm(correct_rows, desc="correct grad dot"):
        g = grad_one_sample(model, tok, r, named_params)
        dot = grad_dot(g, mean_wrong_grad)

        is_wrong = r["label"] != r["predict_label"]

        # Categorize statistics
        if dot < 0:
            dot_neg_total += 1
            if is_wrong:
                dot_neg_wrong += 1
        else:
            dot_pos_total += 1
            if is_wrong:
                dot_pos_wrong += 1

        # Save JSONL (contains question, label, predict_label, grad_dot)
        results.append({
            "question": r["question"],
            "label": r["label"],
            "predict_label": r["predict_label"],
            "grad_dot": dot
        })

    # Save JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save CSV: each sample + dot<0/dot>=0 labels + summary statistics
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        csv_columns = ["category", "total", "wrong", "wrong_ratio"]
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        # dot < 0
        writer.writerow({
            "category": "dot < 0",
            "total": dot_neg_total,
            "wrong": dot_neg_wrong,
            "wrong_ratio": f"{(dot_neg_wrong/dot_neg_total*100):.2f}%" if dot_neg_total > 0 else "0.00%"
        })

        # dot >= 0
        writer.writerow({
            "category": "dot >= 0",
            "total": dot_pos_total,
            "wrong": dot_pos_wrong,
            "wrong_ratio": f"{(dot_pos_wrong/dot_pos_total*100):.2f}%" if dot_pos_total > 0 else "0.00%"
        })

if __name__ == "__main__":
    main()
