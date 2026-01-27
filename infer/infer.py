# -*- coding: utf-8 -*-
import os, json, argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--jsonl", type=str, required=True, help="Full input jsonl, e.g. ./data/mmlu.jsonl")
    p.add_argument("--out_wrong_jsonl", type=str, required=True)
    p.add_argument("--out_correct_jsonl", type=str, required=True)
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    return p.parse_args()

args = parse_args()

if args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= IO =======================
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ======================= Basic =======================
def build_inputs(tok, question):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True
    )
    return tok(prompt, return_tensors="pt").to(DEVICE)

@torch.no_grad()
def predict_abcd(model, tok, question, cand_ids):
    inp = build_inputs(tok, question)
    logits = model(**inp).logits[:, -1, :][0]
    idx = int(torch.argmax(logits[cand_ids]).item())
    return "ABCD"[idx]

# ======================= Main Process =======================
def main():
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    data = read_jsonl(args.jsonl)

    cand_ids = [
        tok(c, add_special_tokens=False, return_tensors="pt").input_ids[0, -1].item()
        for c in "ABCD"
    ]

    num_correct = 0
    num_wrong = 0

    with open(args.out_wrong_jsonl, "w", encoding="utf-8") as fw, \
         open(args.out_correct_jsonl, "w", encoding="utf-8") as fc:

        for ex in tqdm(data, desc="infer & split"):
            pred = predict_abcd(model, tok, ex["question"], cand_ids)

            out_ex = dict(ex)
            out_ex["predict_label"] = pred

            if pred == ex["label"]:
                num_correct += 1
                write_jsonl_line(fc, out_ex)
            else:
                num_wrong += 1
                write_jsonl_line(fw, out_ex)

    total = len(data)
    acc = num_correct / total if total > 0 else 0.0

    print("========== Inference Summary ==========")
    print(f"Total    : {total}")
    print(f"Correct  : {num_correct}")
    print(f"Wrong    : {num_wrong}")
    print(f"Accuracy : {acc:.4f}")

if __name__ == "__main__":
    main()
