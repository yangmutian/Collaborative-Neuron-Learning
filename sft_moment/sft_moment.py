# -*- coding: utf-8 -*-
import os, json, csv, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================= CLI =======================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument(
        "--use_freeze", type=int, choices=[0, 1], required=True,
        help="1: CSL (mask at update); 0: normal momentum SGD"
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--wrong_jsonl", type=str, required=True)
    p.add_argument("--correct_jsonl", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    return p.parse_args()

args = parse_args()

# ======================= Config =======================
if args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = args.lr
EPOCHS = args.epochs
USE_FREEZE = bool(args.use_freeze)

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

JSONL_DIR = os.path.join(OUT_DIR, "jsonl")
os.makedirs(JSONL_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")

print("USE_FREEZE:", USE_FREEZE)
print("DEVICE:", DEVICE)

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

# ======================= Model utils =======================
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

# ======================= Mean grad on correct set =======================
def mean_grad_correct(model, tok, rows, named_params, desc):
    model.train()
    model.zero_grad(set_to_none=True)

    n = len(rows)
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

# ======================= Train: Momentum SGD + CSL =======================
def train_wrong_sgd_momentum_update_mask(
    model,
    tok,
    wrong_rows,
    lr,
    desc,
    named_params,
    velocity,
    momentum=0.9,
    grad_ref_gpu=None
):
    model.train()
    loss_sum = 0.0
    n = len(wrong_rows)

    for ex in tqdm(wrong_rows, desc=desc):
        model.zero_grad(set_to_none=True)
        loss = next_token_loss(model, tok, ex["question"], ex["label"])
        loss.backward()

        with torch.no_grad():
            for name, p in named_params:
                g = p.grad
                if g is None:
                    continue

                # ---- momentum accumulation ----
                velocity[name].mul_(momentum).add_(g)

                # ---- CSL: mask only at update ----
                if grad_ref_gpu is not None:
                    gref = grad_ref_gpu.get(name, None)
                    if gref is not None:
                        mask = (velocity[name] * gref).ge(0)
                        p.add_(velocity[name] * mask, alpha=-lr)
                        continue

                # ---- vanilla momentum SGD ----
                p.add_(velocity[name], alpha=-lr)

        loss_sum += float(loss.detach())
        del loss

    return loss_sum / n

# ======================= Inference =======================
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
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    wrong_data = read_jsonl(args.wrong_jsonl)
    correct_data = read_jsonl(args.correct_jsonl)

    cand_ids = [
        tok(c, add_special_tokens=False,
            return_tensors="pt").input_ids[0, -1].item()
        for c in "ABCD"
    ]

    named_params = list(model.named_parameters())

    # ===== initialize velocity =====
    velocity = {
        name: torch.zeros_like(p, device=p.device)
        for name, p in named_params
        if p.requires_grad
    }

    header = ["epoch", "train_avg_loss", "wrong_to_correct", "correct_to_wrong"]

    for ep in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {ep} =====")

        grad_c_ref_gpu = None
        if USE_FREEZE:
            grad_c_ref_gpu = mean_grad_correct(
                model, tok, correct_data, named_params,
                f"correct mean-grad ep{ep}"
            )

        train_loss = train_wrong_sgd_momentum_update_mask(
            model, tok, wrong_data,
            lr=LR,
            desc=f"train wrong ep{ep}",
            named_params=named_params,
            velocity=velocity,
            momentum=0.9,
            grad_ref_gpu=grad_c_ref_gpu if USE_FREEZE else None
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
