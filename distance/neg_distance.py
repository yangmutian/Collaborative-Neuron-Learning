# python distance/neg_distance.py --dataset mmlu
# python distance/neg_distance.py --dataset mmlu medqa

import json
import argparse
import os

def grad_dot_abs_mismatch_stats(jsonl_path):
    rows_negative = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["grad_dot"] < 0:
                r["grad_dot_abs"] = abs(r["grad_dot"])
                rows_negative.append(r)
    
    def compute_stats(rows_subset):
        total = len(rows_subset)
        wrong = sum(1 for r in rows_subset if r["label"] != r["predict_label"])
        ratio = (wrong / total * 100) if total > 0 else 0.0
        return {
            "total": total, 
            "wrong": wrong, 
            "wrong_ratio": f"{ratio:.1f}%"
        }
    
    rows_negative.sort(key=lambda x: x["grad_dot_abs"])
    
    n_negative = len(rows_negative)
    third_size = n_negative // 3
    
    front_third = rows_negative[:third_size] if third_size > 0 else []
    back_third = rows_negative[-third_size:] if third_size > 0 else []
    
    return {
        "negative_grad_dot_front_third": compute_stats(front_third) if front_third else {"total": 0, "wrong": 0, "wrong_ratio": "0.0%"},
        "negative_grad_dot_back_third": compute_stats(back_third) if back_third else {"total": 0, "wrong": 0, "wrong_ratio": "0.0%"}
    }
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        required=True,
        help="Dataset name(s), e.g., mmlu, medqa"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="Model name(s), e.g., Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct"
    )
    
    args = parser.parse_args()
    
    models = args.model if args.model else [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]
    
    jsonl_files = []
    for dataset in args.dataset:
        for model in models:
            jsonl_path = f"./distance/{dataset}_{model}/correct_with_grad_dot.jsonl"
            jsonl_files.append(jsonl_path)
    
    for jsonl_path in jsonl_files:
        if not os.path.exists(jsonl_path):
            print(f"Warning: File not found, skipping: {jsonl_path}")
            continue
        
        print(f"\nProcessing: {jsonl_path}")
        result = grad_dot_abs_mismatch_stats(jsonl_path)
        print(result)

if __name__ == "__main__":
    main()