# -*- coding: utf-8 -*-
import json
import random
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Split a dataset into two parts.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input jsonl file.")
    parser.add_argument("--output1", type=str, required=True, help="Output path for split1.")
    parser.add_argument("--output2", type=str, required=True, help="Output path for split2.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # Load samples
    all_items = []
    if not os.path.exists(args.input_jsonl):
        print(f"Error: {args.input_jsonl} not found.")
        return

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            all_items.append(json.loads(line))

    print(f"Total samples: {len(all_items)}")

    # Shuffle and split
    random.shuffle(all_items)
    mid = (len(all_items) + 1) // 2

    split1 = all_items[:mid]
    split2 = all_items[mid:]

    print(f"Split done: part1={len(split1)}, part2={len(split2)}")

    # Save
    for path, data in [(args.output1, split1), (args.output2, split2)]:
        with open(path, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Saved to: {path}")

if __name__ == "__main__":
    main()
