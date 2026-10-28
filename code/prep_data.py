"""Build unified jsonl files: {instruction, input, output} for training, {instruction, input, answer} for test."""
import json
import os
import random

from datasets import load_dataset
from huggingface_hub import snapshot_download

D = "/root/autodl-tmp/data"
N_TRAIN = int(os.environ.get("N_TRAIN", 20000))
SEED = 42
CS_TESTS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]


def dump(rows, name):
    with open(os.path.join(D, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


snapshot_download("Qwen/Qwen2.5-1.5B", local_dir="/root/autodl-tmp/models/Qwen2.5-1.5B")

# commonsense (LLM-Adapters / DoRA protocol)
cs = load_dataset("zwhe99/commonsense_170k", split="train")
print("commonsense_170k columns:", cs.column_names, len(cs))
cs = cs.shuffle(seed=SEED).select(range(N_TRAIN))
dump([{"instruction": r["instruction"], "input": r.get("input") or "", "output": r["output"]} for r in cs], "cs_train.jsonl")

for t in CS_TESTS:
    rows = json.load(open(os.path.join(D, "raw", f"{t}.json")))
    dump([{"instruction": r["instruction"], "input": r.get("input") or "", "answer": r["answer"]} for r in rows],
         f"test_{t}.jsonl")

# math
mm = load_dataset("meta-math/MetaMathQA", split="train")
print("MetaMathQA columns:", mm.column_names, len(mm))
mm = mm.shuffle(seed=SEED).select(range(N_TRAIN))
dump([{"instruction": r["query"], "input": "", "output": r["response"]} for r in mm], "math_train.jsonl")

gs = load_dataset("openai/gsm8k", "main", split="test")
dump([{"instruction": r["question"], "input": "",
       "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in gs], "test_gsm8k.jsonl")

random.seed(SEED)
print("sample cs train:", json.dumps(cs[0], ensure_ascii=False)[:600])
print("sample math train:", json.dumps(mm[0], ensure_ascii=False)[:600])
