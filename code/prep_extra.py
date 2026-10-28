"""Extra benchmarks: MMLU (5-shot, knowledge retention) and MATH-500 (harder math)."""
import json
import os

from datasets import load_dataset

D = "/root/autodl-tmp/data"


def dump(rows, name):
    with open(os.path.join(D, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


for split, name in (("test", "mmlu_test.jsonl"), ("dev", "mmlu_dev.jsonl")):
    ds = load_dataset("cais/mmlu", "all", split=split)
    dump([{"subject": r["subject"], "question": r["question"], "choices": r["choices"], "answer": r["answer"]}
          for r in ds], name)

m5 = load_dataset("HuggingFaceH4/MATH-500", split="test")
dump([{"instruction": r["problem"], "input": "", "answer": r["answer"], "subject": r["subject"], "level": r["level"]}
      for r in m5], "test_math500.jsonl")
