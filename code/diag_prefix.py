"""Check prefix-tuning eval: generate() bs=1 / bs=32 vs no-cache greedy, same adapter, 100 ARC-Easy questions."""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp/bench")
from run import MODEL, DATA, build_prompt, extract_cs, generate_all, read_jsonl  # noqa: E402

method = sys.argv[1] if len(sys.argv) > 1 else "prefix_mlp"
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(base, f"/root/autodl-tmp/outputs/bench/commonsense/{method}/adapter").eval()
rows = read_jsonl(f"{DATA}/test_ARC-Easy.jsonl", 100)
prompts = [build_prompt(r) for r in rows]
for name, bs, no_cache in (("generate", 1, False), ("generate", 32, False), ("no_cache", 32, True)):
    preds = generate_all(model, tok, prompts, bs, 16, no_cache)
    acc = sum(extract_cs("ARC-Easy", p) == r["answer"] for r, p in zip(rows, preds)) / len(rows)
    dist = {}
    for p in preds:
        a = extract_cs("ARC-Easy", p)
        dist[a] = dist.get(a, 0) + 1
    print(f"{method} {name} bs={bs}: acc={acc:.2%} answer_dist={dist} sample={preds[0]!r}", flush=True)
