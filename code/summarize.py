"""Collect results.json files under OUT into results.md / results.csv."""
import csv
import glob
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/outputs/bench"
ORDER = ["base", "full", "freeze", "lora", "qlora", "dora", "rslora", "pissa", "loraplus", "adalora",
         "vera", "ia3", "prefix", "prefix_mlp", "prompt", "oft", "lokr"]

rows = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "*", "*", "results.json"))]
rows.sort(key=lambda r: (r["task"], ORDER.index(r["method"]) if r["method"] in ORDER else 99))

lines, flat = [], []
for task in sorted({r["task"] for r in rows}):
    trs = [r for r in rows if r["task"] == task]
    tests = [k for k in trs[0]["eval"] if k != "avg"]
    head = ["method", "lr", "trainable", "trainable%", "peak_mem_GB", "train_min", "train_loss"] + tests + ["avg"]
    lines += [f"\n## {task}\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in trs:
        vals = [r["method"], r.get("lr") or "-", f"{r.get('trainable_params', 0) / 1e6:.2f}M",
                r.get("trainable_pct", 0), r.get("peak_mem_alloc_gb", "-"),
                round(r["train_runtime_s"] / 60, 1) if "train_runtime_s" in r else "-", r.get("train_loss", "-")]
        vals += [r["eval"].get(t, "-") for t in tests] + [r["eval"]["avg"]]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
        flat.append(dict(zip(head, vals), task=task))

open(os.path.join(OUT, "results.md"), "w").write("\n".join(lines) + "\n")
if flat:
    keys = ["task"] + list(dict.fromkeys(k for f in flat for k in f if k != "task"))
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
print("\n".join(lines))
