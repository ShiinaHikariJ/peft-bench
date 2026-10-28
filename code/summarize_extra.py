"""Collect extra_results.json (MMLU / MATH-500) into extra_results.md / extra_results.csv.

python summarize_extra.py OUT [OUT2 ...]   # OUT2 e.g. the PiSSA lr=2e-4 re-run, listed as pissa@<dir name>
"""
import csv
import glob
import json
import os
import sys

ORDER = ["base", "full", "freeze", "lora", "qlora", "dora", "rslora", "pissa", "loraplus", "adalora",
         "vera", "ia3", "prefix", "prefix_mlp", "prompt", "oft", "lokr"]

outs = sys.argv[1:] or ["/root/autodl-tmp/outputs/bench"]
rows = {}
for k, out in enumerate(outs):
    for p in glob.glob(os.path.join(out, "*", "*", "extra_results.json")):
        r = json.load(open(p))
        name = r["method"] if k == 0 else f"{r['method']}@{os.path.basename(os.path.normpath(out))}"
        rows[(name, r["task"])] = r
base = rows.get(("base", "math"))

def key(name):
    m = name.split("@")[0]
    return (ORDER.index(m) if m in ORDER else 99, name)

names = sorted({n for n, _ in rows}, key=key)
head = ["method", "MMLU (CS-tuned)", "MMLU (Math-tuned)", "Δ CS", "Δ Math", "MATH-500", "L1", "L2", "L3", "L4", "L5"]
lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
flat = []
b = base["mmlu"]["avg"] if base else None
for n in names:
    cs = (rows.get((n, "commonsense")) or (base if n == "base" else {})).get("mmlu", {}).get("avg")
    ma = rows.get((n, "math"), {})
    mm, m5 = ma.get("mmlu", {}).get("avg"), ma.get("math500", {})
    d = lambda v: round(v - b, 2) if v is not None and b is not None else None  # noqa: E731
    vals = [n, cs, mm, d(cs), d(mm), m5.get("avg")] + [m5.get(f"level{i}") for i in range(1, 6)]
    lines.append("| " + " | ".join("-" if v is None else str(v) for v in vals) + " |")
    flat.append(dict(zip(head, vals)))

dst = outs[0]
open(os.path.join(dst, "extra_results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
with open(os.path.join(dst, "extra_results.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=head)
    w.writeheader()
    w.writerows(flat)
print("\n".join(lines))
