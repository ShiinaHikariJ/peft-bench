"""Round-3 aggregation: mean ± std over seeds, paired bootstrap significance, learning curves, parity memory.

python analyze_r3.py RAW_DIR OUT_DIR      # RAW_DIR contains outputs/{bench,r3,r3_05b}
"""
import json
import os
import sys

import numpy as np

RAW, OUT = sys.argv[1], sys.argv[2]
O = os.path.join(RAW, "outputs")
os.makedirs(OUT, exist_ok=True)
TASKS = ["commonsense", "math"]


def load(d):
    p = os.path.join(d, "results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    e = os.path.join(d, "extra_results.json")
    r["extra"] = json.load(open(e)) if os.path.exists(e) else {}
    c = os.path.join(d, "correct.json")
    r["correct"] = json.load(open(c)) if os.path.exists(c) else None
    r["dir"] = d
    return r


# --------------------------------------------------------------- configurations (1.5B)
def cfg_runs(name, task):
    """Return the list of seed runs for one configuration on one task."""
    r3, b = os.path.join(O, "r3"), os.path.join(O, "bench")
    s = lambda sub: load(os.path.join(r3, "seed", task, sub))  # noqa: E731
    if name in ("full", "lora"):            # seed 42 re-run (with learning curve) + 43 + 44
        runs = [s(f"{name}_s{k}") for k in (42, 43, 44)]
    elif name == "full_lr1e-5":
        runs = [s(f"full_lr1e-5_s{k}") for k in (42, 43, 44)]
    elif name == "lora_lr5e-4":
        runs = [s(f"lora_lr5e-4_s{k}") for k in (42, 43, 44)] if task == "math" else \
            [load(os.path.join(r3, "lr", task, "lora_lr5e-4"))]
    elif name in ("qlora", "dora", "pissa", "vera", "oft", "adalora", "lokr"):
        runs = [load(os.path.join(b, task, name))] + [s(f"{name}_s{k}") for k in (43, 44)]
    else:                                   # single-seed round-1 methods
        runs = [load(os.path.join(b, task, name))]
    return [r for r in runs if r]


CONFIGS = ["full", "full_lr1e-5", "lora", "lora_lr5e-4", "qlora", "dora", "pissa", "oft", "adalora", "lokr", "vera",
           "rslora", "loraplus", "ia3", "freeze", "prompt", "prefix_mlp", "prefix", "base"]
LABEL = {"full": "Full FT (lr 2e-5)", "full_lr1e-5": "Full FT (lr 1e-5, tuned)", "lora": "LoRA (lr 2e-4)",
         "lora_lr5e-4": "LoRA (lr 5e-4)", "qlora": "QLoRA", "dora": "DoRA", "pissa": "PiSSA", "oft": "OFT",
         "adalora": "AdaLoRA", "lokr": "LoKr", "vera": "VeRA", "rslora": "rsLoRA", "loraplus": "LoRA+",
         "ia3": "IA³", "freeze": "Freeze", "prompt": "Prompt", "prefix_mlp": "Prefix-MLP", "prefix": "Prefix",
         "base": "base"}


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(mean=round(float(np.mean(vals)), 2), std=round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else None,
                n=len(vals), vals=[round(v, 2) for v in vals])


table = {}
for c in CONFIGS:
    row = {}
    for t in TASKS:
        runs = cfg_runs(c, t)
        row[t] = dict(
            score=ms([r["eval"]["avg"] for r in runs]),
            mmlu=ms([r["extra"].get("mmlu", {}).get("avg") for r in runs]),
            math500=ms([r["extra"].get("math500", {}).get("avg") for r in runs]) if t == "math" else None,
            time_s=ms([r.get("train_runtime_s") for r in runs]),
            mem_gb=ms([r.get("peak_mem_alloc_gb") for r in runs]),
            trainable_pct=runs[0].get("trainable_pct") if runs else None)
    table[c] = row


# --------------------------------------------------------------- paired bootstrap on per-question correctness
def per_q(c, task):
    """Mean correctness over seeds for each question: {dataset: np.array}."""
    runs = [r for r in cfg_runs(c, task) if r["correct"]]
    if not runs:
        return None, 0
    keys = runs[0]["correct"].keys()
    return {k: np.mean([[ch == "1" for ch in r["correct"][k]] for r in runs], axis=0) for k in keys}, len(runs)


def boot(a, b, n=5000, seed=0):
    """Macro-average (over datasets) difference a - b with a 95% bootstrap CI, resampling questions per dataset."""
    rng = np.random.default_rng(seed)
    keys = list(a)
    diffs = {k: a[k] - b[k] for k in keys}
    point = np.mean([d.mean() for d in diffs.values()]) * 100
    samples = np.zeros(n)
    for k in keys:
        d = diffs[k]
        idx = rng.integers(0, len(d), size=(n, len(d)))
        samples += d[idx].mean(axis=1)
    samples = samples / len(keys) * 100
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return dict(diff=round(point, 2), ci=[round(lo, 2), round(hi, 2)], p=round(float(max(p, 1 / n)), 4))


PAIRS = [("lora", "full"), ("lora", "full_lr1e-5"), ("lora_lr5e-4", "full_lr1e-5"), ("full_lr1e-5", "full"),
         ("qlora", "lora"), ("dora", "lora"), ("pissa", "lora"), ("oft", "lora"), ("adalora", "lora"),
         ("lokr", "lora"), ("vera", "lora")]
sig = {}
for t in TASKS:
    for a, b in PAIRS:
        qa, na = per_q(a, t)
        qb, nb = per_q(b, t)
        if qa is None or qb is None:
            continue
        sig[f"{t}|{a}|{b}"] = dict(boot(qa, qb), seeds=[na, nb])


# --------------------------------------------------------------- learning curves & time-to-target
def curves(c, task, root="r3", model_dir="seed"):
    out = []
    for r in cfg_runs(c, task) if root == "r3" else []:
        if r.get("curve"):
            out.append([(p["frac"], p["train_time_s"], p["eval"]["avg"]) for p in r["curve"]])
    return out


def mean_curve(cs):
    if not cs:
        return None
    n = min(len(c) for c in cs)
    return [(cs[0][i][0], float(np.mean([c[i][1] for c in cs])), float(np.mean([c[i][2] for c in cs])))
            for i in range(n)]


def time_to(curve, target):
    """Linear interpolation of training time at which the (mean) curve first reaches target; None if never."""
    prev = (0.0, 0.0, None)
    for f, ts, acc in curve:
        if acc >= target:
            if prev[2] is None or prev[2] >= target:
                return ts
            w = (target - prev[2]) / (acc - prev[2])
            return prev[1] + w * (ts - prev[1])
        prev = (f, ts, acc)
    return None


curve_out, ttt = {}, {}
for t in TASKS:
    for c in ("full", "full_lr1e-5", "lora", "lora_lr5e-4"):
        mc = mean_curve(curves(c, t))
        if mc:
            curve_out[f"{t}|{c}"] = [dict(frac=f, time_s=round(ts, 1), acc=round(a, 2)) for f, ts, a in mc]
    ref = table["full_lr1e-5"][t]["score"]["mean"]
    ref_time = table["full_lr1e-5"][t]["time_s"]["mean"]
    for c in ("full", "full_lr1e-5", "lora", "lora_lr5e-4"):
        k = f"{t}|{c}"
        if k not in curve_out:
            continue
        mc = [(p["frac"], p["time_s"], p["acc"]) for p in curve_out[k]]
        t95 = time_to(mc, 0.95 * ref)
        ttt[k] = dict(target=round(0.95 * ref, 2), time_to_95_s=round(t95, 1) if t95 else None,
                      speedup_vs_full_run=round(ref_time / t95, 2) if t95 else None, full_run_time_s=ref_time)

# --------------------------------------------------------------- 3 epochs
ep3 = {}
for t in TASKS:
    for m in ("full", "lora", "qlora", "pissa"):
        r = load(os.path.join(O, "r3", "ep3", t, m))
        if r:
            ep3[f"{t}|{m}"] = dict(per_epoch=[p["eval"]["avg"] for p in r["curve"]], mmlu=r["extra"].get("mmlu", {}).get("avg"),
                                   math500=r["extra"].get("math500", {}).get("avg"), time_s=r["train_runtime_s"])

# --------------------------------------------------------------- lr sweep, optimizer
lr = {}
for t in TASKS:
    for sub in ("full_lr1e-5", "full_lr5e-5", "lora_lr1e-4", "lora_lr5e-4", "full_adamw32"):
        r = load(os.path.join(O, "r3", "lr", t, sub))
        if r:
            lr[f"{t}|{sub}"] = dict(score=r["eval"]["avg"], mmlu=r["extra"].get("mmlu", {}).get("avg"),
                                    math500=r["extra"].get("math500", {}).get("avg"))

# --------------------------------------------------------------- 0.5B
small = {}
R5 = os.path.join(O, "r3_05b")
for t in TASKS:
    for m in ("base", "full", "lora", "qlora", "dora", "pissa", "vera", "ia3"):
        runs = [load(os.path.join(R5, "s42", t, m))]
        if m in ("full", "lora"):
            runs += [load(os.path.join(R5, "seed", t, f"{m}_s{k}")) for k in (43, 44)]
        runs = [r for r in runs if r]
        small[f"{t}|{m}"] = dict(score=ms([r["eval"]["avg"] for r in runs]),
                                 mmlu=ms([r["extra"].get("mmlu", {}).get("avg") for r in runs]),
                                 math500=ms([r["extra"].get("math500", {}).get("avg") for r in runs]) if t == "math" else None,
                                 time_s=ms([r.get("train_runtime_s") for r in runs]),
                                 mem_gb=ms([r.get("peak_mem_alloc_gb") for r in runs]))

# --------------------------------------------------------------- micro-batch parity (LoRA micro_bs=4 vs full micro_bs=4)
parity = {}
for size, root, full_runs in (("1.5B", os.path.join(O, "r3"), lambda t: cfg_runs("full_lr1e-5", t)),
                              ("0.5B", R5, lambda t: [load(os.path.join(R5, "s42", t, "full"))])):
    for t in TASKS:
        r = load(os.path.join(root, "parity", t, "lora_mb4"))
        f = [x for x in full_runs(t) if x]
        if r and f:
            fm = float(np.mean([x["peak_mem_alloc_gb"] for x in f]))
            ft = float(np.mean([x["train_runtime_s"] for x in f]))
            parity[f"{size}|{t}"] = dict(lora_mem=r["peak_mem_alloc_gb"], full_mem=round(fm, 2),
                                         mem_saving_pct=round(100 * (1 - r["peak_mem_alloc_gb"] / fm), 1),
                                         lora_time_s=r["train_runtime_s"], full_time_s=round(ft, 1))

res = dict(table=table, significance=sig, curves=curve_out, time_to_95=ttt, epochs3=ep3, lr=lr, small=small,
           parity=parity, labels=LABEL)
json.dump(res, open(os.path.join(OUT, "r3_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# --------------------------------------------------------------- markdown
def f(x, nd=2):
    if x is None:
        return "–"
    if x.get("std") is None:
        return f"{x['mean']:.{nd}f}"
    return f"{x['mean']:.{nd}f} ± {x['std']:.{nd}f}"


L = ["## 1.5B main table (mean ± std over seeds; n = number of seeds)", "",
     "| config | n | Commonsense | GSM8K | MATH-500 | MMLU after CS | MMLU after Math | mem CS/Math GB | time CS/Math s |",
     "|---|---|---|---|---|---|---|---|---|"]
for c in CONFIGS:
    cs, ma = table[c]["commonsense"], table[c]["math"]
    n = (cs["score"] or {}).get("n", 0)
    L.append(f"| {LABEL[c]} | {n} | {f(cs['score'])} | {f(ma['score'])} | {f(ma['math500'])} | {f(cs['mmlu'])} | "
             f"{f(ma['mmlu'])} | {f(cs['mem_gb'])} / {f(ma['mem_gb'])} | {f(cs['time_s'], 0)} / {f(ma['time_s'], 0)} |")
L += ["", "## Paired bootstrap (per-question correctness averaged over seeds; 95% CI of A − B, points)", "",
      "| task | A | B | A − B | 95% CI | p |", "|---|---|---|---|---|---|"]
for k, v in sig.items():
    t, a, b = k.split("|")
    L.append(f"| {t} | {LABEL[a]} | {LABEL[b]} | {v['diff']:+.2f} | [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] | {v['p']} |")
L += ["", "## Learning curves (mean over seeds) and time to 95% of tuned Full FT", ""]
for k, v in curve_out.items():
    L.append(f"- {k}: " + ", ".join(f"{p['frac']:.2f}→{p['acc']} ({p['time_s']:.0f}s)" for p in v))
L.append("")
for k, v in ttt.items():
    L.append(f"- {k}: target {v['target']}, time {v['time_to_95_s']} s, full tuned run {v['full_run_time_s']} s, "
             f"speed-up {v['speedup_vs_full_run']}x")
L += ["", "## 3 epochs (seed 42, eval each epoch)", ""]
for k, v in ep3.items():
    L.append(f"- {k}: {v['per_epoch']}  MMLU {v['mmlu']}  MATH-500 {v['math500']}  time {v['time_s']} s")
L += ["", "## Learning rate / optimizer (seed 42)", ""]
for k, v in lr.items():
    L.append(f"- {k}: {v['score']}  MMLU {v['mmlu']}  MATH-500 {v['math500']}")
L += ["", "## Qwen2.5-0.5B", "", "| method | CS | GSM8K | MATH-500 | MMLU CS / Math | mem CS/Math | time CS/Math |",
      "|---|---|---|---|---|---|---|"]
for m in ("base", "full", "lora", "qlora", "dora", "pissa", "vera", "ia3"):
    a, b = small[f"commonsense|{m}"], small[f"math|{m}"]
    L.append(f"| {m} | {f(a['score'])} | {f(b['score'])} | {f(b['math500'])} | {f(a['mmlu'])} / {f(b['mmlu'])} | "
             f"{f(a['mem_gb'])} / {f(b['mem_gb'])} | {f(a['time_s'], 0)} / {f(b['time_s'], 0)} |")
L += ["", "## Same micro-batch (4) memory / time", ""]
for k, v in parity.items():
    L.append(f"- {k}: LoRA {v['lora_mem']} GB vs Full {v['full_mem']} GB (−{v['mem_saving_pct']}%), "
             f"time LoRA {v['lora_time_s']} s vs Full {v['full_time_s']} s")
open(os.path.join(OUT, "r3_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
