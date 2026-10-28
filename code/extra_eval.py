"""Evaluate a trained (method, task) checkpoint on extra benchmarks; writes extra_results.json in out_dir.

  mmlu    : 5-shot, plain MMLU prompt, pick the highest next-token logit among " A".." D" (no generation)
  math500 : Alpaca prompt + greedy generation (same as GSM8K), answer checked with math-verify

python extra_eval.py --method lora --task math --out_dir /root/autodl-tmp/outputs/bench/math/lora --benches mmlu,math500
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import DATA, MERGEABLE, MODEL, build_prompt, generate_all, load_base, read_jsonl  # noqa: E402

MMLU_CATS = {
    "STEM": "abstract_algebra anatomy astronomy college_biology college_chemistry college_computer_science "
            "college_mathematics college_physics computer_security conceptual_physics electrical_engineering "
            "elementary_mathematics high_school_biology high_school_chemistry high_school_computer_science "
            "high_school_mathematics high_school_physics high_school_statistics machine_learning",
    "humanities": "formal_logic high_school_european_history high_school_us_history high_school_world_history "
                  "international_law jurisprudence logical_fallacies moral_disputes moral_scenarios philosophy "
                  "prehistory professional_law world_religions",
    "social_sciences": "econometrics high_school_geography high_school_government_and_politics "
                       "high_school_macroeconomics high_school_microeconomics high_school_psychology human_sexuality "
                       "professional_psychology public_relations security_studies sociology us_foreign_policy",
}
SUBJECT_CAT = {s: c for c, subs in MMLU_CATS.items() for s in subs.split()}
LETTERS = "ABCD"
MAX_PROMPT = 2048
TOKEN_BUDGET = 12288  # tokens per forward batch (full-vocab logits are the memory hog)


def load_model(method, out_dir):
    if method in ("full", "freeze"):
        return AutoModelForCausalLM.from_pretrained(os.path.join(out_dir, "model"), dtype=torch.bfloat16,
                                                    device_map={"": 0}).eval()
    model = load_base(method)
    if method == "base":
        return model.eval()
    model = PeftModel.from_pretrained(model, os.path.join(out_dir, "adapter"))
    if method in MERGEABLE or method == "pissa":  # saved PiSSA adapter is already converted to a plain LoRA
        model = model.merge_and_unload()
    return model.eval()


def mmlu_block(r, with_answer):
    s = r["question"].strip() + "\n" + "".join(f"{l}. {c}\n" for l, c in zip(LETTERS, r["choices"])) + "Answer:"
    return s + (f" {LETTERS[r['answer']]}\n\n" if with_answer else "")


@torch.inference_mode()
def eval_mmlu(model, tok, max_eval=None):
    test, dev = read_jsonl(f"{DATA}/mmlu_test.jsonl"), read_jsonl(f"{DATA}/mmlu_dev.jsonl")
    if max_eval:
        test = test[::max(1, len(test) // max_eval)][:max_eval]
    shots = defaultdict(list)
    for r in dev:
        shots[r["subject"]].append(r)
    prompts = []
    for r in test:
        head = f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
        q = mmlu_block(r, False)
        k = 5
        while True:
            ids = tok(head + "".join(mmlu_block(s, True) for s in shots[r["subject"]][:k]) + q,
                      add_special_tokens=False)["input_ids"]
            if len(ids) <= MAX_PROMPT or k == 0:
                break
            k -= 1
        prompts.append(ids[-MAX_PROMPT:])
    letter_ids = [tok(f" {l}", add_special_tokens=False)["input_ids"][-1] for l in LETTERS]

    # right padding, like training; prompt-learning methods prepend virtual tokens, handled via offset
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    preds = [None] * len(prompts)
    s = 0
    while s < len(order):
        n = max(1, TOKEN_BUDGET // len(prompts[order[s]]))
        idx = order[s:s + n]
        L = len(prompts[idx[0]])
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(prompts[i])] = torch.tensor(prompts[i])
            attn[j, :len(prompts[i])] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=attn.cuda()).logits
        off = logits.shape[1] - L
        for j, i in enumerate(idx):
            preds[i] = int(logits[j, off + len(prompts[i]) - 1, letter_ids].argmax())
        del logits
        s += n
    by_sub, by_cat = defaultdict(list), defaultdict(list)
    for r, p in zip(test, preds):
        by_sub[r["subject"]].append(p == r["answer"])
        by_cat[SUBJECT_CAT.get(r["subject"], "other")].append(p == r["answer"])
    pct = lambda xs: round(100 * sum(xs) / len(xs), 2)  # noqa: E731
    allc = [p == r["answer"] for r, p in zip(test, preds)]
    return {"avg": pct(allc), **{c: pct(v) for c, v in sorted(by_cat.items())},
            "subjects": {k: pct(v) for k, v in sorted(by_sub.items())},
            "pred_dist": {LETTERS[k]: preds.count(k) for k in range(4)}, "n": len(test)}


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def extract_math(text):
    m = re.search(r"The answer is:?\s*(.+)", text)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    return last_boxed(text) or ""


def norm(s):
    s = s.strip().strip("$").replace("\\!", "").replace("\\left", "").replace("\\right", "").replace("dfrac", "frac")
    s = s.replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace(" ", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    return s.rstrip(".")


def math_equal(pred, gold):
    if not pred:
        return False
    if norm(pred) == norm(gold):
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


def eval_math500(model, tok, out_dir, max_eval=None):
    rows = read_jsonl(f"{DATA}/test_math500.jsonl", max_eval)
    bs = 32 if isinstance(model, PeftModel) else 128  # unmerged adapters (4-bit LoRA, prompt learning) need more memory
    preds = generate_all(model, tok, [build_prompt(r) for r in rows], bs, 1024)
    ext = [extract_math(p) for p in preds]
    correct = [math_equal(e, r["answer"]) for e, r in zip(ext, rows)]
    by_level = defaultdict(list)
    for r, c in zip(rows, correct):
        by_level[f"level{r['level']}"].append(c)
    json.dump([{"q": r["instruction"][-200:], "pred_tail": p[-300:], "extracted": e, "answer": r["answer"],
                "correct": c} for r, p, e, c in list(zip(rows, preds, ext, correct))[:30]],
              open(os.path.join(out_dir, "math500_samples.json"), "w"), ensure_ascii=False, indent=1)
    return {"avg": round(100 * sum(correct) / len(rows), 2),
            **{k: round(100 * sum(v) / len(v), 2) for k, v in sorted(by_level.items())}, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--benches", default="mmlu,math500")
    ap.add_argument("--max_eval", type=int, default=None, help="smoke test: print only, nothing written")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    path = os.path.join(args.out_dir, "extra_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {"method": args.method, "task": args.task}
    todo = [b for b in args.benches.split(",") if args.force or args.max_eval or b not in res]
    if not todo:
        print("nothing to do")
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.method, args.out_dir)
    if hasattr(model, "config"):
        model.config.use_cache = True
    for b in todo:
        t0 = time.time()
        out_dir = "/tmp" if args.max_eval else args.out_dir
        r = eval_mmlu(model, tok, args.max_eval) if b == "mmlu" else eval_math500(model, tok, out_dir, args.max_eval)
        r["runtime_s"] = round(time.time() - t0, 1)
        print(f"[{b}] {args.task}/{args.method}: {r['avg']} n={r['n']} ({r['runtime_s']}s) "
              f"{ {k: v for k, v in r.items() if k not in ('subjects',)} }", flush=True)
        if args.max_eval:
            continue
        res[b] = r
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
