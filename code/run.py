"""Unified fine-tuning benchmark: one method x one task -> train, evaluate, write results.json.

python run.py --method lora --task commonsense --out_dir /root/autodl-tmp/outputs/bench/commonsense/lora
"""
import argparse
import gc
import json
import os
import re
import time

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
                          TrainingArguments)
from peft import (AdaLoraConfig, IA3Config, LoKrConfig, LoraConfig, OFTConfig, PeftModel, PrefixTuningConfig,
                  PromptTuningConfig, PromptTuningInit, TaskType, VeraConfig, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.optimizers import create_loraplus_optimizer

DATA = "/root/autodl-tmp/data"
MODEL = "/root/autodl-tmp/models/Qwen2.5-1.5B"
EFFECTIVE_BS = 32

TEMPLATE = ("Below is an instruction that describes a task. Write a response that appropriately completes the request."
            "\n\n### Instruction:\n{instruction}\n\n### Response:\n")
TEMPLATE_IN = ("Below is an instruction that describes a task, paired with an input that provides further context. "
               "Write a response that appropriately completes the request.\n\n"
               "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

TASKS = {
    "commonsense": dict(train="cs_train.jsonl", max_len=384, max_new_tokens=32, eval_bs=128,
                        tests=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
                               "ARC-Easy", "ARC-Challenge", "openbookqa"]),
    "math": dict(train="math_train.jsonl", max_len=768, max_new_tokens=512, eval_bs=128, tests=["gsm8k"]),
}

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# default hyper-parameters follow each method's paper / official PEFT examples
METHODS = {
    "base":     dict(lr=None),
    "full":     dict(lr=2e-5, micro_bs=4, optim="paged_adamw_8bit"),
    "freeze":   dict(lr=5e-5),
    "lora":     dict(lr=2e-4),
    "qlora":    dict(lr=2e-4),
    "dora":     dict(lr=2e-4),
    "rslora":   dict(lr=2e-4),
    "pissa":    dict(lr=2e-5),
    "loraplus": dict(lr=1e-4),
    "adalora":  dict(lr=5e-4),
    "vera":     dict(lr=2e-3),
    "ia3":      dict(lr=3e-3),
    "prefix":   dict(lr=5e-3, gc=False),
    "prefix_mlp": dict(lr=1e-3, gc=False),
    "prompt":   dict(lr=3e-2, gc=False),
    "oft":      dict(lr=2e-4),
    "lokr":     dict(lr=2e-4),
}
MERGEABLE = {"lora", "dora", "rslora", "loraplus", "adalora", "vera", "ia3", "oft", "lokr"}
NO_CACHE_EVAL = set()  # greedy_no_cache was checked on prefix tuning: same accuracy as generate()


def build_prompt(ex):
    if ex.get("input"):
        return TEMPLATE_IN.format(instruction=ex["instruction"], input=ex["input"])
    return TEMPLATE.format(instruction=ex["instruction"])


def read_jsonl(path, n=None):
    rows = [json.loads(line) for line in open(path)]
    return rows[:n] if n else rows


class SFTData(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items, self.truncated = [], 0
        for ex in rows:
            p = tok(build_prompt(ex), add_special_tokens=False)["input_ids"]
            r = tok(ex["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids, labels = p + r, [-100] * len(p) + r
            if len(ids) > max_len:
                ids, labels = ids[:max_len], labels[:max_len]
                self.truncated += 1
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        n = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (ids, lab) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            labels[i, :len(lab)] = torch.tensor(lab)
            attn[i, :len(ids)] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
    return collate


class AdaLoraCallback(TrainerCallback):
    """AdaLoRA needs update_and_allocate() after every optimizer step, while gradients still exist."""

    def __init__(self, model):
        self.model = model

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.model.base_model.update_and_allocate(state.global_step)


def peft_config(method, args, total_steps):
    r, alpha = args.rank, 2 * args.rank
    if method in ("lora", "qlora", "loraplus"):
        return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=ALL_LINEAR, task_type=TaskType.CAUSAL_LM)
    if method == "dora":
        return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=ALL_LINEAR, use_dora=True,
                          task_type=TaskType.CAUSAL_LM)
    if method == "rslora":
        return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=ALL_LINEAR, use_rslora=True,
                          task_type=TaskType.CAUSAL_LM)
    if method == "pissa":
        # PiSSA recommends alpha == r and no dropout
        return LoraConfig(r=r, lora_alpha=r, lora_dropout=0.0, target_modules=ALL_LINEAR,
                          init_lora_weights="pissa_niter_16", task_type=TaskType.CAUSAL_LM)
    if method == "adalora":
        return AdaLoraConfig(init_r=int(r * 1.5), target_r=r, lora_alpha=alpha, lora_dropout=0.05,
                             target_modules=ALL_LINEAR, total_step=total_steps, tinit=int(0.1 * total_steps),
                             tfinal=int(0.2 * total_steps), deltaT=10, task_type=TaskType.CAUSAL_LM)
    if method == "vera":
        return VeraConfig(r=256, target_modules=ALL_LINEAR, vera_dropout=0.0, d_initial=0.1, task_type=TaskType.CAUSAL_LM)
    if method == "ia3":
        return IA3Config(target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"],
                         task_type=TaskType.CAUSAL_LM)
    if method == "prefix":
        return PrefixTuningConfig(num_virtual_tokens=32, task_type=TaskType.CAUSAL_LM)
    if method == "prefix_mlp":
        # original Prefix-Tuning reparameterization: prefix produced by an MLP, more stable than raw embeddings
        return PrefixTuningConfig(num_virtual_tokens=32, prefix_projection=True, encoder_hidden_size=512,
                                  task_type=TaskType.CAUSAL_LM)
    if method == "prompt":
        return PromptTuningConfig(num_virtual_tokens=32, prompt_tuning_init=PromptTuningInit.TEXT,
                                  prompt_tuning_init_text="Read the question carefully and answer it correctly.",
                                  tokenizer_name_or_path=MODEL, task_type=TaskType.CAUSAL_LM)
    if method == "oft":
        return OFTConfig(r=0, oft_block_size=32, target_modules=ALL_LINEAR, task_type=TaskType.CAUSAL_LM)
    if method == "lokr":
        return LoKrConfig(r=r, alpha=alpha, target_modules=ALL_LINEAR, task_type=TaskType.CAUSAL_LM)
    raise ValueError(method)


def load_base(method):
    if method == "qlora":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, dtype=torch.bfloat16,
                                                    device_map={"": 0})
    # full / freeze keep fp32 master weights (bf16 autocast); PEFT methods use a bf16 frozen base
    dtype = torch.float32 if method in ("full", "freeze") else torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map={"": 0})


def count_params(model):
    if isinstance(model, PeftModel):
        return model.get_nb_trainable_parameters()
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total


# ---------------------------------------------------------------- evaluation
CS_PATTERNS = {"boolq": r"true|false", "piqa": r"solution1|solution2", "social_i_qa": r"answer1|answer2|answer3",
               "hellaswag": r"ending1|ending2|ending3|ending4", "winogrande": r"option1|option2"}


def extract_cs(task, text):
    m = re.findall(CS_PATTERNS.get(task, r"answer1|answer2|answer3|answer4|answer5"), text.strip().lower())
    return m[0] if m else ""


def extract_num(text):
    m = re.search(r"The answer is:?\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


@torch.no_grad()
def greedy_no_cache(model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id):
    """Greedy decoding with a full forward pass per step (no KV cache).

    Used for prefix tuning: PEFT's generate() path for prefix tuning rewrites the cache / attention mask and gives
    outputs inconsistent with the training forward pass; a plain forward is exactly what training optimised.
    """
    done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    new_tokens = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                       logits_to_keep=1).logits[:, -1, :]
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        new_tokens.append(nxt)
        done |= nxt == eos_id
        input_ids = torch.cat([input_ids, nxt[:, None]], dim=1)
        attention_mask = torch.cat([attention_mask, (~done | (nxt == eos_id)).long()[:, None]], dim=1)
        if done.all():
            break
    return torch.stack(new_tokens, dim=1)


@torch.no_grad()
def generate_all(model, tok, prompts, bs, max_new_tokens, no_cache=False):
    tok.padding_side = "left"
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        if no_cache:
            new = greedy_no_cache(model, enc["input_ids"], enc["attention_mask"], max_new_tokens,
                                  tok.eos_token_id, tok.pad_token_id)
        else:
            gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            outs[i] = tok.decode(new[j], skip_special_tokens=True)
    return outs


def evaluate(model, tok, task, cfg, max_eval, out_dir, no_cache=False):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    metrics, samples = {}, {}
    bs = cfg["eval_bs"] // (4 if no_cache else 1)
    for t in cfg["tests"]:
        rows = read_jsonl(os.path.join(DATA, f"test_{t}.jsonl"), max_eval)
        preds = generate_all(model, tok, [build_prompt(r) for r in rows], bs, cfg["max_new_tokens"], no_cache)
        if task == "math":
            correct = []
            for r, p in zip(rows, preds):
                v = extract_num(p)
                correct.append(v is not None and abs(v - float(r["answer"])) < 1e-4)
        else:
            correct = [extract_cs(t, p) == r["answer"] for r, p in zip(rows, preds)]
        metrics[t] = round(100 * sum(correct) / len(rows), 2)
        samples[t] = [{"prompt_tail": build_prompt(r)[-200:], "pred": p, "answer": r["answer"], "correct": c}
                      for r, p, c in list(zip(rows, preds, correct))[:20]]
        print(f"[eval] {t}: {metrics[t]} ({len(rows)} samples)", flush=True)
    metrics["avg"] = round(sum(metrics[t] for t in cfg["tests"]) / len(cfg["tests"]), 2)
    json.dump(samples, open(os.path.join(out_dir, "eval_samples.json"), "w"), ensure_ascii=False, indent=1)
    return metrics


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", action="store_true", help="re-evaluate the saved adapter/model in out_dir")
    args = ap.parse_args()
    if args.eval_only:
        return eval_only(args)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg, mcfg = TASKS[args.task], METHODS[args.method]
    lr = args.lr or mcfg["lr"]
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_base(args.method)
    result = dict(method=args.method, task=args.task, model=os.path.basename(MODEL), lr=lr, rank=args.rank,
                  seed=args.seed, epochs=args.epochs)

    if args.method != "base":
        tok.padding_side = "right"
        train = SFTData(read_jsonl(os.path.join(DATA, cfg["train"]), args.max_train), tok, cfg["max_len"])
        use_gc = mcfg.get("gc", True)
        micro_bs = mcfg.get("micro_bs", 8)
        if not use_gc and args.task == "math":
            micro_bs = 4  # no gradient checkpointing + 768-token sequences does not fit 24GB at micro_bs=8
        accum = EFFECTIVE_BS // micro_bs
        total_steps = max(1, int(len(train) * args.epochs) // EFFECTIVE_BS)

        if args.method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc,
                                                    gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.method == "freeze":
            n_train_layers = 2
            for p in model.parameters():
                p.requires_grad = False
            for layer in model.model.layers[-n_train_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            result["freeze_trainable_layers"] = n_train_layers
        elif args.method != "full":
            model = get_peft_model(model, peft_config(args.method, args, total_steps))
            if args.method == "pissa":
                # save the init adapter so the final adapter can be converted to a plain LoRA on the original base
                model.peft_config["default"].init_lora_weights = True
                model.save_pretrained(os.path.join(args.out_dir, "pissa_init"))
                model.peft_config["default"].init_lora_weights = "pissa_niter_16"
        if use_gc and args.method != "qlora":
            model.config.use_cache = False

        trainable, total = count_params(model)
        result.update(trainable_params=trainable, total_params=total, trainable_pct=round(100 * trainable / total, 4),
                      n_train=len(train), truncated=train.truncated, total_steps=total_steps, micro_bs=micro_bs)
        print(f"[train] {args.method}/{args.task} trainable {trainable:,} / {total:,} "
              f"({result['trainable_pct']}%) steps={total_steps} lr={lr}", flush=True)

        targs = TrainingArguments(
            output_dir=args.out_dir, per_device_train_batch_size=micro_bs, gradient_accumulation_steps=accum,
            learning_rate=lr, num_train_epochs=args.epochs, lr_scheduler_type="cosine",
            warmup_steps=max(1, int(0.03 * total_steps)), weight_decay=0.0, max_grad_norm=1.0, bf16=True,
            optim=mcfg.get("optim", "adamw_torch"), gradient_checkpointing=use_gc and args.method != "qlora",
            gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10, save_strategy="no",
            report_to="none", remove_unused_columns=False, dataloader_num_workers=2, seed=args.seed,
            disable_tqdm=True)
        optimizers = (None, None)
        if args.method == "loraplus":
            optimizers = (create_loraplus_optimizer(model, optimizer_cls=torch.optim.AdamW, lr=lr,
                                                    loraplus_lr_ratio=16.0), None)
        callbacks = [AdaLoraCallback(model)] if args.method == "adalora" else []
        trainer = Trainer(model=model, args=targs, train_dataset=train, data_collator=make_collate(tok.pad_token_id),
                          optimizers=optimizers, callbacks=callbacks)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = trainer.train()
        result.update(train_runtime_s=round(time.time() - t0, 1), train_loss=round(out.training_loss, 4),
                      train_samples_per_s=round(len(train) * args.epochs / (time.time() - t0), 2),
                      peak_mem_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                      peak_mem_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
        json.dump(trainer.state.log_history, open(os.path.join(args.out_dir, "log_history.json"), "w"), indent=1)
        print(f"[train] done {result['train_runtime_s']}s loss={result['train_loss']} "
              f"peak={result['peak_mem_alloc_gb']}GB", flush=True)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if args.method in ("full", "freeze"):
            model = model.to(torch.bfloat16)
            model.save_pretrained(os.path.join(args.out_dir, "model"))
        elif args.method != "pissa":
            model.save_pretrained(os.path.join(args.out_dir, "adapter"))
        if args.method in MERGEABLE:
            try:
                model = model.merge_and_unload()
                result["merged_for_eval"] = True
            except Exception as e:  # evaluation still works unmerged, just slower
                print(f"[eval] merge failed ({e}); evaluating unmerged", flush=True)
                result["merged_for_eval"] = False
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    else:
        total = sum(p.numel() for p in model.parameters())
        result.update(trainable_params=0, total_params=total, trainable_pct=0.0)

    t0 = time.time()
    no_cache = args.method in NO_CACHE_EVAL
    result["eval"] = evaluate(model, tok, args.task, cfg, args.max_eval, args.out_dir, no_cache)
    result["eval_runtime_s"] = round(time.time() - t0, 1)
    result["eval_decoding"] = "greedy_no_cache" if no_cache else "generate"

    if args.method == "pissa":
        model.save_pretrained(os.path.join(args.out_dir, "adapter"),
                              path_initial_model_for_weight_conversion=os.path.join(args.out_dir, "pissa_init"))

    json.dump(result, open(os.path.join(args.out_dir, "results.json"), "w"), indent=1)
    print("[result]", json.dumps(result), flush=True)


def eval_only(args):
    cfg = TASKS[args.task]
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.method in ("full", "freeze"):
        model = AutoModelForCausalLM.from_pretrained(os.path.join(args.out_dir, "model"), dtype=torch.bfloat16,
                                                     device_map={"": 0})
    else:
        model = load_base(args.method)
        if args.method != "base":
            model = PeftModel.from_pretrained(model, os.path.join(args.out_dir, "adapter"))
    path = os.path.join(args.out_dir, "results.json")
    result = json.load(open(path)) if os.path.exists(path) else dict(method=args.method, task=args.task)
    no_cache = args.method in NO_CACHE_EVAL
    t0 = time.time()
    metrics = evaluate(model, tok, args.task, cfg, args.max_eval, args.out_dir, no_cache)
    if args.max_eval:  # partial re-eval: report only, keep results.json untouched
        print("[eval_only partial]", json.dumps(metrics), flush=True)
        return
    result.update(eval=metrics, eval_runtime_s=round(time.time() - t0, 1),
                  eval_decoding="greedy_no_cache" if no_cache else "generate")
    json.dump(result, open(path, "w"), indent=1)
    print("[result]", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
