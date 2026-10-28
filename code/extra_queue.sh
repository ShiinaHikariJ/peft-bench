#!/bin/bash
# Round 2: MMLU on every checkpoint, MATH-500 on math checkpoints, then PiSSA re-run at the LoRA learning rate.
# Skips anything already in extra_results.json / results.json, so it can be restarted safely.
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/bench
OUT=/root/autodl-tmp/outputs/bench
OUT2=/root/autodl-tmp/outputs/bench_pissa_lr2e-4
METHODS="base lora qlora dora full pissa rslora loraplus adalora vera ia3 prefix_mlp prefix prompt oft lokr freeze"

extra() {  # method task out_dir benches
  echo "[$(date '+%F %T')] EXTRA $2/$1 $4"
  python extra_eval.py --method $1 --task $2 --out_dir $3 --benches $4 >> $3/extra.log 2>&1 \
    && echo "[$(date '+%F %T')] OK $2/$1: $(grep -E '^\[(mmlu|math500)\]' $3/extra.log | tail -2 | cut -d' ' -f1-3 | tr '\n' ' ')" \
    || echo "[$(date '+%F %T')] FAIL $2/$1: $(grep -E 'Error' $3/extra.log | tail -1 | cut -c1-200)"
}

for m in $METHODS; do
  extra $m math $OUT/math/$m mmlu,math500
  [ $m != base ] && extra $m commonsense $OUT/commonsense/$m mmlu  # base is the same model under both tasks
done

METHODS=pissa OUT=$OUT2 EXTRA="--lr 2e-4" bash queue.sh
extra pissa math $OUT2/math/pissa mmlu,math500
extra pissa commonsense $OUT2/commonsense/pissa mmlu
echo EXTRA_DONE
