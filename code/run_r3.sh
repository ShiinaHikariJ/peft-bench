#!/bin/bash
# round 3: re-evaluate round-1 seed-42 checkpoints (adds per-question correctness), then 1.5B jobs, then 0.5B jobs
export PATH=/root/miniconda3/bin:$PATH TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/bench
B=/root/autodl-tmp/outputs/bench
for t in commonsense math; do for m in full lora qlora dora pissa vera; do
  d=$B/$t/$m
  [ -f $d/correct.json ] && continue
  [ -f $d/results_r1.json ] || cp $d/results.json $d/results_r1.json
  python run.py --method $m --task $t --out_dir $d --eval_only > $d/reeval.log 2>&1 \
    && echo "[$(date '+%F %T')] REEVAL $t/$m $(python -c "import json;print(json.load(open('$d/results_r1.json'))['eval']['avg'],json.load(open('$d/results.json'))['eval']['avg'])")" \
    || echo "[$(date '+%F %T')] REEVAL FAIL $t/$m"
done; done
JOBS=jobs_r3.txt OUT=/root/autodl-tmp/outputs/r3 bash queue2.sh
BENCH_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B JOBS=jobs_05b.txt OUT=/root/autodl-tmp/outputs/r3_05b bash queue2.sh
echo ALL_DONE
