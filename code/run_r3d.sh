#!/bin/bash
# runs after the other round-3 queues; re-evaluates round-1 seed 42 for per-question correctness, then seeds 43, 44
export PATH=/root/miniconda3/bin:$PATH TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/bench
while pgrep -f "run_r3.sh|run_r3b.sh|run_r3c.sh" > /dev/null; do sleep 120; done
B=/root/autodl-tmp/outputs/bench
for t in commonsense math; do for m in oft adalora lokr; do
  d=$B/$t/$m
  [ -f $d/correct.json ] && continue
  [ -f $d/results_r1.json ] || cp $d/results.json $d/results_r1.json
  python run.py --method $m --task $t --out_dir $d --eval_only > $d/reeval.log 2>&1 \
    && echo "[$(date '+%F %T')] REEVAL $t/$m $(python -c "import json;print(json.load(open('$d/results_r1.json'))['eval']['avg'],json.load(open('$d/results.json'))['eval']['avg'])")" \
    || echo "[$(date '+%F %T')] REEVAL FAIL $t/$m"
done; done
JOBS=jobs_r3d.txt OUT=/root/autodl-tmp/outputs/r3 bash queue2.sh
echo R3D_DONE
