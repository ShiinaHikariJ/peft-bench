#!/bin/bash
# Run every (method, task) pair in order; skip pairs that already have results.json; keep going on failure.
# env overrides: METHODS, TASKS, OUT, EXTRA
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/bench
METHODS=${METHODS:-"base lora qlora dora full pissa rslora loraplus adalora vera ia3 prefix_mlp prefix prompt oft lokr freeze"}
TASKS=${TASKS:-"commonsense math"}
OUT=${OUT:-/root/autodl-tmp/outputs/bench}
for m in $METHODS; do
  for t in $TASKS; do
    d=$OUT/$t/$m
    if [ -f $d/results.json ]; then echo "skip $t/$m"; continue; fi
    mkdir -p $d
    echo "[$(date '+%F %T')] START $t/$m"
    if python run.py --method $m --task $t --out_dir $d $EXTRA > $d/run.log 2>&1; then
      echo "[$(date '+%F %T')] OK $t/$m"
    else
      echo "[$(date '+%F %T')] FAIL $t/$m: $(grep -E 'Error|error' $d/run.log | tail -1 | cut -c1-200)"
    fi
  done
done
python summarize.py $OUT
echo QUEUE_DONE
