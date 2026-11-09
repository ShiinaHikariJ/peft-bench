#!/bin/bash
# wait for the 7B download, then run the 7B jobs with smaller eval batches
cd /root/autodl-tmp/bench
while pgrep -f "hf download" > /dev/null; do sleep 60; done
ls /root/autodl-tmp/models/Qwen2.5-7B/*.safetensors > /dev/null || { echo "7B DOWNLOAD FAIL"; exit 1; }
echo "[$(date '+%F %T')] 7B downloaded"
EVAL_BS=32 MATH_BS=24 TOKEN_BUDGET=6144 BENCH_MODEL=/root/autodl-tmp/models/Qwen2.5-7B \
  JOBS=jobs_7b.txt OUT=/root/autodl-tmp/outputs/r3_7b bash queue2.sh
echo R7B_DONE
