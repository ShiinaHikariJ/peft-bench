#!/bin/bash
# runs after run_r3.sh and run_r3b.sh finish
cd /root/autodl-tmp/bench
while pgrep -f "run_r3.sh|run_r3b.sh" > /dev/null; do sleep 120; done
SKIP_EXTRA=1 JOBS=jobs_parity.txt OUT=/root/autodl-tmp/outputs/r3 bash queue2.sh
SKIP_EXTRA=1 BENCH_MODEL=/root/autodl-tmp/models/Qwen2.5-0.5B JOBS=jobs_parity.txt OUT=/root/autodl-tmp/outputs/r3_05b bash queue2.sh
echo R3C_DONE
