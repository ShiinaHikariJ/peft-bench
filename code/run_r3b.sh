#!/bin/bash
# runs after run_r3.sh finishes (it is not edited while running)
cd /root/autodl-tmp/bench
while pgrep -f run_r3.sh > /dev/null; do sleep 120; done
JOBS=jobs_r3b.txt OUT=/root/autodl-tmp/outputs/r3 bash queue2.sh
echo R3B_DONE
