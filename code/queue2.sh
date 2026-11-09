#!/bin/bash
# Round-3 queue. Each line of JOBS: <name> <method> <task> [run.py args...]   ('#' lines ignored)
#   name = sub-directory under OUT; results.json / extra_results.json already present -> step skipped.
# After training: extra benchmarks (MMLU; + MATH-500 for math), then full/freeze weights are deleted to save disk
# (results, curves and per-question correctness are kept).
# env: JOBS (file), OUT, BENCH_MODEL (base model path, read by run.py / extra_eval.py)
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/bench
JOBS=${JOBS:?set JOBS}
OUT=${OUT:-/root/autodl-tmp/outputs/r3}
grep -vE '^\s*(#|$)' "$JOBS" | while read -r name method task rest; do
  d=$OUT/$name
  mkdir -p "$d"
  if [ ! -f "$d/results.json" ]; then
    echo "[$(date '+%F %T')] TRAIN $name"
    if ! python run.py --method "$method" --task "$task" --out_dir "$d" $rest > "$d/run.log" 2>&1; then
      echo "[$(date '+%F %T')] FAIL $name: $(grep -E 'Error|error' "$d/run.log" | tail -1 | cut -c1-200)"
      continue
    fi
    echo "[$(date '+%F %T')] OK $name $(python -c "import json;r=json.load(open('$d/results.json'));print(r['eval']['avg'],r.get('train_runtime_s'),r.get('peak_mem_alloc_gb'))")"
  fi
  if [ ! -f "$d/extra_results.json" ] && [ "${SKIP_EXTRA:-0}" != 1 ]; then
    benches=mmlu; [ "$task" = math ] && benches=mmlu,math500
    if python extra_eval.py --method "$method" --task "$task" --out_dir "$d" --benches $benches > "$d/extra.log" 2>&1; then
      echo "[$(date '+%F %T')] EXTRA OK $name"
    else
      echo "[$(date '+%F %T')] EXTRA FAIL $name: $(grep -E 'Error|error' "$d/extra.log" | tail -1 | cut -c1-200)"
    fi
  fi
  if [ -f "$d/extra_results.json" ] && [ -d "$d/model" ]; then rm -rf "$d/model"; fi
done
echo "[$(date '+%F %T')] QUEUE_DONE $JOBS"
