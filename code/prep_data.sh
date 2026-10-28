#!/bin/bash
# Download base model, training sets and test sets into /root/autodl-tmp/{models,data}
set -x
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_DISABLE_XET=1
D=/root/autodl-tmp/data
mkdir -p $D/raw

URL=https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset
for t in boolq piqa social_i_qa hellaswag winogrande ARC-Easy ARC-Challenge openbookqa; do
  f=$D/raw/$t.json
  [ -s $f ] || curl -sfL --max-time 90 -o $f $URL/$t/test.json \
    || ( source /etc/network_turbo >/dev/null; curl -sfL --max-time 180 -o $f $URL/$t/test.json ) \
    || { echo "download failed: $t"; exit 1; }
done

ok=0
for attempt in 1 2 3 4 5; do
  python /root/autodl-tmp/bench/prep_data.py && { ok=1; break; }
  echo "prep attempt $attempt failed, retrying"; sleep 20
done
[ $ok = 1 ] || exit 1
ls -la $D
echo PREP_DONE
