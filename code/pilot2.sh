#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/bench
P=/root/autodl-tmp/outputs/pilot2
run(){ n=$1; shift; mkdir -p $P/$n; python run.py --out_dir $P/$n "$@" > $P/$n/run.log 2>&1 && echo "OK $n" || echo "FAIL $n"; }
run math_lora   --method lora --task math --max_train 64 --max_eval 16
run prefix_5e-3 --method prefix --task commonsense --max_train 960 --max_eval 16
run pmlp_1e-3   --method prefix_mlp --task commonsense --max_train 960 --max_eval 16
run pmlp_5e-3   --method prefix_mlp --task commonsense --max_train 960 --max_eval 16 --lr 5e-3
run lora_ref    --method lora --task commonsense --max_train 960 --max_eval 16
echo PILOT2_DONE
