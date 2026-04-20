#!/bin/bash
set -euo pipefail
export PYTHONPATH=/nfs/oagrawal/wan/Wan2.1
export CUDA_VISIBLE_DEVICES=0

cd /nfs/oagrawal/wan/Wan2.1

python3 dicache_results/batch_generate_wan_dicache.py \
  --start-idx 0 \
  --end-idx 17 \
  --modes all \
  2>&1 | tee dicache_results/logs/gpu0_fixed_sweep.log
