#!/bin/bash
set -euo pipefail
export PYTHONPATH=/nfs/oagrawal/wan/Wan2.1
export CUDA_VISIBLE_DEVICES=0

docker exec -e PYTHONPATH=/nfs/oagrawal/wan/Wan2.1 -e CUDA_VISIBLE_DEVICES=0 hv_eval_wan \
  python3 /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/batch_generate_wan_teacache.py \
    --start-idx 0 \
    --end-idx 9 \
  2>&1 | tee /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/logs/gpu0.log
