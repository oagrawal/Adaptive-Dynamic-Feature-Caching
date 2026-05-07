#!/bin/bash
set -euo pipefail

docker exec -e PYTHONPATH=/nfs/oagrawal/wan/Wan2.1 -e CUDA_VISIBLE_DEVICES=1 hv_eval_wan \
  python3 /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/batch_generate_wan_teacache.py \
    --start-idx 9 \
    --end-idx 17 \
  2>&1 | tee /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/logs/gpu1.log
