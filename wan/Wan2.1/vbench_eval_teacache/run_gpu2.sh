#!/bin/bash
set -euo pipefail

docker exec -e PYTHONPATH=/nfs/oagrawal/wan/Wan2.1 -e CUDA_VISIBLE_DEVICES=2 hv_eval_wan \
  python3 /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/batch_generate_wan_teacache.py \
    --start-idx 17 \
    --end-idx 25 \
  2>&1 | tee /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/logs/gpu2.log
