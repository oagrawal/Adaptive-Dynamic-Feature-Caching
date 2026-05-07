#!/bin/bash
set -euo pipefail

docker exec -e PYTHONPATH=/nfs/oagrawal/wan/Wan2.1 -e CUDA_VISIBLE_DEVICES=3 hv_eval_wan \
  python3 /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/batch_generate_wan_teacache.py \
    --start-idx 25 \
    --end-idx 33 \
  2>&1 | tee /nfs/oagrawal/wan/Wan2.1/vbench_eval_teacache/logs/gpu3.log
