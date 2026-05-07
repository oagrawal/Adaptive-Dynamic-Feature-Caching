#!/bin/bash
# GPU 3 — re-run easycache_adaptive_0.0375_0.050 at 544x960 (was broken at 720x1280).
# Run INSIDE the `hunyuanvideo` docker container at /workspace.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
mkdir -p /workspace/vbench_eval_easycache/logs

LOG=/workspace/vbench_eval_easycache/logs/ec_gpu3_easycache_adaptive_0.0375_0.050_redo.log

cd /workspace
export PYTHONPATH=/workspace:${PYTHONPATH:-}
python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --flow-reverse \
  --use-cpu-offload \
  --modes easycache_adaptive_0.0375_0.050 \
  --start-idx 0 --end-idx 33 \
  2>&1 | tee "$LOG"
