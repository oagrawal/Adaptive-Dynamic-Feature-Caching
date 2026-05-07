#!/bin/bash
# GPU 0 — easycache_fixed_0.060 + easycache_adaptive_0.025_0.075_f12l10
# Run INSIDE the `hunyuanvideo` docker container at /workspace.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
mkdir -p /workspace/vbench_eval_easycache/logs

LOG=/workspace/vbench_eval_easycache/logs/ec_gpu0_new_modes.log

cd /workspace
export PYTHONPATH=/workspace:${PYTHONPATH:-}
python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --flow-reverse \
  --use-cpu-offload \
  --modes easycache_fixed_0.060,easycache_adaptive_0.025_0.075_f12l10 \
  --start-idx 0 --end-idx 33 \
  2>&1 | tee "$LOG"
