#!/bin/bash
# GPU 3 — easycache_adaptive_0.050_0.120_f15l6
# Run INSIDE the `hunyuanvideo` docker container at /workspace.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
mkdir -p /workspace/vbench_eval_easycache/logs

LOG=/workspace/vbench_eval_easycache/logs/ec_gpu3_phase3.log

cd /workspace
export PYTHONPATH=/workspace:${PYTHONPATH:-}
python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --flow-reverse \
  --use-cpu-offload \
  --prompts-file vbench_eval_teacache/prompts_subset.json \
  --modes easycache_adaptive_0.050_0.120_f15l6 \
  --start-idx 0 --end-idx 33 \
  2>&1 | tee "$LOG"
