#!/bin/bash
# GPU 2 — easycache_adaptive_0.010_0.120_f12l4
# Run INSIDE the `hunyuanvideo` docker container at /workspace.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
mkdir -p /workspace/vbench_eval_easycache/logs

LOG=/workspace/vbench_eval_easycache/logs/ec_gpu2_phase3.log

cd /workspace
export PYTHONPATH=/workspace:${PYTHONPATH:-}
python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --flow-reverse \
  --use-cpu-offload \
  --prompts-file vbench_eval_teacache/prompts_subset.json \
  --modes easycache_adaptive_0.010_0.120_f12l4 \
  --start-idx 0 --end-idx 33 \
  2>&1 | tee "$LOG"
