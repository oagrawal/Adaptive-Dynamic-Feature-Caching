#!/bin/bash
# GPU 2 — all phase4 modes, prompts 17-25
# Modes: fixed_0.040, fixed_0.045, adaptive_0.030_0.060_f4l10, adaptive_0.035_0.065_f4l10
# Run INSIDE the `hunyuanvideo` docker container at /workspace.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
mkdir -p /workspace/vbench_eval_easycache/logs

LOG=/workspace/vbench_eval_easycache/logs/ec_gpu2_phase4p.log

if [ -f "$LOG" ]; then
  echo "ERROR: $LOG already exists. Aborting to avoid overwrite." >&2
  exit 1
fi

cd /workspace
export PYTHONPATH=/workspace:${PYTHONPATH:-}
python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --flow-reverse \
  --use-cpu-offload \
  --prompts-file vbench_eval_teacache/prompts_subset.json \
  --modes easycache_fixed_0.040,easycache_fixed_0.045,easycache_adaptive_0.030_0.060_f4l10,easycache_adaptive_0.035_0.065_f4l10 \
  --start-idx 17 --end-idx 25 \
  2>&1 | tee "$LOG"
