#!/bin/bash
# GPU 2 — prompts 16-24, all 4 new adaptive modes
# Run INSIDE the hunyuanvideo container at /workspace.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
mkdir -p /workspace/vbench_eval_teacache/logs

LOG=/workspace/vbench_eval_teacache/logs/tc_gpu2.log

cd /workspace
export PYTHONPATH=/workspace:${PYTHONPATH:-}
python3 vbench_eval_teacache/batch_generate.py \
  --video-size 544 960 \
  --video-length 129 \
  --infer-steps 50 \
  --flow-reverse \
  --use-cpu-offload \
  --modes hunyuan_tc_adaptive_lo0.1_hi0.3,hunyuan_tc_adaptive_lo0.15_hi0.3,hunyuan_tc_adaptive_lo0.2_hi0.3,hunyuan_tc_adaptive_lo0.1_hi0.25 \
  --start-idx 16 --end-idx 24 \
  2>&1 | tee "$LOG"
