#!/bin/bash
# Launch one-prompt EasyCache diagnostics on GPUs 0-3.
# Run from the host. Outputs stay under vbench_eval_easycache/diagnostic_runs.

set -euo pipefail

SESSION="hv_ec_diag"
PROMPT="a person swimming in ocean"
BASE_DIR="/workspace/vbench_eval_easycache/diagnostic_runs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

docker start hunyuanvideo >/dev/null || true

tmux new-session -d -s "$SESSION" -n gpu0 \
  "docker exec hunyuanvideo bash -lc 'cd /workspace && export PYTHONPATH=/workspace:\${PYTHONPATH:-} && export CUDA_VISIBLE_DEVICES=0 && mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/outputs && python3 vbench_eval_easycache/diagnostic_runs/easycache_diagnostic_video.py --easycache-mode baseline --prompt \"${PROMPT}\" --seed 0 --video-size 544 960 --video-length 129 --infer-steps 50 --flow-reverse --use-cpu-offload --save-path ${BASE_DIR}/outputs 2>&1 | tee ${BASE_DIR}/logs/diag_gpu0_baseline_pred_change.log'"

tmux new-window -t "$SESSION" -n gpu1 \
  "docker exec hunyuanvideo bash -lc 'cd /workspace && export PYTHONPATH=/workspace:\${PYTHONPATH:-} && export CUDA_VISIBLE_DEVICES=1 && mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/outputs && python3 vbench_eval_easycache/diagnostic_runs/easycache_diagnostic_video.py --easycache-mode easycache --easycache-thresh 0.040 --prompt \"${PROMPT}\" --seed 0 --video-size 544 960 --video-length 129 --infer-steps 50 --flow-reverse --use-cpu-offload --save-path ${BASE_DIR}/outputs 2>&1 | tee ${BASE_DIR}/logs/diag_gpu1_fixed_0040.log'"

tmux new-window -t "$SESSION" -n gpu2 \
  "docker exec hunyuanvideo bash -lc 'cd /workspace && export PYTHONPATH=/workspace:\${PYTHONPATH:-} && export CUDA_VISIBLE_DEVICES=2 && mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/outputs && python3 vbench_eval_easycache/diagnostic_runs/easycache_diagnostic_video.py --easycache-mode adaptive --easycache-thresh-low 0.025 --easycache-thresh-high 0.060 --easycache-first-steps 12 --easycache-last-steps 10 --prompt \"${PROMPT}\" --seed 0 --video-size 544 960 --video-length 129 --infer-steps 50 --flow-reverse --use-cpu-offload --save-path ${BASE_DIR}/outputs 2>&1 | tee ${BASE_DIR}/logs/diag_gpu2_adapt_0025_0060_f12l10.log'"

tmux new-window -t "$SESSION" -n gpu3 \
  "docker exec hunyuanvideo bash -lc 'cd /workspace && export PYTHONPATH=/workspace:\${PYTHONPATH:-} && export CUDA_VISIBLE_DEVICES=3 && mkdir -p ${BASE_DIR}/logs ${BASE_DIR}/outputs && python3 vbench_eval_easycache/diagnostic_runs/easycache_diagnostic_video.py --easycache-mode adaptive --easycache-thresh-low 0.030 --easycache-thresh-high 0.060 --easycache-first-steps 12 --easycache-last-steps 10 --prompt \"${PROMPT}\" --seed 0 --video-size 544 960 --video-length 129 --infer-steps 50 --flow-reverse --use-cpu-offload --save-path ${BASE_DIR}/outputs 2>&1 | tee ${BASE_DIR}/logs/diag_gpu3_adapt_0030_0060_f12l10.log'"

echo "Launched ${SESSION}. Attach with: tmux attach -t ${SESSION}"
