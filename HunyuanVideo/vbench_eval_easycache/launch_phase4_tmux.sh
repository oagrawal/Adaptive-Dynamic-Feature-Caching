#!/bin/bash
# Launch phase4 generation across 4 GPUs in a tmux session.
# Run from the HOST (not inside the container).
#
# All GPUs run all 4 modes; split by prompts:
# GPU 0: prompts 0-9
# GPU 1: prompts 9-17
# GPU 2: prompts 17-25
# GPU 3: prompts 25-33
#
# Usage: bash vbench_eval_easycache/launch_phase4_tmux.sh

set -euo pipefail

SESSION="hv_ec_phase4"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

docker start hunyuanvideo 2>/dev/null && echo "Container started." || echo "Container already running."

tmux new-session -d -s "$SESSION" -n "gpu0"
tmux send-keys -t "${SESSION}:gpu0" \
  "docker exec hunyuanvideo bash /workspace/vbench_eval_easycache/run_phase4_gpu0.sh" Enter

tmux new-window -t "$SESSION" -n "gpu1"
tmux send-keys -t "${SESSION}:gpu1" \
  "docker exec hunyuanvideo bash /workspace/vbench_eval_easycache/run_phase4_gpu1.sh" Enter

tmux new-window -t "$SESSION" -n "gpu2"
tmux send-keys -t "${SESSION}:gpu2" \
  "docker exec hunyuanvideo bash /workspace/vbench_eval_easycache/run_phase4_gpu2.sh" Enter

tmux new-window -t "$SESSION" -n "gpu3"
tmux send-keys -t "${SESSION}:gpu3" \
  "docker exec hunyuanvideo bash /workspace/vbench_eval_easycache/run_phase4_gpu3.sh" Enter

tmux select-window -t "${SESSION}:gpu0"

echo ""
echo "Launched. Attach with:"
echo "  tmux attach -t $SESSION"
echo ""
echo "Switch windows: Ctrl-b 0/1/2/3"
echo "Logs (from host):"
echo "  tail -f /nfs/oagrawal/HunyuanVideo/vbench_eval_easycache/logs/ec_gpu{0,1,2,3}_phase4p.log"
