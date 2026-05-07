#!/bin/bash
# Launch phase3 fidelity eval across 4 GPUs in a tmux session.
# Run directly on the host (no container needed).
#
# Usage: bash vbench_eval_easycache/launch_fidelity_phase3_tmux.sh

set -euo pipefail

SESSION="hv_ec_fidelity_phase3"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

BASE=/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache
mkdir -p "$BASE/logs"

tmux new-session -d -s "$SESSION" -n "gpu0"
tmux send-keys -t "${SESSION}:gpu0" \
  "bash $BASE/run_fidelity_phase3_gpu0.sh 2>&1 | tee $BASE/logs/fidelity_gpu0_phase3.log" Enter

tmux new-window -t "$SESSION" -n "gpu1"
tmux send-keys -t "${SESSION}:gpu1" \
  "bash $BASE/run_fidelity_phase3_gpu1.sh 2>&1 | tee $BASE/logs/fidelity_gpu1_phase3.log" Enter

tmux new-window -t "$SESSION" -n "gpu2"
tmux send-keys -t "${SESSION}:gpu2" \
  "bash $BASE/run_fidelity_phase3_gpu2.sh 2>&1 | tee $BASE/logs/fidelity_gpu2_phase3.log" Enter

tmux new-window -t "$SESSION" -n "gpu3"
tmux send-keys -t "${SESSION}:gpu3" \
  "bash $BASE/run_fidelity_phase3_gpu3.sh 2>&1 | tee $BASE/logs/fidelity_gpu3_phase3.log" Enter

tmux select-window -t "${SESSION}:gpu0"

echo ""
echo "Launched. Attach with:"
echo "  tmux attach -t $SESSION"
echo ""
echo "Switch windows: Ctrl-b 0/1/2/3"
echo "Logs:"
echo "  tail -f $BASE/logs/fidelity_gpu{0,1,2,3}_phase3.log"
