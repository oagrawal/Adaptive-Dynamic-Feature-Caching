#!/bin/bash
# Launch phase4 fidelity eval across 4 GPUs in a tmux session.
# Run directly on the host (no container needed).
#
# GPU 0: easycache_fixed_0.040
# GPU 1: easycache_fixed_0.045
# GPU 2: easycache_adaptive_0.030_0.060_f4l10
# GPU 3: easycache_adaptive_0.035_0.065_f4l10
#
# Usage: bash vbench_eval_easycache/launch_fidelity_phase4_tmux.sh

set -euo pipefail

SESSION="hv_ec_fidelity_phase4"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 1
fi

BASE=/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache
mkdir -p "$BASE/logs"

for i in 0 1 2 3; do
  LOG=$BASE/logs/fidelity_gpu${i}_phase4.log
  if [ -f "$LOG" ]; then
    echo "ERROR: $LOG already exists. Aborting to avoid overwrite." >&2
    exit 1
  fi
done

tmux new-session -d -s "$SESSION" -n "gpu0"
tmux send-keys -t "${SESSION}:gpu0" \
  "docker exec hv_eval_wan bash -c 'bash $BASE/run_fidelity_phase4_gpu0.sh 2>&1 | tee $BASE/logs/fidelity_gpu0_phase4.log'" Enter

tmux new-window -t "$SESSION" -n "gpu1"
tmux send-keys -t "${SESSION}:gpu1" \
  "docker exec hv_eval_wan bash -c 'bash $BASE/run_fidelity_phase4_gpu1.sh 2>&1 | tee $BASE/logs/fidelity_gpu1_phase4.log'" Enter

tmux new-window -t "$SESSION" -n "gpu2"
tmux send-keys -t "${SESSION}:gpu2" \
  "docker exec hv_eval_wan bash -c 'bash $BASE/run_fidelity_phase4_gpu2.sh 2>&1 | tee $BASE/logs/fidelity_gpu2_phase4.log'" Enter

tmux new-window -t "$SESSION" -n "gpu3"
tmux send-keys -t "${SESSION}:gpu3" \
  "docker exec hv_eval_wan bash -c 'bash $BASE/run_fidelity_phase4_gpu3.sh 2>&1 | tee $BASE/logs/fidelity_gpu3_phase4.log'" Enter

tmux select-window -t "${SESSION}:gpu0"

echo ""
echo "Launched. Attach with:"
echo "  tmux attach -t $SESSION"
echo ""
echo "Switch windows: Ctrl-b 0/1/2/3"
echo "Logs:"
echo "  tail -f $BASE/logs/fidelity_gpu{0,1,2,3}_phase4.log"
