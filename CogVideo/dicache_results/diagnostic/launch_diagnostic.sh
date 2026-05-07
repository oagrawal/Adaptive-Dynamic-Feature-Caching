#!/usr/bin/env bash
# Launch one-prompt CogVideoX + DiCache diagnostics across GPUs 0-3.
#
# Run from the host:
#   bash /nfs/oagrawal/CogVideo/dicache_results/diagnostic/launch_diagnostic.sh

set -euo pipefail

HOST_DIAG=/nfs/oagrawal/CogVideo/dicache_results/diagnostic
if docker container inspect cogvideo >/dev/null 2>&1; then
  CONTAINER=cogvideo
  COG_ROOT=/workspace/cogvideo
else
  CONTAINER=hv_eval_wan
  COG_ROOT=/nfs/oagrawal/CogVideo
fi
DIAG=${COG_ROOT}/dicache_results/diagnostic

mkdir -p "${HOST_DIAG}/logs"

docker exec -d "${CONTAINER}" bash -lc "cd ${COG_ROOT} && mkdir -p ${DIAG}/logs && CUDA_VISIBLE_DEVICES=0 python3 ${DIAG}/cogvideo_dicache_diagnostic.py --prompt-idx 0 --modes baseline_probe 2>&1 | tee ${DIAG}/logs/gpu0_baseline_probe.log"

docker exec -d "${CONTAINER}" bash -lc "cd ${COG_ROOT} && mkdir -p ${DIAG}/logs && CUDA_VISIBLE_DEVICES=1 python3 ${DIAG}/cogvideo_dicache_diagnostic.py --prompt-idx 0 --modes fixed_0.20,fixed_0.30,fixed_0.40 2>&1 | tee ${DIAG}/logs/gpu1_fixed.log"

docker exec -d "${CONTAINER}" bash -lc "cd ${COG_ROOT} && mkdir -p ${DIAG}/logs && CUDA_VISIBLE_DEVICES=2 python3 ${DIAG}/cogvideo_dicache_diagnostic.py --prompt-idx 0 --modes adaptive_hi0.30_lo0.05_mid15_48 2>&1 | tee ${DIAG}/logs/gpu2_adaptive_hi0.30.log"

docker exec -d "${CONTAINER}" bash -lc "cd ${COG_ROOT} && mkdir -p ${DIAG}/logs && CUDA_VISIBLE_DEVICES=3 python3 ${DIAG}/cogvideo_dicache_diagnostic.py --prompt-idx 0 --modes adaptive_hi0.40_lo0.05_mid15_48,adaptive_hi0.40_lo0.05_mid15_48_force_last1 2>&1 | tee ${DIAG}/logs/gpu3_adaptive_hi0.40.log"

echo "Launched diagnostics in ${CONTAINER}. Logs: ${HOST_DIAG}/logs"
echo "After all logs show DONE, run:"
echo "  docker exec ${CONTAINER} bash -lc 'cd ${COG_ROOT} && python3 ${DIAG}/cogvideo_dicache_diagnostic.py --plot-only'"
