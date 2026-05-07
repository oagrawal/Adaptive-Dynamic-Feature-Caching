#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2

BASE=/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache
GT=$BASE/videos/easycache_baseline
OUTDIR=$BASE/fidelity_metrics
cd /nfs/oagrawal/CogVideo/dicache_results/metrics

for MODE in easycache_adaptive easycache_adaptive_0.025_0.075; do
  echo "[GPU2] Running $MODE"
  python eval_with_json.py \
    --gt_video_dir $GT \
    --generated_video_dir $BASE/videos/$MODE \
    --output_json $OUTDIR/${MODE}_vs_easycache_baseline.json \
    --mode_name $MODE
  echo "[GPU2] Done $MODE"
done
