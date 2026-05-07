#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

BASE=/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache
GT=$BASE/videos/easycache_baseline
OUTDIR=$BASE/fidelity_metrics
cd /nfs/oagrawal/CogVideo/dicache_results/metrics

for MODE in easycache_fixed_0.060 easycache_adaptive_0.025_0.075_f12l10; do
  echo "[GPU0] Running $MODE"
  python eval_with_json.py \
    --gt_video_dir $GT \
    --generated_video_dir $BASE/videos/$MODE \
    --output_json $OUTDIR/${MODE}_vs_easycache_baseline.json \
    --mode_name $MODE
  echo "[GPU0] Done $MODE"
done
