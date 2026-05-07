#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

BASE=/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache
GT=$BASE/videos/easycache_baseline
OUTDIR=$BASE/fidelity_metrics
mkdir -p "$OUTDIR"
cd /nfs/oagrawal/CogVideo/dicache_results/metrics

MODE=easycache_fixed_0.045
echo "[GPU1] Running $MODE"
python eval_with_json.py \
  --gt_video_dir $GT \
  --generated_video_dir $BASE/videos/$MODE \
  --output_json $OUTDIR/${MODE}_vs_easycache_baseline.json \
  --mode_name $MODE
echo "[GPU1] Done $MODE"
