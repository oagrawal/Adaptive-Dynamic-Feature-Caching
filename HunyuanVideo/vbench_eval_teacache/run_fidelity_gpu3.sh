#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3

BASE=/nfs/oagrawal/HunyuanVideo/vbench_eval_teacache
GT=$BASE/videos/hunyuan_baseline
OUTDIR=$BASE/fidelity_metrics
cd /nfs/oagrawal/CogVideo/dicache_results/metrics

MODE=hunyuan_tc_adaptive_lo0.1_hi0.25
echo "[GPU3] Running $MODE"
python3 eval_with_json.py \
  --gt_video_dir $GT \
  --generated_video_dir $BASE/videos/$MODE \
  --output_json $OUTDIR/${MODE}_vs_hunyuan_baseline.json \
  --mode_name $MODE
echo "[GPU3] Done $MODE"
