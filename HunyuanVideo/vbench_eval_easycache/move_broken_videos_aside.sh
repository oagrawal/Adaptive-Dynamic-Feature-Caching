#!/bin/bash
# Move the 4 broken-mode (720x1280, noise-output) video directories aside before re-running at 544x960.
# batch_generate.py skips any video file that already exists; if these dirs aren't moved, the rerun is a no-op.

set -euo pipefail

VIDEOS=/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache/videos
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE=$VIDEOS/_broken_720p_$STAMP

mkdir -p "$ARCHIVE"

for MODE in \
  easycache_fixed_0.0375 \
  easycache_fixed_0.075 \
  easycache_adaptive_0.025_0.075 \
  easycache_adaptive_0.0375_0.050; do
  if [ -d "$VIDEOS/$MODE" ]; then
    mv "$VIDEOS/$MODE" "$ARCHIVE/$MODE"
    echo "Moved: $MODE -> $ARCHIVE/$MODE"
  else
    echo "Skipped (not present): $MODE"
  fi
done

# Also wipe any stale fidelity JSONs for these 4 modes so they don't get reused.
FID=/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache/fidelity_metrics
for f in \
  "$FID/easycache_fixed_0.0375_vs_easycache_baseline.json" \
  "$FID/easycache_fixed_0.075_vs_easycache_baseline.json" \
  "$FID/easycache_adaptive_0.025_0.075_vs_easycache_baseline.json" \
  "$FID/easycache_adaptive_0.0375_0.050_vs_easycache_baseline.json"; do
  if [ -f "$f" ]; then
    mv "$f" "$ARCHIVE/$(basename "$f")"
    echo "Moved stale fidelity: $(basename "$f")"
  fi
done

echo
echo "Done. Broken-mode artifacts archived under:"
echo "  $ARCHIVE"
