#!/bin/bash
set -euo pipefail
export PYTHONPATH=/nfs/oagrawal/wan/Wan2.1
export CUDA_VISIBLE_DEVICES=2

cd /nfs/oagrawal/wan/Wan2.1

python3 dicache_results/batch_generate_wan_dicache.py   --start-idx 17   --end-idx 25   --modes wan_dc_fixed_0.07,wan_dc_fixed_0.08,wan_dc_adaptive_hi0.25_lo0.07,wan_dc_adaptive_hi0.225_lo0.07   2>&1 | tee dicache_results/logs/gpu2_new_modes.log
