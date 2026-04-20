# /nfs/oagrawal — project context

Framework paper on **dynamic feature caching methods** (TeaCache, EasyCache, DiCache) for open-source video-generation diffusion models (HunyuanVideo, Wan 2.1, CogVideo, Mochi).

## Supported (caching method × model) combos

| | Hunyuan | Wan 2.1 | CogVideo | Mochi |
|---|---|---|---|---|
| **TeaCache** | ✅ | ✅ | ✅ | ✅ |
| **EasyCache** | ✅ | ✅ | ✅ (self-impl) | — |
| **DiCache** | ✅ | ✅ | ✅ (self-impl) | — |

Mochi is TeaCache-only. CogVideo's EasyCache and DiCache were self-implemented and may have bugs — see `analysis/cogvideo_selfimpl_notes.md` when it exists.

⚠️ **CogVideo EasyCache legacy dir**: `CogVideo/vbench_eval_easycache/` used the wrong model (2B, 720×480, 49 frames). **Never use it.** Correct EasyCache videos are in `CogVideo/easycache_updated_exp/` (5B, 1360×768, 81 frames, 16fps).

## Directory layout (top level of `/nfs/oagrawal`)

- `CogVideo/`, `HunyuanVideo/`, `wan/`, `mochi/` — per-model repos with caching impls + eval harnesses.
- `vbench_models/` — shared VBench weights. **Do not move or rename.**
- `analysis/` — cross-model outputs: `hyperparam_audit.md`, `fidelity_summary.csv` (once built), `pareto_index.md` (once built), `cross_model_comparison/`, `cogvideo_selfimpl_notes.md` (when written).
- `other-stuff/` — unrelated college projects. Ignore.

## Git

Single repo at `/nfs/oagrawal`, remote: `git@github.com:oagrawal/Adaptive-Dynamic-Feature-Caching.git` (origin/main). No git LFS — weights/videos excluded via `.gitignore`. `other-stuff/` fully excluded.

Nested per-model `.git` removed. Upstream refs preserved in `UPSTREAMS.md`.

**Known gotcha**: `HunyuanVideo/.gitignore` previously excluded `vbench_eval/videos/` as a directory (this would block generation log JSONs). Fixed 2026-04-20 to exclude only `*.mp4` etc.

## Docker containers + mount paths (critical — do not break)

| Container | Mount | Purpose |
|---|---|---|
| `hv_eval_wan` | `-v /nfs/oagrawal:/nfs/oagrawal` | Everything: generation + VBench + fidelity. Whole tree visible at same path. |
| `hunyuanvideo` | `pwd=/nfs/oagrawal/HunyuanVideo → /workspace` | HunyuanVideo generation. |
| `cogvideo` | `/nfs/oagrawal/CogVideo → /workspace/cogvideo` | CogVideo generation. |
| `mochi` | `/nfs/oagrawal/mochi → /workspace/mochi` | Mochi generation. |

**Never move** `CogVideo`, `HunyuanVideo`, `wan`, `mochi`, `vbench_models`. Recreate refs: `HunyuanVideo/vbench_eval/INSTRUCTIONS.md:419` and `CogVideo/INSTRUCTIONS_COGVIDEO_EASYCACHE.txt:27`.

## Comparability rule (do not violate)

**Within a model**: baseline + all caching methods must share steps, resolution, frames, fps, seed, prompt list, checkpoint. Cross-model differences (Mochi@64 steps, Cog frames) are fine — speedup is always `baseline_latency_same_model / cached_latency_same_model`.

Details: `analysis/hyperparam_audit.md` (verified 2026-04-20, all trusted combos consistent).

## Standard eval setup

- **Prompts**: 33-prompt subset at `{model}/vbench_eval/prompts_subset.json`. Seed 0.
- **Baselines**: Hunyuan 50 steps 544×960 129f 24fps · Wan 50 steps 832×480 81f 16fps (T2V-1.3B) · CogVideo 50 steps 1360×768 81f 16fps (5B) · Mochi 64 steps 848×480 163f 30fps
- **VBench transformers**: 4.33.2 for eval, restore 4.46.3 for generation.

## Fidelity state (as of 2026-04-20)

Primary metric is PSNR/SSIM/LPIPS (Pareto curves). Script: `CogVideo/dicache_results/metrics/eval_with_json.py`. Output: `{model}/{eval_dir}/fidelity_metrics/{mode}_vs_baseline.json`.

| Combo | Video modes | Fidelity JSONs | Pareto PNGs |
|---|---|---|---|
| TeaCache × HV | 4 | 0 — **needs compute** | 0 |
| TeaCache × Wan | 4 | 0 — **needs compute** | 0 |
| TeaCache × CogVideo | 6 | 6 ✓ | 0 — **needs plot** |
| TeaCache × Mochi | 5 | 0 — **needs compute** | 0 |
| EasyCache × HV | 8 | 0 — **needs compute** | 0 |
| EasyCache × Wan | 22 | 21 ✓ (1 partial mode) | 0 — **needs plot** |
| EasyCache × CogVideo (updated_exp) | 16 | 0 — deferred (suspect impl) | 1 |
| DiCache × HV | 13 | 12 ✓ | 0 — **needs plot** |
| DiCache × Wan | 14 | 13 ✓ | 7 ✓ |
| DiCache × CogVideo | 25 | 23 ✓ | 4 ✓ (suspect frontier) |
| DiCache + EasyCache × Mochi | — | — | — (Mochi = TC only) |

## Latency / speedup data (needed for Pareto plots)

All combos have generation log JSONs with `time_seconds` per prompt except:
- **HV EasyCache**: text logs at `HunyuanVideo/vbench_eval_easycache/logs/ec_gpu*.log` — extract timing from `[N/33] Generating:` timestamp deltas.
- **HV DiCache**: text logs at `HunyuanVideo/dicache_results/logs/*.log` — each log has `time: <seconds>` per video; also see `dicache_results/results/` CSV.

Generation log JSON paths: `{model}/{eval_dir}/videos/generation_log*.json` (HV/Wan/Mochi/Cog TeaCache) or `{model}/{eval_dir}/generation_log_gpu*.json` (Cog DiCache/EasyCache).

## Sanity checks for fidelity

- PSNR ↓ monotonically as threshold ↑; LPIPS ↑ monotonically.
- Baseline-vs-baseline: PSNR=100, SSIM=1.0, LPIPS=0.
- Monotonicity violation on HV/Wan = codec/pairing bug, not real result.

## Environment notes

- Shared Linux host, **no sudo**. Use docker for dependency isolation.
- NFS mount — file ownership matters; do not `chown -R` broadly.
- Video generation = hours to a day per 33-prompt run. **Reuse existing videos** unless hyperparams genuinely differ.
