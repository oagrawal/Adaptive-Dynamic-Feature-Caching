# /nfs/oagrawal — project context

Framework paper on **dynamic feature caching methods** (TeaCache, EasyCache, DiCache) for open-source video-generation diffusion models (HunyuanVideo, Wan 2.1, CogVideo, Mochi).

## Supported (caching method × model) combos

| | Hunyuan | Wan 2.1 | CogVideo | Mochi |
|---|---|---|---|---|
| **TeaCache** | ✅ | ✅ | ✅ | ✅ |
| **EasyCache** | ✅ | ✅ | ✅ (self-impl) | — |
| **DiCache** | ✅ | ✅ | ✅ (self-impl) | — |

Mochi is TeaCache-only. CogVideo's EasyCache and DiCache implementations were written by the user and currently show **odd fidelity-metric behavior** — under investigation (see `analysis/cogvideo_selfimpl_notes.md` when it exists).

## Directory layout (top level of `/nfs/oagrawal`)

- `CogVideo/`, `HunyuanVideo/`, `wan/`, `mochi/` — per-model repos. Each contains that model's caching-method entry points, batch generation scripts, and eval harnesses (`vbench_eval/`, `dicache_results/`, etc.).
- `vbench_models/` — shared VBench model weights, referenced by all per-model eval pipelines. Do not move or rename.
- `analysis/` — cross-model analysis outputs (cross_model_comparison, hyperparam audit, fidelity summary, investigation notes).
- `other-stuff/` — unrelated legacy work (LLM/RL/robotics experiments). Ignore.
- `AGENT_TODO.md` — in-flight experiment tracking (currently Wan2.1 DiCache phases).
- `.cursor/` — editor config.

## Git

Single git repo at `/nfs/oagrawal/` covers the entire project tree. Nested per-model `.git` folders were removed; upstream remote URLs + last-synced commit hashes are preserved in `/nfs/oagrawal/UPSTREAMS.md` for manual re-sync if needed. No git LFS — large binaries (weights, videos, vbench_models/) are excluded via `/nfs/oagrawal/.gitignore`. `other-stuff/` is excluded from git entirely.

Remote: `git@github.com:oagrawal/Adaptive-Dynamic-Feature-Caching.git` (origin/main).

## Docker containers + mount paths (critical — do not break)

| Container | Mount | Purpose |
|---|---|---|
| `hv_eval_wan` | `-v /nfs/oagrawal:/nfs/oagrawal` | "Everything" container: generation + VBench + fidelity. Whole NFS tree is visible at same path inside. |
| `hunyuanvideo` | `$(pwd):/workspace` with `pwd = /nfs/oagrawal/HunyuanVideo` | Hunyuan-specific generation. |
| `cogvideo` | `/nfs/oagrawal/CogVideo:/workspace/cogvideo` | CogVideo-specific generation. |

Because these containers pin to specific paths, **never move the per-model top-level dirs** (`CogVideo`, `HunyuanVideo`, `wan`, `mochi`, `vbench_models`). Moving unrelated dirs under `/nfs/oagrawal/` is safe for all three containers.

Recreate docker run reference: `HunyuanVideo/vbench_eval/INSTRUCTIONS.md:419` and `CogVideo/INSTRUCTIONS_COGVIDEO_EASYCACHE.txt:27`.

## Comparability rule (do not violate)

**Within a single model**, baseline and all caching methods must share:
- sampling steps
- resolution, frame count, FPS
- seed
- prompt list
- model checkpoint/version

**Across models** these can differ naturally (e.g., Mochi uses 64 steps, Cog has fewer frames). Speedup is always reported as `baseline_latency_same_model / cached_latency_same_model`, so cross-model speedup comparison is still meaningful.

Hyperparameter audit per model: `analysis/hyperparam_audit.md` (once written).

## Standard eval setup

- **Prompts**: 33-prompt subset at `{model}/vbench_eval/prompts_subset.json` (same 33 prompts across all models).
- **Seed**: `0`.
- **Baselines known good**:
  - Hunyuan: 50 steps, 544×960, 129 frames, 24 fps
  - Wan 2.1: 50 steps, 832×480, 81 frames, 16 fps, checkpoint `Wan2.1-T2V-1.3B`
  - CogVideo: 50 steps (verify frames/resolution per-script)
  - Mochi: 64 steps (TeaCache only; within-model-only comparability)
- **VBench transformers pin**: 4.33.2 for VBench eval, restore 4.46.3 for generation.

## Primary quality metric

**Fidelity** (PSNR / SSIM / LPIPS) is now the primary Pareto-frontier indicator; VBench is secondary. Reuse these scripts:

- `HunyuanVideo/vbench_eval/run_fidelity_metrics.py` — paired prompt+seed, per-mode-vs-baseline JSON. PSNR (MSE-based, capped 100), SSIM (Gaussian 11×11), LPIPS (AlexNet).
- `CogVideo/dicache_results/metrics/eval_with_json.py` — docker-wrapped variant.

Per-combo results at `{model}/*/fidelity_metrics/{mode}_vs_baseline.json`. Aggregated cross-combo table at `analysis/fidelity_summary.csv` (columns: `model, caching_method, mode, psnr_mean, ssim_mean, lpips_mean, speedup`).

## Sanity checks for any fidelity run

- PSNR monotonically decreases as caching threshold increases (more aggressive caching → lower fidelity).
- LPIPS monotonically increases.
- Baseline-vs-baseline canary: PSNR = 100, SSIM = 1.0, LPIPS = 0.
- If monotonicity breaks on a mature combo (Hunyuan or Wan), suspect a codec/pairing bug before believing the result.

## Environment notes

- Shared Linux host, **no sudo**. Dependency isolation is via docker.
- `/nfs/oagrawal` is an NFS mount accessed from multiple machines. Be conscious of file ownership; do not `chown -R` broadly.
- Generation of all 33 prompts for a single (model, caching mode) typically takes several hours to a day — **reuse existing videos** whenever possible; regenerate only when hyperparams genuinely disagree.
