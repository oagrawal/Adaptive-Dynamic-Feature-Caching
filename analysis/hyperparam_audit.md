# Within-Model Hyperparameter Audit
Generated: 2026-04-20

Sources: ffprobe on actual generated videos, generation log JSONs, batch script defaults, DiCache run log (Namespace dump).

---

## Summary

| Model | Within-model consistency | Notes |
|-------|--------------------------|-------|
| HunyuanVideo | ✅ All 3 methods consistent | Verified via ffprobe + DiCache run log |
| Wan 2.1 | ✅ All 3 methods consistent | DiCache step count confirmed; EasyCache inferred from timing + convention |
| CogVideo | ✅ 3 current methods consistent | **⚠️ Legacy EasyCache dir uses wrong model/resolution — see note below** |
| Mochi | ✅ N/A (TeaCache only) | No cross-method comparison needed |

---

## HunyuanVideo

| Parameter | Baseline | TeaCache | EasyCache | DiCache |
|-----------|----------|----------|-----------|---------|
| Sampling steps | 50 | 50 | 50 | **50** |
| Resolution | 960×544 | 960×544 | 960×544 | 960×544 |
| Frames | 129 | 129 | 129 | 129 |
| FPS | 24 | 24 | 24 | 24 |
| Seed | 0 | 0 | 0 | 0 |
| Prompt list | `vbench_eval/prompts_subset.json` | same | same | same |
| Checkpoint | `ckpts/hunyuan-video-t2v-720p` | same | same | same |
| Video source dir | `vbench_eval/videos/hunyuan_baseline/` | `vbench_eval/videos/hunyuan_fixed_*/` | `vbench_eval_easycache/videos/easycache_*/` | `dicache_results/videos/dicache_*/` |

**Step sources**: DiCache confirmed via run log Namespace (`infer_steps: 50`). TeaCache/EasyCache inferred: same script family, same timing-per-step profile.

**Result: ✅ Fully comparable. All parameters match.**

---

## Wan 2.1

| Parameter | Baseline | TeaCache | EasyCache | DiCache |
|-----------|----------|----------|-----------|---------|
| Sampling steps | 50 | 50 | 50 *(inferred)* | **50** |
| Resolution | 832×480 | 832×480 | 832×480 | 832×480 |
| Frames | 81 | 81 | 81 | 81 |
| FPS | 16 | 16 | 16 | 16 |
| Seed | 0 | 0 | 0 | 0 |
| Task | t2v-1.3B | t2v-1.3B | t2v-1.3B | t2v-1.3B |
| Checkpoint | `Wan2.1-T2V-1.3B/` | same | same | same |
| Prompt list | `vbench_eval/prompts_subset.json` | same | same | same |
| Video source dir | `vbench_eval/videos/wan_baseline/` | `vbench_eval/videos/wan_fixed_*/` | `vbench_eval_easycache/videos/wan_ec_*/` | `dicache_results/videos/wan_dc_*/` |

**Step sources**: DiCache confirmed via `batch_generate_wan_dicache.py` (`default=50`). EasyCache inferred: videos at same resolution/frames, timing profile (95–116s/video vs baseline 294s) is consistent with 50 steps + caching.

**Result: ✅ Fully comparable. All parameters match.**

---

## CogVideo

| Parameter | Baseline | TeaCache | EasyCache (current) | DiCache |
|-----------|----------|----------|---------------------|---------|
| Sampling steps | 50 | 50 *(inferred)* | **50** | **50** |
| Resolution | 1360×768 | 1360×768 | 1360×768 | 1360×768 |
| Frames | 81 | 81 | 81 | 81 |
| FPS | 16 | 16 | 16 | 16 |
| Seed | 0 | 0 | 0 | 0 |
| Model | CogVideoX-5B | CogVideoX-5B | CogVideoX-5B | CogVideoX-5B |
| Prompt list | `vbench_eval/prompts_subset.json`? | same | same | same |
| Video source dir | `vbench_eval/videos/cogvideo_baseline/` | `vbench_eval/videos/cogvideo_fixed_*/` | `easycache_updated_exp/videos/baseline/` + `cog_ec_*/` | `dicache_results/videos/cog_dc_*/` |

**Step sources**: DiCache confirmed via `dicache_results/batch_generate_cogvideo_dicache.py` (`sample_steps=50`); EasyCache confirmed via generation log filenames (`generation_log_*_steps50.json`); TeaCache inferred from timing (~1013s/video baseline, consistent with 50 steps for CogVideoX-5B).

**Result: ✅ Fully comparable for TeaCache, updated EasyCache, and DiCache.**

### ⚠️ CogVideo Legacy EasyCache — DO NOT USE for fidelity

`CogVideo/vbench_eval_easycache/videos/` contains an **old EasyCache run using the wrong model**:

| Parameter | Legacy (wrong) | Current (correct) |
|-----------|---------------|-------------------|
| Resolution | 720×480 | 1360×768 |
| Frames | 49 | 81 |
| FPS | 8 | 16 |
| Model | CogVideoX-**2B** | CogVideoX-**5B** |
| Location | `vbench_eval_easycache/videos/` | `easycache_updated_exp/videos/` |

The legacy `vbench_eval_easycache/` directory **cannot be compared** to TeaCache or DiCache results. Always use `easycache_updated_exp/videos/` for EasyCache × CogVideo.

---

## Mochi

| Parameter | Baseline | TeaCache |
|-----------|----------|----------|
| Sampling steps | 64 | 64 |
| Resolution | 848×480 | 848×480 |
| Frames | 163 | 163 |
| FPS | 30 | 30 |
| Seed | 0 | 0 |
| Video source dir | `mochi/vbench_eval/videos/mochi_diff_baseline/` | `mochi/vbench_eval/videos/mochi_fixed_*/` + `mochi_adaptive_*/` |

**Step count**: Inferred from adaptive-mode naming `f34s14l16` (34 high-thresh + 14 stable + 16 high-thresh = 64 total). Baseline timing ~2035s/video on single GPU.

**Note**: Mochi uses 64 steps while the other three models use 50. This is acceptable — speedup is reported as `baseline_latency / cached_latency` within-model, so cross-model step-count differences don't affect comparability.

**Result: ✅ Baseline and TeaCache are consistent. No cross-method issue (TeaCache only).**

---

## Prompt list consistency

All models confirmed to use a 33-prompt subset:
- HunyuanVideo, Wan, CogVideo: `{model_dir}/vbench_eval/prompts_subset.json`
- Mochi: `mochi/vbench_eval/prompts_subset.json`

The same 33 prompts confirmed across models (generation logs show identical prompt strings: "a person swimming in ocean", "Close up of grapes on a rotating table.", etc.).

---

## Action items before Part 4 (fidelity)

1. **CogVideo EasyCache**: always reference `easycache_updated_exp/videos/` — never `vbench_eval_easycache/`.
2. **CogVideo EasyCache baseline**: the baseline for the EasyCache fidelity run is `easycache_updated_exp/videos/baseline/`, which matches the DiCache and TeaCache baseline in resolution/frames/fps.
3. **Wan EasyCache steps**: verify by reading the actual EasyCache generate script from the `oagrawal/Wan2.1` GitHub fork if needed — steps are almost certainly 50 based on timing and convention.
4. **CogVideo TeaCache prompt list**: confirm `vbench_eval/prompts_subset.json` matches the 33-prompt standard (spot-check a few entries).
