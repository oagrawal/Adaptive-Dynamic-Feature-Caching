#!/usr/bin/env python3
"""
Batch video generation for DiCache VBench evaluation — Wan2.1 T2V-1.3B.

Loads the Wan model ONCE then loops over prompts and DiCache modes.
Saves videos in VBench naming format: dicache_results/videos/{mode}/{prompt}-0.mp4

IMPORTANT — instance attribute fix (same pattern as EasyCache batch_generate_wan.py):
  DiCache state lives on class attributes after the first video, Python creates
  instance attributes that shadow the class attributes on reset. configure_dicache()
  sets all state on the MODEL INSTANCE (model.cnt = 0, ...) so resets are correct
  across runs. The forward method is still set on the class (nn.Module dispatch).

GPU split: use --start-idx / --end-idx with disjoint prompt ranges (no race conditions).
Resume: skips video files that already exist.
"""

import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

WAN_ROOT = str(Path(__file__).resolve().parent.parent)
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video

# ---------------------------------------------------------------------------
# Import dicache_forward from the working copy in dicache_exp/
# ---------------------------------------------------------------------------
_dicache_spec = importlib.util.spec_from_file_location(
    "run_wan_dicache",
    os.path.join(WAN_ROOT, "dicache_exp", "run_wan_dicache.py"),
)
_dicache_mod = importlib.util.module_from_spec(_dicache_spec)
_dicache_spec.loader.exec_module(_dicache_mod)
dicache_forward = _dicache_mod.dicache_forward

# ---------------------------------------------------------------------------
# Modes for the fixed-threshold sweep
# ---------------------------------------------------------------------------
MODES = [
    {"name": "wan_dc_baseline",   "type": "baseline"},
    {"name": "wan_dc_fixed_0.05", "type": "fixed", "thresh": 0.05},
    {"name": "wan_dc_fixed_0.10", "type": "fixed", "thresh": 0.10},
    {"name": "wan_dc_fixed_0.15", "type": "fixed", "thresh": 0.15},
    {"name": "wan_dc_fixed_0.20", "type": "fixed", "thresh": 0.20},
    {"name": "wan_dc_fixed_0.25", "type": "fixed", "thresh": 0.25},
    {"name": "wan_dc_fixed_0.30", "type": "fixed", "thresh": 0.30},
    {"name": "wan_dc_fixed_0.225",                "type": "fixed",    "thresh": 0.225},
    {"name": "wan_dc_adaptive_hi0.25_lo0.05",     "type": "adaptive", "thresh_high": 0.25,  "thresh_low": 0.05,  "stable_start": 16, "stable_end": 70},
    {"name": "wan_dc_adaptive_hi0.25_lo0.10",     "type": "adaptive", "thresh_high": 0.25,  "thresh_low": 0.10,  "stable_start": 16, "stable_end": 70},
    {"name": "wan_dc_adaptive_hi0.25_lo0.15",     "type": "adaptive", "thresh_high": 0.25,  "thresh_low": 0.15,  "stable_start": 16, "stable_end": 70},
    {"name": "wan_dc_adaptive_hi0.225_lo0.05",    "type": "adaptive", "thresh_high": 0.225, "thresh_low": 0.05,  "stable_start": 16, "stable_end": 70},
    {"name": "wan_dc_adaptive_hi0.225_lo0.10",    "type": "adaptive", "thresh_high": 0.225, "thresh_low": 0.10,  "stable_start": 16, "stable_end": 70},
    {"name": "wan_dc_adaptive_hi0.225_lo0.15",    "type": "adaptive", "thresh_high": 0.225, "thresh_low": 0.15,  "stable_start": 16, "stable_end": 70},
]

# ---------------------------------------------------------------------------
# Model configuration helpers
# ---------------------------------------------------------------------------
_original_forward = None  # set once after model load


def configure_dicache(model, mode_config, sample_steps=50):
    """
    Configure model for one DiCache mode.

    State is set on the MODEL INSTANCE so that class-attr shadowing after the
    first video's reset block does not cause stale values on subsequent videos.
    The forward method is set on the class (required for nn.Module dispatch).
    """
    model_cls = model.__class__

    if mode_config["type"] == "baseline":
        model_cls.forward = _original_forward
        return

    # Patch forward on the class
    model_cls.forward = dicache_forward

    # All per-run state on the INSTANCE
    model.cnt = 0
    model.probe_depth = 1
    model.num_steps = sample_steps * 2
    model.rel_l1_thresh = mode_config.get("thresh", mode_config.get("thresh_high", 0.10))
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.ret_ratio = 0.0
    model.residual_cache = [None, None]
    model.probe_residual_cache = [None, None]
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.previous_output = [None, None]
    model.resume_flag = [False, False]
    model.calibrate = False
    model.calibration_deltas = []
    model.adaptive = (mode_config.get("type") == "adaptive")
    model.thresh_high = mode_config.get("thresh_high", 0.20)
    model.thresh_low = mode_config.get("thresh_low", 0.05)
    model.stable_start = mode_config.get("stable_start", 16)
    model.stable_end = mode_config.get("stable_end", 70)


# ---------------------------------------------------------------------------
# Generation log helpers
# ---------------------------------------------------------------------------
def load_gen_log(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_gen_log(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Wan2.1 DiCache VBench batch generation")
    p.add_argument("--prompts-file", type=str,
                   default=os.path.join(WAN_ROOT, "vbench_eval", "prompts_subset.json"))
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(WAN_ROOT, "dicache_results", "videos"))
    p.add_argument("--ckpt-dir", type=str,
                   default="/nfs/oagrawal/wan/Wan2.1-T2V-1.3B")
    p.add_argument("--task", type=str, default="t2v-1.3B")
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    p.add_argument("--modes", type=str, default="all",
                   help="Comma-separated mode names or 'all'")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.prompts_file) as f:
        all_prompts = json.load(f)
    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    prompts = all_prompts[args.start_idx:end_idx]

    if args.modes == "all":
        modes = MODES
    else:
        names = {m.strip() for m in args.modes.split(",")}
        modes = [m for m in MODES if m["name"] in names]
        if not modes:
            print("ERROR: no matching modes. Available: " + str([m["name"] for m in MODES]))
            sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    seed = args.seed
    total = len(prompts) * len(modes)

    print("=" * 70)
    print("Wan2.1 DiCache VBench Batch Generation")
    print("=" * 70)
    print("Prompts : [%d, %d)  (%d prompts)" % (args.start_idx, end_idx, len(prompts)))
    print("Modes   : " + str([m["name"] for m in modes]))
    print("Total   : %d videos" % total)
    print("Output  : " + output_dir)
    print("=" * 70)

    if args.dry_run:
        for entry in prompts:
            prompt = entry["prompt_en"]
            for m in modes:
                path = os.path.join(output_dir, m["name"], "%s-%d.mp4" % (prompt, seed))
                print("  %-6s  %s/%s" % ("EXISTS" if os.path.exists(path) else "NEW", m["name"], prompt[:60]))
        return

    # Load model once
    print("\nLoading Wan2.1 model ...")
    cfg = WAN_CONFIGS[args.task]
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
    )
    print("Model loaded.\n")

    # Save original forward BEFORE any patching (for baseline restore)
    global _original_forward
    _original_forward = wan_t2v.model.__class__.forward

    log_path = os.path.join(output_dir, "generation_log_%d-%d.json" % (args.start_idx, end_idx))
    os.makedirs(output_dir, exist_ok=True)
    gen_log = load_gen_log(log_path)

    completed, skipped, failed = 0, 0, 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = args.start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, "%s-%d.mp4" % (prompt, seed))
            run_num = prompt_idx * len(modes) + mode_idx + 1

            if os.path.exists(video_path):
                print("[%3d/%d] SKIP  %s | %s..." % (run_num, total, mode_name, prompt[:55]))
                skipped += 1
                continue

            configure_dicache(wan_t2v.model, mode, sample_steps=args.sample_steps)

            print("[%3d/%d] GEN   %s | %s..." % (run_num, total, mode_name, prompt[:55]))
            os.makedirs(video_dir, exist_ok=True)

            try:
                t0 = time.time()
                video = wan_t2v.generate(
                    prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=81,
                    shift=5.0,
                    sample_solver="unipc",
                    sampling_steps=args.sample_steps,
                    guide_scale=5.0,
                    seed=seed,
                    offload_model=True,
                )
                gen_time = time.time() - t0

                cache_video(
                    tensor=video[None],
                    save_file=video_path,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

                dit_time = getattr(wan_t2v, "cost_time", None)
                gen_log["runs"].append({
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode_name,
                    "prompt_index": global_idx,
                    "time_seconds": round(gen_time, 2),
                    "dit_time_seconds": round(dit_time, 2) if dit_time else None,
                    "video_path": video_path,
                    "timestamp": datetime.now().isoformat(),
                })
                gen_log["completed_keys"].append("%s|%s|%d" % (mode_name, prompt, seed))
                save_gen_log(log_path, gen_log)

                completed += 1
                total_gen_time += gen_time
                dit_str = (", %ds DiT" % dit_time) if dit_time else ""
                print("         -> saved (%ds e2e%s)" % (gen_time, dit_str))

            except Exception as e:
                print("         -> FAILED: " + str(e))
                import traceback; traceback.print_exc()
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "prompt_index": global_idx,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                save_gen_log(log_path, gen_log)

    print("\n" + "=" * 70)
    print("Completed: %d   Skipped: %d   Failed: %d" % (completed, skipped, failed))
    if completed:
        print("Total time: %.1fh   Avg/video: %.0fs" % (total_gen_time / 3600, total_gen_time / completed))
    print("=" * 70)


if __name__ == "__main__":
    main()
