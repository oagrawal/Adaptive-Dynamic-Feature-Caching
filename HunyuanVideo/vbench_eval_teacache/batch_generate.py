#!/usr/bin/env python3
"""
Batch video generation for TeaCache VBench evaluation — HunyuanVideo.

Loads HunyuanVideo ONCE then loops over prompts × modes.
Output: vbench_eval_teacache/videos/{mode}/{prompt}-0.mp4

TeaCache state is set on the INSTANCE per video to ensure clean resets.
The forward method is set on the CLASS (required for nn.Module dispatch).

Usage (inside hunyuanvideo container at /workspace):
    python vbench_eval_teacache/batch_generate.py \
        --video-size 544 960 --video-length 129 --infer-steps 50 \
        --flow-reverse --use-cpu-offload \
        --start-idx 0 --end-idx 17

    python vbench_eval_teacache/batch_generate.py \
        --video-size 544 960 --video-length 129 --infer-steps 50 \
        --flow-reverse --use-cpu-offload \
        --start-idx 17 --end-idx 33

    # Specific modes only:
    python vbench_eval_teacache/batch_generate.py \
        --video-size 544 960 --video-length 129 --infer-steps 50 \
        --flow-reverse --use-cpu-offload \
        --modes hunyuan_fixed_0.05,hunyuan_fixed_0.3 \
        --start-idx 0 --end-idx 33
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict

import numpy as np
import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TeaCache forward — adapted from teacache_sample_video.py.
# Supports fixed threshold and adaptive (lower thresh at boundary steps).
# ---------------------------------------------------------------------------

def teacache_forward(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    text_states: torch.Tensor = None,
    text_mask: torch.Tensor = None,
    text_states_2: Optional[torch.Tensor] = None,
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
    guidance: torch.Tensor = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    from hyvideo.modules.modulate_layers import modulate
    from hyvideo.modules.attenion import get_cu_seqlens

    out = {}
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // self.patch_size[0],
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )

    vec = self.time_in(t)
    vec = vec + self.vector_in(text_states_2)
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q
    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    if self.enable_teacache:
        inp = img.clone()
        vec_ = vec.clone()
        (
            img_mod1_shift, img_mod1_scale, img_mod1_gate,
            img_mod2_shift, img_mod2_scale, img_mod2_gate,
        ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
        normed_inp = self.double_blocks[0].img_norm1(inp)
        modulated_inp = modulate(normed_inp, shift=img_mod1_shift, scale=img_mod1_scale)

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            # Adaptive: low thresh (conservative) at boundary steps to protect
            # early/late denoising; high thresh (aggressive) in the stable middle.
            # Fixed: single thresh throughout.
            if getattr(self, "adaptive", False):
                if self.cnt < self.boundary_steps_start or self.cnt >= (self.num_steps - self.boundary_steps_end):
                    effective_thresh = self.thresh_low
                else:
                    effective_thresh = self.thresh_high
            else:
                effective_thresh = self.rel_l1_thresh

            coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01,
                            -3.14987800e+00, 9.61237896e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.previous_modulated_input).abs().mean()
                 / self.previous_modulated_input.abs().mean()).cpu().item()
            )
            if self.accumulated_rel_l1_distance < effective_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    if self.enable_teacache:
        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            for _, block in enumerate(self.double_blocks):
                img, txt = block(img, txt, vec, cu_seqlens_q, cu_seqlens_kv,
                                 max_seqlen_q, max_seqlen_kv, freqs_cis)
            x = torch.cat((img, txt), 1)
            for _, block in enumerate(self.single_blocks):
                x = block(x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv,
                          max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin))
            img = x[:, :img_seq_len, ...]
            self.previous_residual = img - ori_img
    else:
        for _, block in enumerate(self.double_blocks):
            img, txt = block(img, txt, vec, cu_seqlens_q, cu_seqlens_kv,
                             max_seqlen_q, max_seqlen_kv, freqs_cis)
        x = torch.cat((img, txt), 1)
        for _, block in enumerate(self.single_blocks):
            x = block(x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv,
                      max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin))
        img = x[:, :img_seq_len, ...]

    img = self.final_layer(img, vec)
    img = self.unpatchify(img, tt, th, tw)
    if return_dict:
        out["x"] = img
        return out
    return img


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

# boundary_steps_start: first N steps use thresh_low (conservative).
# boundary_steps_end:   last N steps use thresh_low (conservative).
# Steps in between use thresh_high (aggressive).
_BOUNDARY_STEPS_START = 10
_BOUNDARY_STEPS_END   = 10

MODES = [
    # ---- existing modes — names must match existing video dirs for resume ----
    {"name": "hunyuan_baseline",  "type": "baseline"},
    {"name": "hunyuan_fixed_0.1", "type": "fixed", "thresh": 0.1},
    {"name": "hunyuan_fixed_0.2", "type": "fixed", "thresh": 0.2},
    # Adaptive: thresh_low at first/last 10 steps, thresh_high in stable middle.
    {"name": "hunyuan_adaptive",
     "type": "adaptive", "thresh_low": 0.1, "thresh_high": 0.2,
     "boundary_steps_start": _BOUNDARY_STEPS_START, "boundary_steps_end": _BOUNDARY_STEPS_END},

    # ---- new modes ----
    {"name": "hunyuan_fixed_0.05", "type": "fixed", "thresh": 0.05},
    {"name": "hunyuan_fixed_0.3",  "type": "fixed", "thresh": 0.3},
    {"name": "hunyuan_tc_adaptive_lo0.1_hi0.3",
     "type": "adaptive", "thresh_low": 0.1, "thresh_high": 0.3,
     "boundary_steps_start": _BOUNDARY_STEPS_START, "boundary_steps_end": _BOUNDARY_STEPS_END},
    {"name": "hunyuan_tc_adaptive_lo0.15_hi0.3",
     "type": "adaptive", "thresh_low": 0.15, "thresh_high": 0.3,
     "boundary_steps_start": _BOUNDARY_STEPS_START, "boundary_steps_end": _BOUNDARY_STEPS_END},
    {"name": "hunyuan_tc_adaptive_lo0.2_hi0.3",
     "type": "adaptive", "thresh_low": 0.2, "thresh_high": 0.3,
     "boundary_steps_start": _BOUNDARY_STEPS_START, "boundary_steps_end": _BOUNDARY_STEPS_END},
    {"name": "hunyuan_tc_adaptive_lo0.1_hi0.25",
     "type": "adaptive", "thresh_low": 0.1, "thresh_high": 0.25,
     "boundary_steps_start": _BOUNDARY_STEPS_START, "boundary_steps_end": _BOUNDARY_STEPS_END},
]


# ---------------------------------------------------------------------------
# Configure / reset helpers
# ---------------------------------------------------------------------------

def configure_teacache(transformer, mode_cfg, infer_steps):
    """Patch the transformer for one mode. Call before each generate()."""
    transformer.__class__.forward = teacache_forward

    # Instance-level state — must be on the instance so it shadows any stale
    # class-level attrs and resets properly between runs.
    transformer.cnt = 0
    transformer.num_steps = infer_steps
    transformer.accumulated_rel_l1_distance = 0
    transformer.previous_modulated_input = None
    transformer.previous_residual = None

    if mode_cfg["type"] == "baseline":
        transformer.enable_teacache = False
        transformer.adaptive = False
        transformer.rel_l1_thresh = None
        transformer.thresh_low = None
        transformer.thresh_high = None
        transformer.boundary_steps_start = None
        transformer.boundary_steps_end   = None
    elif mode_cfg["type"] == "fixed":
        transformer.enable_teacache = True
        transformer.adaptive = False
        transformer.rel_l1_thresh = mode_cfg["thresh"]
        transformer.thresh_low = None
        transformer.thresh_high = None
        transformer.boundary_steps_start = None
        transformer.boundary_steps_end   = None
    elif mode_cfg["type"] == "adaptive":
        transformer.enable_teacache = True
        transformer.adaptive = True
        transformer.rel_l1_thresh = mode_cfg["thresh_low"]  # fallback
        transformer.thresh_low = mode_cfg["thresh_low"]
        transformer.thresh_high = mode_cfg["thresh_high"]
        transformer.boundary_steps_start = mode_cfg["boundary_steps_start"]
        transformer.boundary_steps_end   = mode_cfg["boundary_steps_end"]


# ---------------------------------------------------------------------------
# Generation log helpers
# ---------------------------------------------------------------------------

def load_generation_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, log_data):
    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp_path, log_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument("--prompts-file", type=str,
                              default="vbench_eval_teacache/prompts_subset.json")
    batch_parser.add_argument("--output-dir", type=str,
                              default="vbench_eval_teacache/videos")
    batch_parser.add_argument("--generation-seed", type=int, default=0)
    batch_parser.add_argument("--start-idx", type=int, default=0)
    batch_parser.add_argument("--end-idx", type=int, default=-1)
    batch_parser.add_argument("--modes", type=str, default="all",
                              help="Comma-separated mode names or 'all'")
    batch_parser.add_argument("--dry-run", action="store_true")
    batch_args, remaining_argv = batch_parser.parse_known_args()

    with open(batch_args.prompts_file, "r") as f:
        all_prompts = json.load(f)
    end_idx = len(all_prompts) if batch_args.end_idx == -1 else batch_args.end_idx
    prompts = all_prompts[batch_args.start_idx:end_idx]

    if batch_args.modes == "all":
        modes = MODES
    else:
        names = [m.strip() for m in batch_args.modes.split(",")]
        modes = [m for m in MODES if m["name"] in names]
        if not modes:
            print(f"ERROR: no matching modes. Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed = batch_args.generation_seed
    output_dir = batch_args.output_dir
    start_idx = batch_args.start_idx
    total_videos = len(prompts) * len(modes)

    print("=" * 70)
    print("TeaCache VBench Batch Generation — HunyuanVideo")
    print("=" * 70)
    print(f"Prompts : [{start_idx}, {end_idx})  ({len(prompts)} prompts)")
    print(f"Modes   : {[m['name'] for m in modes]}")
    print(f"Total   : {total_videos} videos")
    print(f"Output  : {output_dir}")
    print("=" * 70)

    if batch_args.dry_run:
        for entry in prompts:
            prompt = entry["prompt_en"]
            for m in modes:
                p = os.path.join(output_dir, m["name"], f"{prompt}-{seed}.mp4")
                tag = "EXISTS" if os.path.exists(p) else "NEW   "
                print(f"  {tag}  {m['name']} | {prompt[:60]}")
        return

    # Heavy imports deferred until after dry-run check
    if "--save-path" not in remaining_argv:
        remaining_argv += ["--save-path", output_dir]
    sys.argv = [sys.argv[0]] + remaining_argv

    import hyvideo_pyc_loader  # noqa: F401 — installs finder for .pyc-only hyvideo
    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    args = parse_args()

    log_filename = f"generation_log_{start_idx}-{end_idx}.json"
    log_path = os.path.join(output_dir, log_filename)
    gen_log = load_generation_log(log_path)

    print("\nLoading HunyuanVideo...")
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model path not found: {models_root_path}")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args
    print("Model loaded.\n")

    transformer = hunyuan_video_sampler.pipeline.transformer
    completed, skipped, failed = 0, 0, 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, f"{prompt}-{seed}.mp4")
            run_num = prompt_idx * len(modes) + mode_idx + 1

            if os.path.exists(video_path):
                logger.info(f"[{run_num:3d}/{total_videos}] SKIP  {mode_name} | {prompt[:55]}...")
                skipped += 1
                continue

            configure_teacache(transformer, mode, args.infer_steps)
            logger.info(f"[{run_num:3d}/{total_videos}] GEN   {mode_name} | {prompt[:55]}...")

            try:
                t0 = time.time()
                outputs = hunyuan_video_sampler.predict(
                    prompt=prompt,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=1,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale,
                )
                gen_time = time.time() - t0

                os.makedirs(video_dir, exist_ok=True)
                save_videos_grid(outputs["samples"][0].unsqueeze(0), video_path, fps=24)

                run_key = f"{mode_name}|{prompt}|{seed}"
                gen_log["runs"].append({
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode_name,
                    "time_seconds": round(gen_time, 2),
                    "video_path": video_path,
                    "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                })
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)

                completed += 1
                total_gen_time += gen_time
                logger.info(f"         -> saved ({gen_time:.0f}s)")

            except Exception as e:
                import traceback
                logger.error(f"         -> FAILED: {e}")
                traceback.print_exc()
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode_name,
                    "prompt_index": global_idx,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                save_generation_log(log_path, gen_log)

    print("\n" + "=" * 70)
    print("TeaCache batch generation complete")
    print(f"Completed: {completed}   Skipped: {skipped}   Failed: {failed}")
    if completed:
        print(f"Total time: {total_gen_time/3600:.1f}h   Avg/video: {total_gen_time/completed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
