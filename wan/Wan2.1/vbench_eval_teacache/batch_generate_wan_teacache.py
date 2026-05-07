#!/usr/bin/env python3
"""
Batch video generation for TeaCache VBench evaluation — Wan2.1 T2V-1.3B.

Loads the Wan model ONCE then loops over prompts × modes.
Output: vbench_eval_teacache/videos/{mode}/{prompt}-0.mp4

Loop order mirrors DiCache: prompts outer, modes inner — so all modes for
each prompt are generated before moving to the next prompt.

TeaCache state is patched on the MODEL INSTANCE per video to ensure clean
resets between runs. The forward method is set on the class (required for
nn.Module dispatch).

Usage:
    python batch_generate_wan_teacache.py --start-idx 0  --end-idx 17
    python batch_generate_wan_teacache.py --start-idx 17 --end-idx 33

    # Run a subset of modes (comma-separated):
    python batch_generate_wan_teacache.py --start-idx 0 --end-idx 17 \
        --modes wan_tc_fixed_0.15,wan_tc_adaptive_hi0.25_lo0.05
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

warnings.filterwarnings("ignore")

WAN_ROOT = str(Path(__file__).resolve().parent.parent)
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video

# ---------------------------------------------------------------------------
# TeaCache forward — supports both fixed and adaptive threshold modes.
#
# Adaptive: thresh_low is used when cnt < stable_start or cnt >= stable_end
#           (edges of denoising — first/last N steps), thresh_high in middle.
# Fixed:    teacache_thresh used throughout.
# ---------------------------------------------------------------------------

def teacache_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
        freqs=self.freqs, context=context, context_lens=context_lens,
    )

    if self.enable_teacache:
        # Determine effective threshold for this step
        if self.adaptive:
            if self.stable_start <= self.cnt < self.stable_end:
                effective_thresh = self.thresh_high
            else:
                effective_thresh = self.thresh_low
        else:
            effective_thresh = self.teacache_thresh

        modulated_inp = e0 if self.use_ref_steps else e

        if self.cnt % 2 == 0:  # even → conditional
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.previous_e0_even).abs().mean()
                     / self.previous_e0_even.abs().mean()).cpu().item()
                )
                if self.accumulated_rel_l1_distance_even < effective_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()
        else:  # odd → unconditional
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(
                    ((modulated_inp - self.previous_e0_odd).abs().mean()
                     / self.previous_e0_odd.abs().mean()).cpu().item()
                )
                if self.accumulated_rel_l1_distance_odd < effective_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

    if self.enable_teacache:
        if self.is_even:
            if not should_calc_even:
                x += self.previous_residual_even
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_even = x - ori_x
        else:
            if not should_calc_odd:
                x += self.previous_residual_odd
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                self.previous_residual_odd = x - ori_x
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    return [u.float() for u in x]


# Replacement generate — from teacache_generate.py (t2v_generate)
def t2v_generate(self, input_prompt, size=(1280, 720), frame_num=81, shift=5.0,
                 sample_solver="unipc", sampling_steps=50, guide_scale=5.0,
                 n_prompt="", seed=-1, offload_model=True):
    F = frame_num
    target_shape = (
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        size[1] // self.vae_stride[1],
        size[0] // self.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (self.patch_size[1] * self.patch_size[2])
        * target_shape[1] / self.sp_size
    ) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [torch.randn(
        target_shape[0], target_shape[1], target_shape[2], target_shape[3],
        dtype=torch.float32, device=self.device, generator=seed_g,
    )]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device,
                                              sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        self.model.to(self.device)
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = torch.stack([t])
            noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents, sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()

    return videos[0] if self.rank == 0 else None


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

# Polynomial coefficients for Wan2.1 T2V-1.3B (from teacache_generate.py).
# Maps raw rel-L1 distance → rescaled accumulated distance used for skip decisions.
_COEFF = [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01]

SAMPLE_STEPS = 50

# Adaptive boundary: thresh_low for first 10 denoising steps (cnt 0–19) and
# last 10 denoising steps (cnt 80–99); thresh_high for middle (cnt 20–79).
# Each denoising step = 2 CFG calls (cond + uncond), so 10 steps = 20 cnt units.
_STABLE_START = 20
_STABLE_END   = 80

MODES = [
    # Baseline — no TeaCache, plain generation
    {"name": "wan_baseline", "type": "baseline"},

    # Fixed threshold — single thresh throughout
    {"name": "wan_tc_fixed_0.05", "type": "fixed", "thresh": 0.05},
    {"name": "wan_fixed_0.1",  "type": "fixed", "thresh": 0.10},
    {"name": "wan_fixed_0.2",  "type": "fixed", "thresh": 0.20},
    {"name": "wan_tc_fixed_0.15", "type": "fixed", "thresh": 0.15},
    {"name": "wan_tc_fixed_0.25", "type": "fixed", "thresh": 0.25},

    # Adaptive threshold — thresh_low at first/last 10 steps, thresh_high in middle
    {"name": "wan_adaptive",
     "type": "adaptive", "thresh_high": 0.20, "thresh_low": 0.10,
     "stable_start": _STABLE_START, "stable_end": _STABLE_END},
    {"name": "wan_tc_adaptive_hi0.25_lo0.05",
     "type": "adaptive", "thresh_high": 0.25, "thresh_low": 0.05,
     "stable_start": _STABLE_START, "stable_end": _STABLE_END},
    {"name": "wan_tc_adaptive_hi0.2_lo0.05",
     "type": "adaptive", "thresh_high": 0.20, "thresh_low": 0.05,
     "stable_start": _STABLE_START, "stable_end": _STABLE_END},
]

# ---------------------------------------------------------------------------
# Model configuration helpers
# ---------------------------------------------------------------------------

_original_forward = None


def configure_mode(model, mode_cfg, sample_steps=SAMPLE_STEPS):
    """Configure model for the given mode, restoring original forward for baseline."""
    if mode_cfg["type"] == "baseline":
        model.__class__.forward = _original_forward
        model.enable_teacache = False
    else:
        configure_teacache(model, mode_cfg, sample_steps)


def configure_teacache(model, mode_cfg, sample_steps=SAMPLE_STEPS):
    model_cls = model.__class__
    model_cls.forward = teacache_forward

    model.enable_teacache = True
    model.use_ref_steps   = False
    model.coefficients    = _COEFF
    model.ret_steps       = 1 * 2          # first 1 CFG pair always computed
    model.cutoff_steps    = sample_steps * 2 - 2  # last 1 CFG pair always computed
    model.num_steps       = sample_steps * 2
    model.adaptive        = (mode_cfg["type"] == "adaptive")

    if model.adaptive:
        model.thresh_high   = mode_cfg["thresh_high"]
        model.thresh_low    = mode_cfg["thresh_low"]
        model.stable_start  = mode_cfg["stable_start"]
        model.stable_end    = mode_cfg["stable_end"]
        model.teacache_thresh = None   # unused in adaptive mode
    else:
        model.teacache_thresh = mode_cfg["thresh"]
        model.thresh_high   = None
        model.thresh_low    = None
        model.stable_start  = None
        model.stable_end    = None

    # Per-video accumulator state (also reset here as the initial configure)
    _reset_state(model)


def _reset_state(model):
    model.cnt = 0
    model.is_even = True
    model.accumulated_rel_l1_distance_even = 0
    model.accumulated_rel_l1_distance_odd  = 0
    model.previous_e0_even       = None
    model.previous_e0_odd        = None
    model.previous_residual_even = None
    model.previous_residual_odd  = None


# ---------------------------------------------------------------------------
# Generation log helpers (atomic write, matching DiCache format)
# ---------------------------------------------------------------------------

def load_log(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {"runs": [], "completed_keys": []}


def save_log(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-idx", type=int, default=0,
                   help="First prompt index (inclusive)")
    p.add_argument("--end-idx", type=int, default=33,
                   help="Last prompt index (exclusive)")
    p.add_argument("--modes", type=str, default="all",
                   help="Comma-separated mode names, or 'all'")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be generated without running")
    return p.parse_args()


def main():
    global _original_forward
    args = parse_args()

    HERE        = Path(__file__).resolve().parent
    ckpt_dir    = str(HERE.parent.parent / "Wan2.1-T2V-1.3B")
    videos_root = HERE / "videos"
    log_path    = videos_root / f"generation_log_{args.start_idx}-{args.end_idx}.json"

    prompts_all = json.loads((HERE / "prompts_subset.json").read_text())
    prompts_slice = prompts_all[args.start_idx:args.end_idx]

    if args.modes == "all":
        active_modes = MODES
    else:
        names = {m.strip() for m in args.modes.split(",")}
        active_modes = [m for m in MODES if m["name"] in names]
        if not active_modes:
            print("ERROR: no matching modes. Available:", [m["name"] for m in MODES])
            sys.exit(1)

    total = len(prompts_slice) * len(active_modes)
    print("=" * 70)
    print("Wan2.1 TeaCache VBench Batch Generation")
    print("=" * 70)
    print(f"Prompts : [{args.start_idx}, {args.end_idx})  ({len(prompts_slice)} prompts)")
    print(f"Modes   : {[m['name'] for m in active_modes]}")
    print(f"Total   : {total} videos")
    print(f"Output  : {videos_root}")
    print("=" * 70)

    if args.dry_run:
        for entry in prompts_slice:
            prompt = entry["prompt_en"] if isinstance(entry, dict) else entry
            for m in active_modes:
                vpath = videos_root / m["name"] / f"{prompt[:80].replace('/', '_')}-0.mp4"
                tag = "EXISTS" if vpath.exists() else "NEW   "
                print(f"  {tag}  {m['name']} | {prompt[:55]}")
        return

    print("\nLoading Wan T2V-1.3B ...")
    cfg = WAN_CONFIGS["t2v-1.3B"]
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=0,
        rank=0,
    )
    _original_forward = wan_t2v.model.__class__.forward
    wan_t2v.__class__.generate = t2v_generate
    print("Model loaded.\n")

    videos_root.mkdir(parents=True, exist_ok=True)
    gen_log = load_log(log_path)
    completed = set(gen_log["completed_keys"])

    done, skipped, failed = 0, 0, 0
    total_time = 0.0
    run_num = 0

    for prompt_idx, entry in enumerate(prompts_slice):
        prompt = entry["prompt_en"] if isinstance(entry, dict) else entry
        global_idx = args.start_idx + prompt_idx
        safe_prompt = prompt.replace("/", "_")[:80]

        for mode_cfg in active_modes:
            mode_name = mode_cfg["name"]
            run_num += 1
            video_dir  = videos_root / mode_name
            video_path = video_dir / f"{safe_prompt}-0.mp4"
            key = f"{mode_name}|{prompt}|0"

            if key in completed:
                print(f"[{run_num:3d}/{total}] SKIP  {mode_name} | {prompt[:55]}")
                skipped += 1
                continue

            configure_mode(wan_t2v.model, mode_cfg)
            video_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{run_num:3d}/{total}] GEN   {mode_name} | {prompt[:55]}")
            try:
                t0 = time.time()
                video = wan_t2v.generate(
                    prompt,
                    size=SIZE_CONFIGS["832*480"],
                    frame_num=81,
                    shift=5.0,
                    sample_solver="unipc",
                    sampling_steps=SAMPLE_STEPS,
                    guide_scale=5.0,
                    seed=0,
                    offload_model=True,
                )
                gen_time = time.time() - t0

                cache_video(
                    tensor=video[None],
                    save_file=str(video_path),
                    fps=16,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

                gen_log["runs"].append({
                    "prompt": prompt,
                    "seed": 0,
                    "mode": mode_name,
                    "prompt_index": global_idx,
                    "time_seconds": round(gen_time, 2),
                    "video_path": str(video_path),
                    "timestamp": datetime.now().isoformat(),
                })
                gen_log["completed_keys"].append(key)
                completed.add(key)
                save_log(log_path, gen_log)

                done += 1
                total_time += gen_time
                print(f"         -> saved ({gen_time:.0f}s)")

            except Exception as e:
                import traceback
                print(f"         -> FAILED: {e}")
                traceback.print_exc()
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt, "seed": 0, "mode": mode_name,
                    "prompt_index": global_idx, "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                save_log(log_path, gen_log)

    print("\n" + "=" * 70)
    print(f"Completed: {done}   Skipped: {skipped}   Failed: {failed}")
    if done:
        print(f"Total time: {total_time/3600:.1f}h   Avg/video: {total_time/done:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
