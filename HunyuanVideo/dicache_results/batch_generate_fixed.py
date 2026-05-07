#!/usr/bin/env python3
"""
Batch video generation for HunyuanVideo DiCache VBench evaluation.

This is derived from unedited_dicache/run_hunyuanvideo_dicache.py and keeps that
reference file untouched. It loads HunyuanVideo once, then loops over prompts and
modes with resumable VBench-compatible outputs:

    dicache_results/videos/{mode}/{prompt}-0.mp4

Use --start-idx/--end-idx for prompt splits and --modes for one or more modes.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
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


def dicache_forward(
    self,
    x: "torch.Tensor",
    t: "torch.Tensor",
    text_states: "torch.Tensor" = None,
    text_mask: "torch.Tensor" = None,
    text_states_2: Optional["torch.Tensor"] = None,
    freqs_cos: Optional["torch.Tensor"] = None,
    freqs_sin: Optional["torch.Tensor"] = None,
    guidance: "torch.Tensor" = None,
    return_dict: bool = True,
) -> Union["torch.Tensor", Dict[str, "torch.Tensor"]]:
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
    skip_forward = False
    probe_ran = False

    # Online probe profiling scheme from the original DiCache script.
    if (
        self.cnt >= int(self.ret_ratio * self.num_steps)
        and self.previous_input is not None
        and self.previous_probe_img is not None
    ):
        probe_ran = True
        test_img, test_txt = img.clone(), txt.clone()
        probe_blocks = self.double_blocks[0:self.probe_depth]
        for probe_block in probe_blocks:
            test_double_block_args = [
                test_img,
                test_txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]
            test_img, test_txt = probe_block(*test_double_block_args)

        delta_y = (
            (test_img - self.previous_probe_img).abs().mean()
            / self.previous_probe_img.abs().mean()
        )
        self.accumulated_rel_l1_distance += delta_y

        effective_thresh = self.rel_l1_thresh
        if getattr(self, "adaptive", False):
            stable_start = self.stable_start
            stable_end = self.stable_end
            if self.cnt < stable_start or self.cnt >= stable_end:
                effective_thresh = self.thresh_low
            else:
                effective_thresh = self.thresh_high

        if self.accumulated_rel_l1_distance <= effective_thresh:
            skip_forward = True
            self.resume_flag = False
        else:
            self.accumulated_rel_l1_distance = 0
            self.resume_flag = True

    if skip_forward:
        ori_img = img.clone()

        if len(self.residual_window) >= 2:
            current_residual_indicator = test_img - img
            gamma = (
                (current_residual_indicator - self.probe_residual_window[-2]).abs().mean()
                / (self.probe_residual_window[-1] - self.probe_residual_window[-2]).abs().mean()
            ).clip(1, 1.5)
            img = img + self.residual_window[-2] + gamma * (
                self.residual_window[-1] - self.residual_window[-2]
            )
        else:
            img = img + self.residual_cache

        self.previous_probe_img = test_img
        self.previous_input = ori_img
    else:
        ori_img = img
        if self.resume_flag and probe_ran:
            img = test_img
            txt = test_txt
            unpass_blocks = self.double_blocks[self.probe_depth:]
        else:
            unpass_blocks = self.double_blocks

        for index_block, block in enumerate(unpass_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]

            img, txt = block(*double_block_args)

            if index_block == self.probe_depth - 1:
                if probe_ran:
                    self.previous_probe_img = test_img
                else:
                    self.previous_probe_img = img

        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                ]
                x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]
        self.residual_cache = img - ori_img
        self.probe_residual_cache = self.previous_probe_img - ori_img
        self.previous_input = ori_img

        if len(self.residual_window) <= 2:
            self.residual_window.append(self.residual_cache)
            self.probe_residual_window.append(self.probe_residual_cache)
        else:
            self.residual_window[-2] = self.residual_window[-1]
            self.residual_window[-1] = self.residual_cache
            self.probe_residual_window[-2] = self.probe_residual_window[-1]
            self.probe_residual_window[-1] = self.probe_residual_cache

    img = self.final_layer(img, vec)
    img = self.unpatchify(img, tt, th, tw)

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0
        self.resume_flag = False
        self.residual_window = []
        self.probe_residual_window = []

    if return_dict:
        out["x"] = img
        return out
    return img


MODES = [
    {"name": "dicache_baseline", "type": "baseline"},
    {"name": "dicache_fixed_0.05", "type": "fixed", "thresh": 0.05},
    {"name": "dicache_fixed_0.10", "type": "fixed", "thresh": 0.10},
    {"name": "dicache_fixed_0.15", "type": "fixed", "thresh": 0.15},
    {"name": "dicache_fixed_0.20", "type": "fixed", "thresh": 0.20},
    {"name": "dicache_fixed_0.25", "type": "fixed", "thresh": 0.25},
    {"name": "dicache_fixed_0.30", "type": "fixed", "thresh": 0.30},
    {"name": "dicache_fixed_0.35", "type": "fixed", "thresh": 0.35},
    {"name": "dicache_fixed_0.40", "type": "fixed", "thresh": 0.40},
    {"name": "dicache_fixed_0.07", "type": "fixed", "thresh": 0.07},
    {"name": "dicache_fixed_0.08", "type": "fixed", "thresh": 0.08},
    {"name": "dicache_fixed_0.60", "type": "fixed", "thresh": 0.60},
    {
        "name": "dicache_adaptive_0.05_0.20",
        "type": "adaptive",
        "thresh_low": 0.05,
        "thresh_high": 0.20,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.10_0.30",
        "type": "adaptive",
        "thresh_low": 0.10,
        "thresh_high": 0.30,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.15_0.40",
        "type": "adaptive",
        "thresh_low": 0.15,
        "thresh_high": 0.40,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.05_0.10",
        "type": "adaptive",
        "thresh_low": 0.05,
        "thresh_high": 0.10,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.05_0.15",
        "type": "adaptive",
        "thresh_low": 0.05,
        "thresh_high": 0.15,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.05_0.25",
        "type": "adaptive",
        "thresh_low": 0.05,
        "thresh_high": 0.25,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.05_0.30",
        "type": "adaptive",
        "thresh_low": 0.05,
        "thresh_high": 0.30,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.05_0.35",
        "type": "adaptive",
        "thresh_low": 0.05,
        "thresh_high": 0.35,
        "stable_start": 8,
        "stable_end": 40,
    },
    {
        "name": "dicache_adaptive_0.05_0.40",
        "type": "adaptive",
        "thresh_low": 0.05,
        "thresh_high": 0.40,
        "stable_start": 8,
        "stable_end": 40,
    },
]


def configure_dicache(transformer, original_forward, mode_cfg, infer_steps, ret_ratio):
    transformer.cnt = 0
    transformer.probe_depth = 1
    transformer.num_steps = infer_steps
    transformer.ret_ratio = ret_ratio
    transformer.accumulated_rel_l1_distance = 0
    transformer.residual_cache = None
    transformer.probe_residual_cache = None
    transformer.residual_window = []
    transformer.probe_residual_window = []
    transformer.resume_flag = False
    transformer.previous_input = None
    transformer.previous_probe_img = None

    if mode_cfg["type"] == "baseline":
        transformer.adaptive = False
        transformer.__class__.forward = original_forward
        return

    transformer.__class__.forward = dicache_forward
    transformer.adaptive = mode_cfg["type"] == "adaptive"

    if transformer.adaptive:
        transformer.rel_l1_thresh = mode_cfg["thresh_low"]
        transformer.thresh_low = mode_cfg["thresh_low"]
        transformer.thresh_high = mode_cfg["thresh_high"]
        transformer.stable_start = mode_cfg["stable_start"]
        transformer.stable_end = mode_cfg["stable_end"]
    else:
        transformer.rel_l1_thresh = mode_cfg["thresh"]
        transformer.thresh_low = mode_cfg["thresh"]
        transformer.thresh_high = mode_cfg["thresh"]
        transformer.stable_start = 0
        transformer.stable_end = infer_steps


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


def sanitize_prompt_filename(prompt):
    return prompt.replace("/", "")


def main():
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument("--prompts-file", type=str, default="vbench_eval_teacache/prompts_subset.json")
    batch_parser.add_argument("--output-dir", type=str, default="dicache_results/videos")
    batch_parser.add_argument("--generation-seed", type=int, default=0)
    batch_parser.add_argument("--start-idx", type=int, default=0)
    batch_parser.add_argument("--end-idx", type=int, default=-1)
    batch_parser.add_argument("--modes", type=str, default="all", help="Comma-separated mode names or 'all'")
    batch_parser.add_argument("--ret-ratio", type=float, default=0.0)
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
        if len(modes) != len(names):
            found = {m["name"] for m in modes}
            missing = [name for name in names if name not in found]
            print(f"ERROR: unknown modes: {missing}")
            print(f"Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed = batch_args.generation_seed
    output_dir = batch_args.output_dir
    start_idx = batch_args.start_idx
    total_videos = len(prompts) * len(modes)

    print("=" * 70)
    print("DiCache VBench Batch Generation - HunyuanVideo")
    print("=" * 70)
    print(f"Prompts: [{start_idx}, {end_idx}) = {len(prompts)} prompts")
    print(f"Modes: {[m['name'] for m in modes]}")
    print(f"Total: {total_videos} videos")
    print(f"Output: {output_dir}")
    print(f"ret_ratio: {batch_args.ret_ratio}")
    print("=" * 70)

    if batch_args.dry_run:
        for entry in prompts:
            prompt = entry["prompt_en"]
            filename_prompt = sanitize_prompt_filename(prompt)
            for mode in modes:
                video_path = os.path.join(output_dir, mode["name"], f"{filename_prompt}-{seed}.mp4")
                tag = "EXISTS" if os.path.exists(video_path) else "NEW"
                print(f"  {tag:6s} {mode['name']} | {prompt[:60]}")
        return

    if "--save-path" not in remaining_argv:
        remaining_argv += ["--save-path", output_dir]
    sys.argv = [sys.argv[0]] + remaining_argv

    global torch

    import torch
    import hyvideo_pyc_loader  # noqa: F401
    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    args = parse_args()

    mode_tag = "all" if batch_args.modes == "all" else "_".join(m["name"] for m in modes)
    log_filename = f"generation_log_{mode_tag}_{start_idx}-{end_idx}.json"
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
    original_forward = transformer.__class__.forward

    completed, skipped, failed = 0, 0, 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        filename_prompt = sanitize_prompt_filename(prompt)
        global_idx = start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, f"{filename_prompt}-{seed}.mp4")
            run_key = f"{mode_name}|{prompt}|{seed}"
            run_num = prompt_idx * len(modes) + mode_idx + 1

            if os.path.exists(video_path) or run_key in gen_log.get("completed_keys", []):
                logger.info(f"[{run_num:3d}/{total_videos}] SKIP  {mode_name} | {prompt[:55]}...")
                skipped += 1
                continue

            configure_dicache(transformer, original_forward, mode, args.infer_steps, batch_args.ret_ratio)
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

                gen_log["runs"].append({
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode_name,
                    "time_seconds": round(gen_time, 2),
                    "video_path": video_path,
                    "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                    "ret_ratio": batch_args.ret_ratio,
                    "config": {k: v for k, v in mode.items() if k != "name"},
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
    print("DiCache batch generation complete")
    print(f"Completed: {completed}   Skipped: {skipped}   Failed: {failed}")
    if completed:
        print(f"Total time: {total_gen_time/3600:.1f}h   Avg/video: {total_gen_time/completed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
