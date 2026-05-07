"""
One-prompt CogVideoX + DiCache diagnostic runner.

Artifacts are written under dicache_results/diagnostic:
  videos/{mode}/{prompt}-0.mp4
  traces/{mode}_trace.json
  plots/{mode}_trace.png
  plots/comparison.png

This intentionally lives outside the main batch harness so diagnostic schedules can
be adjusted without perturbing the 33-prompt sweep scripts.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from diffusers import CogVideoXPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    export_to_video,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils import logging as diffusers_logging


LOGGER = diffusers_logging.get_logger(__name__)
COGVIDEO_ROOT = Path(__file__).resolve().parents[2]
DIAG_ROOT = Path(__file__).resolve().parent


def diagnostic_dicache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        LOGGER.warning(
            "Passing `scale` via `attention_kwargs` when not using PEFT is ineffective."
        )

    step = int(self.cnt)
    batch_size, num_frames, channels, height, width = hidden_states.shape

    timesteps = timestep
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    if self.ofs_embedding is not None:
        ofs_emb = self.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        emb = emb + self.ofs_embedding(ofs_emb)

    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    if getattr(self, "adaptive", False):
        stable_start = int(getattr(self, "stable_start", 15))
        stable_end = int(getattr(self, "stable_end", 48))
        if step < stable_start or step >= stable_end:
            thresh = float(self.thresh_low)
            schedule_region = "low"
        else:
            thresh = float(self.thresh_high)
            schedule_region = "high"
    else:
        thresh = float(self.rel_l1_thresh)
        schedule_region = "fixed"

    calibrate_mode = bool(getattr(self, "calibrate", False))
    force_first_steps = int(getattr(self, "force_first_steps", 0))
    force_last_steps = int(getattr(self, "force_last_steps", 0))
    forced_full = step < force_first_steps or step >= int(self.num_steps) - force_last_steps

    skip_forward = False
    probe_ran = False
    delta_y_value = None
    accumulator_after_probe = float(self.accumulated_rel_l1_distance)
    decision_reason = "warmup_no_previous_probe"

    ori_hidden_states = hidden_states
    ori_encoder_hidden_states = encoder_hidden_states

    if self.previous_probe_hs is not None:
        probe_ran = True
        test_hs = hidden_states.clone()
        test_ehs = encoder_hidden_states.clone()

        for block in self.transformer_blocks[: self.probe_depth]:
            test_hs, test_ehs = block(
                hidden_states=test_hs,
                encoder_hidden_states=test_ehs,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

        delta_y = (
            (test_hs - self.previous_probe_hs).abs().mean()
            / self.previous_probe_hs.abs().mean()
        )
        delta_y_value = float(delta_y.item())
        self.accumulated_rel_l1_distance += delta_y_value
        accumulator_after_probe = float(self.accumulated_rel_l1_distance)

        if calibrate_mode:
            decision_reason = "calibrate_forced_full"
            self.accumulated_rel_l1_distance = 0.0
            self.resume_flag = True
        elif forced_full:
            decision_reason = "configured_forced_full"
            self.accumulated_rel_l1_distance = 0.0
            self.resume_flag = True
        elif self.accumulated_rel_l1_distance < thresh:
            skip_forward = True
            self.resume_flag = False
            decision_reason = "accumulator_below_threshold"
        else:
            self.accumulated_rel_l1_distance = 0.0
            self.resume_flag = True
            decision_reason = "accumulator_reached_threshold"

    if skip_forward:
        ori_hidden_states = hidden_states.clone()

        if len(self.residual_window_hs) >= 2:
            current_probe_res = test_hs - hidden_states
            denom = (
                self.probe_residual_window[-1] - self.probe_residual_window[-2]
            ).abs().mean()
            if denom > 1e-8:
                gamma = (
                    (current_probe_res - self.probe_residual_window[-2]).abs().mean()
                    / denom
                ).clip(1.0, 2.0)
            else:
                gamma = torch.tensor(1.0, device=hidden_states.device)
            hidden_states = (
                hidden_states
                + self.residual_window_hs[-2]
                + gamma * (self.residual_window_hs[-1] - self.residual_window_hs[-2])
            )
        else:
            hidden_states = hidden_states + self.residual_cache_hs

        encoder_hidden_states = encoder_hidden_states + self.residual_cache_ehs
        self.previous_probe_hs = test_hs

    else:
        ori_hidden_states = hidden_states
        ori_encoder_hidden_states_for_residual = encoder_hidden_states

        if probe_ran and self.resume_flag:
            hidden_states = test_hs
            encoder_hidden_states = test_ehs
            unpass_blocks = self.transformer_blocks[self.probe_depth :]
        else:
            unpass_blocks = self.transformer_blocks

        for ind, block in enumerate(unpass_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if not (probe_ran and self.resume_flag) and ind == self.probe_depth - 1:
                if probe_ran:
                    self.previous_probe_hs = test_hs
                else:
                    self.previous_probe_hs = hidden_states.detach().clone()

        if probe_ran and self.resume_flag:
            self.previous_probe_hs = test_hs

        residual_hs = hidden_states - ori_hidden_states
        residual_ehs = encoder_hidden_states - ori_encoder_hidden_states_for_residual
        probe_res = self.previous_probe_hs - ori_hidden_states

        self.residual_cache_hs = residual_hs
        self.residual_cache_ehs = residual_ehs
        self.probe_residual_cache = probe_res

        if len(self.residual_window_hs) < 2:
            self.residual_window_hs.append(residual_hs)
            self.probe_residual_window.append(probe_res)
        else:
            self.residual_window_hs[-2] = self.residual_window_hs[-1]
            self.residual_window_hs[-1] = residual_hs
            self.probe_residual_window[-2] = self.probe_residual_window[-1]
            self.probe_residual_window[-1] = probe_res

    if not self.config.use_rotary_positional_embeddings:
        hidden_states = self.norm_final(hidden_states)
    else:
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    p = self.config.patch_size
    p_t = self.config.patch_size_t
    if p_t is None:
        output = hidden_states.reshape(
            batch_size, num_frames, height // p, width // p, -1, p, p
        )
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = hidden_states.reshape(
            batch_size,
            (num_frames + p_t - 1) // p_t,
            height // p,
            width // p,
            -1,
            p_t,
            p,
            p,
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if hasattr(self, "diagnostic_trace"):
        self.diagnostic_trace.append(
            {
                "step": step,
                "delta_y": delta_y_value,
                "accumulated_rel_l1_before_decision": accumulator_after_probe,
                "accumulated_rel_l1_after_decision": float(self.accumulated_rel_l1_distance),
                "threshold": thresh,
                "schedule_region": schedule_region,
                "decision": "skip" if skip_forward else "full",
                "reason": decision_reason,
                "probe_ran": probe_ran,
                "forced_full": forced_full,
            }
        )

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.resume_flag = False
        self.previous_probe_hs = None
        self.residual_cache_hs = None
        self.residual_cache_ehs = None
        self.probe_residual_cache = None
        self.residual_window_hs = []
        self.probe_residual_window = []

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def mode_config(mode_name: str) -> Dict[str, Any]:
    modes: Dict[str, Dict[str, Any]] = {
        "baseline_probe": {"type": "calibrate", "rel_l1_thresh": 0.0, "probe_depth": 1},
        "fixed_0.20": {"type": "fixed", "rel_l1_thresh": 0.20, "probe_depth": 1},
        "fixed_0.30": {"type": "fixed", "rel_l1_thresh": 0.30, "probe_depth": 1},
        "fixed_0.40": {"type": "fixed", "rel_l1_thresh": 0.40, "probe_depth": 1},
        "adaptive_hi0.30_lo0.05_mid15_48": {
            "type": "adaptive",
            "thresh_low": 0.05,
            "thresh_high": 0.30,
            "stable_start": 15,
            "stable_end": 48,
            "probe_depth": 1,
        },
        "adaptive_hi0.40_lo0.05_mid15_48": {
            "type": "adaptive",
            "thresh_low": 0.05,
            "thresh_high": 0.40,
            "stable_start": 15,
            "stable_end": 48,
            "probe_depth": 1,
        },
        "adaptive_hi0.40_lo0.05_mid15_48_force_last1": {
            "type": "adaptive",
            "thresh_low": 0.05,
            "thresh_high": 0.40,
            "stable_start": 15,
            "stable_end": 48,
            "force_last_steps": 1,
            "probe_depth": 1,
        },
    }
    if mode_name not in modes:
        raise ValueError(f"Unknown mode {mode_name!r}. Available: {', '.join(modes)}")
    return modes[mode_name]


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    with open(args.prompts_json, "r") as f:
        prompts = json.load(f)
    item = prompts[args.prompt_idx]
    return item["prompt_en"] if isinstance(item, dict) and "prompt_en" in item else str(item)


def reset_transformer(transformer, cfg: Dict[str, Any], steps: int) -> None:
    transformer.cnt = 0
    transformer.probe_depth = int(cfg.get("probe_depth", 1))
    transformer.num_steps = steps
    transformer.rel_l1_thresh = float(cfg.get("rel_l1_thresh", cfg.get("thresh_high", 0.0)))
    transformer.ret_ratio = 0.0
    transformer.accumulated_rel_l1_distance = 0.0
    transformer.resume_flag = False
    transformer.previous_probe_hs = None
    transformer.residual_cache_hs = None
    transformer.residual_cache_ehs = None
    transformer.probe_residual_cache = None
    transformer.residual_window_hs = []
    transformer.probe_residual_window = []
    transformer.calibrate = cfg["type"] == "calibrate"
    transformer.adaptive = cfg["type"] == "adaptive"
    transformer.thresh_low = float(cfg.get("thresh_low", 0.0))
    transformer.thresh_high = float(cfg.get("thresh_high", transformer.rel_l1_thresh))
    transformer.stable_start = int(cfg.get("stable_start", 15))
    transformer.stable_end = int(cfg.get("stable_end", 48))
    transformer.force_first_steps = int(cfg.get("force_first_steps", 0))
    transformer.force_last_steps = int(cfg.get("force_last_steps", 0))
    transformer.diagnostic_trace = []


def plot_trace(trace_path: Path, plot_path: Path) -> None:
    with open(trace_path, "r") as f:
        payload = json.load(f)

    trace = payload["trace"]
    steps = [row["step"] for row in trace]
    deltas = [row["delta_y"] for row in trace]
    accum = [row["accumulated_rel_l1_before_decision"] for row in trace]
    thresholds = [row["threshold"] for row in trace]
    skip_steps = [row["step"] for row in trace if row["decision"] == "skip"]
    full_steps = [row["step"] for row in trace if row["decision"] == "full"]

    plt.figure(figsize=(11, 6))
    plt.plot(steps, accum, label="accumulated rel-L1 before decision", linewidth=1.8)
    plt.plot(steps, thresholds, label="threshold", linestyle="--", linewidth=1.4)
    valid_delta_steps = [s for s, d in zip(steps, deltas) if d is not None]
    valid_deltas = [d for d in deltas if d is not None]
    plt.plot(valid_delta_steps, valid_deltas, label="per-step probe delta_y", alpha=0.65)
    if skip_steps:
        plt.scatter(skip_steps, [thresholds[s] for s in skip_steps], marker="v", s=35, label="skip")
    if full_steps:
        plt.scatter(full_steps, [thresholds[s] for s in full_steps], marker="o", s=18, label="full")
    plt.xlabel("Denoising step")
    plt.ylabel("Relative L1")
    plt.title(payload["mode"])
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def plot_comparison(diag_root: Path) -> None:
    trace_paths = sorted((diag_root / "traces").glob("*_trace.json"))
    if not trace_paths:
        raise FileNotFoundError(f"No traces found under {diag_root / 'traces'}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    summaries = []
    for trace_path in trace_paths:
        with open(trace_path, "r") as f:
            payload = json.load(f)
        trace = payload["trace"]
        mode = payload["mode"]
        steps = [row["step"] for row in trace]
        accum = [row["accumulated_rel_l1_before_decision"] for row in trace]
        thresholds = [row["threshold"] for row in trace]
        skip_count = sum(row["decision"] == "skip" for row in trace)
        full_count = sum(row["decision"] == "full" for row in trace)
        summaries.append((mode, skip_count, full_count, payload.get("elapsed_seconds")))
        axes[0].plot(steps, accum, label=f"{mode} accum")
        axes[1].plot(steps, thresholds, label=f"{mode} threshold")

    axes[0].set_ylabel("Accumulated rel-L1")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("Denoising step")
    axes[1].set_ylabel("Threshold")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("CogVideo DiCache one-prompt diagnostic comparison")
    fig.tight_layout()
    out_path = diag_root / "plots" / "comparison.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    summary_path = diag_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            [
                {
                    "mode": mode,
                    "skip_steps": skip_count,
                    "full_steps": full_count,
                    "elapsed_seconds": elapsed,
                }
                for mode, skip_count, full_count, elapsed in summaries
            ],
            f,
            indent=2,
        )


def run_modes(args: argparse.Namespace) -> None:
    prompt = load_prompt(args)
    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    videos_root = DIAG_ROOT / "videos"
    traces_root = DIAG_ROOT / "traces"
    plots_root = DIAG_ROOT / "plots"
    videos_root.mkdir(parents=True, exist_ok=True)
    traces_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    cls = pipe.transformer.__class__
    if not hasattr(cls, "_diagnostic_original_forward"):
        cls._diagnostic_original_forward = cls.forward
    cls.forward = diagnostic_dicache_forward

    safe_prompt = prompt.replace("/", "-")
    for mode_name in mode_names:
        cfg = mode_config(mode_name)
        reset_transformer(pipe.transformer, cfg, args.steps)

        mode_dir = videos_root / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        video_path = mode_dir / f"{safe_prompt}-0.mp4"

        print(f"GPU={gpu_id} mode={mode_name} prompt={prompt!r}", flush=True)
        t0 = time.time()
        video = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).frames[0]
        elapsed = time.time() - t0
        export_to_video(video, str(video_path), fps=args.fps)

        trace = list(pipe.transformer.diagnostic_trace)
        skip_count = sum(row["decision"] == "skip" for row in trace)
        full_count = sum(row["decision"] == "full" for row in trace)
        trace_path = traces_root / f"{mode_name}_trace.json"
        plot_path = plots_root / f"{mode_name}_trace.png"
        payload = {
            "mode": mode_name,
            "config": cfg,
            "prompt": prompt,
            "seed": args.seed,
            "gpu": gpu_id,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "video_path": str(video_path),
            "elapsed_seconds": round(elapsed, 2),
            "skip_steps": skip_count,
            "full_steps": full_count,
            "trace": trace,
        }
        with open(trace_path, "w") as f:
            json.dump(payload, f, indent=2)
        plot_trace(trace_path, plot_path)
        print(
            f"DONE {mode_name}: {elapsed:.1f}s, skips={skip_count}, full={full_count}, "
            f"trace={trace_path}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogVideoX DiCache one-prompt diagnostics.")
    parser.add_argument("--model-id", default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--prompts-json", default=str(COGVIDEO_ROOT / "vbench_eval" / "prompts_subset.json"))
    parser.add_argument("--prompt-idx", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--modes", required=False, default="baseline_probe")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--comparison-after-run",
        action="store_true",
        help="Also regenerate the cross-mode comparison after this process finishes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        for trace_path in sorted((DIAG_ROOT / "traces").glob("*_trace.json")):
            plot_trace(trace_path, DIAG_ROOT / "plots" / f"{trace_path.stem.replace('_trace', '')}_trace.png")
        plot_comparison(DIAG_ROOT)
        print(f"Wrote comparison plot and summary under {DIAG_ROOT}")
        return
    run_modes(args)
    if args.comparison_after_run:
        plot_comparison(DIAG_ROOT)


if __name__ == "__main__":
    main()
