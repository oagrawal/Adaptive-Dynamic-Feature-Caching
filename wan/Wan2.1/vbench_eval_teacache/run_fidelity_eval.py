#!/usr/bin/env python3
"""
Fidelity eval for TeaCache modes vs baseline.

Handles filename mismatches: new mode videos are truncated to 80 chars while
baseline videos have full-length filenames. Pairs them by prefix matching.

Usage:
    python run_fidelity_eval.py --mode wan_tc_fixed_0.15 --gpu 0
    python run_fidelity_eval.py --mode wan_tc_fixed_0.15 --gpu 0 --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

METRICS_DIR = "/nfs/oagrawal/CogVideo/dicache_results/metrics"
if METRICS_DIR not in sys.path:
    sys.path.insert(0, METRICS_DIR)

from calculate_lpips import calculate_lpips
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
import imageio
import torchvision.transforms.functional as TF
import tqdm


HERE      = Path(__file__).resolve().parent
VIDEOS    = HERE / "videos"
FIDELITY  = HERE / "fidelity_metrics"
BASELINE  = "wan_baseline"


def load_video(path):
    reader = imageio.get_reader(str(path), "ffmpeg")
    frames = [torch.tensor(f).cuda().permute(2, 0, 1) for f in reader]
    return torch.stack(frames)


def preprocess(gt_video, gen_shape):
    T_gen, _, H_gen, W_gen = gen_shape
    T_gt, _, H_gt, W_gt = gt_video.shape
    if H_gt < H_gen or W_gt < W_gen:
        gt_video = TF.resize(gt_video, [max(H_gen, H_gt), max(W_gen, W_gt)])
        T_gt, _, H_gt, W_gt = gt_video.shape
    start_h = (H_gt - H_gen) // 2
    start_w = (W_gt - W_gen) // 2
    min_t = min(T_gen, T_gt)
    return gt_video[:min_t, :, start_h:start_h + H_gen, start_w:start_w + W_gen]


def build_pairs(gen_dir, gt_dir):
    """
    Match generated videos to baseline videos by prefix.
    Generated filenames may be truncated to 80 chars; baseline has full names.
    """
    gt_files = {f.stem: f for f in Path(gt_dir).glob("*.mp4")}
    pairs = []
    unmatched = []
    for gen_file in sorted(Path(gen_dir).glob("*.mp4")):
        stem = gen_file.stem  # e.g. "A beautiful coastal...-0"  (80-char truncated)
        # Try exact match first
        if stem in gt_files:
            pairs.append((gen_file, gt_files[stem]))
            continue
        # Prefix match: find a gt file whose stem starts with our stem
        matches = [gt_f for gt_stem, gt_f in gt_files.items()
                   if gt_stem.startswith(stem)]
        if len(matches) == 1:
            pairs.append((gen_file, matches[0]))
        elif len(matches) > 1:
            unmatched.append((gen_file, f"ambiguous ({len(matches)} matches)"))
        else:
            unmatched.append((gen_file, "no match"))
    return pairs, unmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    FIDELITY.mkdir(exist_ok=True)

    gen_dir = VIDEOS / args.mode
    gt_dir  = VIDEOS / BASELINE
    out_json = FIDELITY / f"{args.mode}_vs_{BASELINE}.json"

    pairs, unmatched = build_pairs(gen_dir, gt_dir)
    print(f"Mode: {args.mode}")
    print(f"Matched: {len(pairs)} pairs")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for f, reason in unmatched:
            print(f"  {f.name}: {reason}")

    if args.dry_run:
        for gen, gt in pairs[:5]:
            print(f"  {gen.name}")
            print(f"  -> {gt.name}")
        return

    psnr_all, ssim_all, lpips_all = [], [], []

    for gen_path, gt_path in tqdm.tqdm(pairs):
        gen_video = load_video(gen_path)
        gt_video  = load_video(gt_path)
        gt_video  = preprocess(gt_video, gen_video.shape)
        gen_video = gen_video[:gt_video.shape[0]]

        gen_t = (gen_video.unsqueeze(0) / 255.0).cpu()
        gt_t  = (gt_video.unsqueeze(0)  / 255.0).cpu()

        p = calculate_psnr(gt_t, gen_t)
        psnr_all.append(np.mean(list(p["value"].values())))

        s = calculate_ssim(gt_t, gen_t)
        ssim_all.append(np.mean(list(s["value"].values())))

        l = calculate_lpips(gt_t, gen_t, device="cuda")
        lpips_all.append(np.mean(list(l["value"].values())))

    result = {
        "mode": args.mode,
        "baseline": BASELINE,
        "num_videos": len(pairs),
        "psnr":  {"mean": float(np.mean(psnr_all)),  "std": float(np.std(psnr_all))},
        "ssim":  {"mean": float(np.mean(ssim_all)),  "std": float(np.std(ssim_all))},
        "lpips": {"mean": float(np.mean(lpips_all)), "std": float(np.std(lpips_all))},
    }
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nPSNR: {result['psnr']['mean']:.4f}  SSIM: {result['ssim']['mean']:.4f}  LPIPS: {result['lpips']['mean']:.4f}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
