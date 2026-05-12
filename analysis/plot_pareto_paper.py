"""
Unified Pareto plots for the paper.

Generates one 3-panel PNG (PSNR | SSIM | LPIPS vs Speedup) per combo.
Run from any directory:
    python /nfs/oagrawal/analysis/plot_pareto_paper.py
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/nfs/oagrawal")

# ---------------------------------------------------------------------------
# Speedup data (copied verbatim from per-combo scripts)
# ---------------------------------------------------------------------------

_HV_TC_AVG_TIMES = {
    "hunyuan_fixed_0.05":               1364.0,
    "hunyuan_fixed_0.1":                 864.4,
    "hunyuan_adaptive":                  651.6,
    "hunyuan_fixed_0.2":                 541.6,
    "hunyuan_fixed_0.3":                 424.0,
    "hunyuan_tc_adaptive_lo0.1_hi0.3":  627.9,
    "hunyuan_tc_adaptive_lo0.15_hi0.3": 554.4,
    "hunyuan_tc_adaptive_lo0.2_hi0.3":  482.9,
    "hunyuan_tc_adaptive_lo0.1_hi0.25": 654.2,
}
_HV_TC_BASELINE = 1358.9

_COG_TC_SPEEDUPS = {
    "cogvideo_fixed_0.1":                 1.41,
    "cogvideo_fixed_0.2":                 1.75,
    "cogvideo_fixed_0.22":                1.76,
    "cogvideo_fixed_0.25":                1.90,
    "cogvideo_fixed_0.3":                 2.19,
    "cogvideo_adaptive_0.1_17_0.3":       1.94,
    "cogvideo_adaptive_0.1_20_0.3":       1.87,
    "cogvideo_adaptive3_0.30_17_0.35_48": 2.31,
    "cogvideo_adaptive3_0.30_17_0.40_48": 2.31,
}

_HV_EC_SPEEDUPS = {
    "easycache_fixed_0.025":                  2.060,
    "easycache_fixed_0.0375":                 2.417,
    "easycache_fixed_0.050":                  2.700,
    "easycache_fixed_0.060":                  2.869,
    "easycache_fixed_0.075":                  3.115,
    "easycache_adaptive":                     2.420,
    "easycache_adaptive_0.025_0.075":         2.547,
    "easycache_adaptive_0.0375_0.050":        2.529,
    "easycache_adaptive_0.010_0.050":         2.167,
    "easycache_adaptive_0.010_0.075":         2.276,
    "easycache_adaptive_0.010_0.100":         2.323,
    "easycache_adaptive_0.025_0.075_f12l10":  2.384,
    "easycache_adaptive_0.025_0.090_f15l4":   2.385,
    "easycache_adaptive_0.025_0.150_f15l4":   2.457,
    "easycache_adaptive_0.010_0.120_f12l4":   2.203,
    "easycache_adaptive_0.050_0.120_f15l6":   2.840,
    "easycache_adaptive_0.025_0.050_f4l10":   2.438,
    "easycache_adaptive_0.030_0.050_f4l10":   2.506,
    "easycache_adaptive_0.025_0.045_f4l10":   2.387,
    "easycache_fixed_0.040":                  2.471,
    "easycache_fixed_0.045":                  2.554,
    "easycache_adaptive_0.030_0.060_f4l10":   2.599,
    "easycache_adaptive_0.035_0.065_f4l10":   2.719,
}

_WAN_EC_SPEEDUPS = {
    "wan_ec_fixed_0.020":         1.73,
    "wan_ec_fixed_0.025":         1.86,
    "wan_ec_fixed_0.040":         2.12,
    "wan_ec_fixed_0.050":         2.28,
    "wan_ec_fixed_0.060":         2.40,
    "wan_ec_fixed_0.070":         2.49,
    "wan_ec_fixed_0.080":         2.57,
    "wan_ec_fixed_0.090":         2.62,
    "wan_ec_fixed_0.125":         2.89,
    "wan_ec_fixed_0.150":         2.98,
    "wan_ec_fixed_0.175":         3.09,
    "wan_ec_fixed_0.200":         3.15,
    "wan_ec_fixed_0.225":         3.25,
    "wan_ec_fixed_0.250":         3.28,
    "wan_ec_adaptive_16_020040":  1.92,
    "wan_ec_adaptive_16_025_020": 2.45,
    "wan_ec_adaptive_16_050_020": 2.68,
    "wan_ec_adaptive_16_075_020": 2.86,
    "wan_ec_adaptive_16_100_020": 2.93,
    "wan_ec_adaptive":            2.10,
}

_HV_DC_SPEEDUPS = {
    "dicache_fixed_0.05":         1.92,
    "dicache_fixed_0.07":         2.02,
    "dicache_fixed_0.08":         2.22,
    "dicache_fixed_0.10":         3.02,
    "dicache_fixed_0.15":         3.65,
    "dicache_fixed_0.20":         4.42,
    "dicache_fixed_0.25":         5.05,
    "dicache_fixed_0.30":         5.59,
    "dicache_fixed_0.35":         6.25,
    "dicache_fixed_0.40":         6.79,
    "dicache_fixed_0.60":         7.69,
    "dicache_adaptive_0.05_0.10": 2.04,
    "dicache_adaptive_0.05_0.15": 2.23,
    "dicache_adaptive_0.05_0.20": 2.72,
    "dicache_adaptive_0.05_0.25": 2.39,
    "dicache_adaptive_0.05_0.30": 2.39,
    "dicache_adaptive_0.05_0.35": 2.50,
    "dicache_adaptive_0.05_0.40": 2.52,
    "dicache_adaptive_0.10_0.30": 3.87,
    "dicache_adaptive_0.15_0.40": 4.60,
}

_WAN_DC_SPEEDUPS = {
    "wan_dc_fixed_0.05":              1.78,
    "wan_dc_fixed_0.07":              2.27,
    "wan_dc_fixed_0.08":              2.44,
    "wan_dc_fixed_0.10":              2.58,
    "wan_dc_fixed_0.15":              3.16,
    "wan_dc_fixed_0.20":              3.66,
    "wan_dc_fixed_0.225":             3.97,
    "wan_dc_fixed_0.25":              4.03,
    "wan_dc_adaptive_hi0.225_lo0.05": 2.28,
    "wan_dc_adaptive_hi0.225_lo0.07": 2.73,
    "wan_dc_adaptive_hi0.225_lo0.10": 2.89,
    "wan_dc_adaptive_hi0.225_lo0.15": 3.33,
    "wan_dc_adaptive_hi0.25_lo0.05":  2.29,
    "wan_dc_adaptive_hi0.25_lo0.07":  2.76,
    "wan_dc_adaptive_hi0.25_lo0.10":  2.94,
    "wan_dc_adaptive_hi0.25_lo0.15":  3.39,
}

# ---------------------------------------------------------------------------
# Combo configs
# ---------------------------------------------------------------------------

COMBO_CONFIGS = [
    {
        "title": "TeaCache — HunyuanVideo",
        "fidelity_dir": ROOT / "HunyuanVideo/vbench_eval_teacache/fidelity_metrics",
        "speedup_source": {"type": "avg_times", "data": _HV_TC_AVG_TIMES, "baseline": _HV_TC_BASELINE},
        "out_path": ROOT / "HunyuanVideo/vbench_eval_teacache/pareto_hv_teacache.png",
        "exclude": {"hunyuan_fixed_0.05"},
    },
    {
        "title": "TeaCache — Wan 2.1",
        "fidelity_dir": ROOT / "wan/Wan2.1/vbench_eval_teacache/fidelity_metrics",
        "speedup_source": {
            "type": "logs",
            "log_dir": ROOT / "wan/Wan2.1/vbench_eval_teacache/videos",
            "baseline_mode": "wan_baseline",
        },
        "out_path": ROOT / "wan/Wan2.1/vbench_eval_teacache/pareto_wan_teacache.png",
        "exclude": set(),
    },
    {
        "title": "TeaCache — CogVideo",
        "fidelity_dir": ROOT / "CogVideo/vbench_eval_teacache/fidelity_metrics",
        "speedup_source": {"type": "direct", "data": _COG_TC_SPEEDUPS},
        "out_path": ROOT / "CogVideo/vbench_eval_teacache/pareto_cog_teacache.png",
        "exclude": set(),
    },
    {
        "title": "EasyCache — HunyuanVideo",
        "fidelity_dir": ROOT / "HunyuanVideo/vbench_eval_easycache/fidelity_metrics",
        "speedup_source": {"type": "direct", "data": _HV_EC_SPEEDUPS},
        "out_path": ROOT / "HunyuanVideo/vbench_eval_easycache/pareto_hv_easycache.png",
        "exclude": set(),
    },
    {
        "title": "EasyCache — Wan 2.1",
        "fidelity_dir": ROOT / "wan/Wan2.1/vbench_eval_easycache/fidelity_metrics",
        "speedup_source": {"type": "direct", "data": _WAN_EC_SPEEDUPS},
        "out_path": ROOT / "wan/Wan2.1/vbench_eval_easycache/pareto_wan_easycache.png",
        "exclude": set(),
    },
    {
        "title": "DiCache — HunyuanVideo",
        "fidelity_dir": ROOT / "HunyuanVideo/dicache_results/fidelity_metrics",
        "speedup_source": {"type": "direct", "data": _HV_DC_SPEEDUPS},
        "out_path": ROOT / "HunyuanVideo/dicache_results/pareto_hv_dicache.png",
        "exclude": set(),
    },
    {
        "title": "DiCache — Wan 2.1",
        "fidelity_dir": ROOT / "wan/Wan2.1/dicache_results/fidelity_metrics",
        "speedup_source": {"type": "direct", "data": _WAN_DC_SPEEDUPS},
        "out_path": ROOT / "wan/Wan2.1/pareto_wan_dicache.png",  # dicache_results/ is root-owned
        "exclude": set(),
    },
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

COLOR_FIXED    = "#6B5B95"
COLOR_ADAPTIVE = "#FF6F61"


def _speedup_from_logs(log_dir: Path, baseline_mode: str) -> dict[str, float]:
    times: dict[str, list[float]] = defaultdict(list)
    for log in glob.glob(str(log_dir / "generation_log_*.json")):
        for run in json.loads(Path(log).read_text()).get("runs", []):
            if "time_seconds" in run:
                times[run["mode"]].append(run["time_seconds"])
    if baseline_mode not in times:
        raise RuntimeError(f"No timing for baseline '{baseline_mode}' in {log_dir}")
    baseline = sum(times[baseline_mode]) / len(times[baseline_mode])
    return {mode: baseline / (sum(ts) / len(ts)) for mode, ts in times.items() if mode != baseline_mode}


def load_combo(cfg: dict) -> pd.DataFrame:
    src = cfg["speedup_source"]
    if src["type"] == "direct":
        speedup_map = src["data"]
    elif src["type"] == "avg_times":
        speedup_map = {m: src["baseline"] / t for m, t in src["data"].items()}
    elif src["type"] == "logs":
        speedup_map = _speedup_from_logs(src["log_dir"], src["baseline_mode"])
    else:
        raise ValueError(f"Unknown speedup source type: {src['type']}")

    rows = []
    for json_path in sorted(cfg["fidelity_dir"].glob("*.json")):
        if json_path.name == "all_fidelity_results.json":
            continue
        d = json.loads(json_path.read_text())
        mode = d["mode"]
        if mode in cfg["exclude"]:
            continue
        if mode not in speedup_map:
            print(f"  [skip] no speedup for {mode}")
            continue
        rows.append({
            "mode":        mode,
            "speedup":     speedup_map[mode],
            "psnr":        d["psnr"]["mean"],
            "ssim":        d["ssim"]["mean"],
            "lpips":       d["lpips"]["mean"],
            "is_adaptive": "adaptive" in mode,
        })

    df = pd.DataFrame(rows).sort_values("speedup").reset_index(drop=True)
    print(f"  Loaded {len(df)} points")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRICS = [
    ("psnr",  "PSNR (dB)", False),
    ("ssim",  "SSIM",      False),
    ("lpips", "LPIPS",     True),   # True = invert y-axis
]


def plot_combo(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fixed    = df[~df["is_adaptive"]].sort_values("speedup")
    adaptive = df[df["is_adaptive"]]

    for ax, (metric, ylabel, invert) in zip(axes, METRICS):
        # Fixed-threshold connecting curve
        if not fixed.empty:
            ax.plot(
                fixed["speedup"], fixed[metric],
                linestyle="-", color=COLOR_FIXED, alpha=0.4, zorder=1,
            )

        # Scatter: fixed
        if not fixed.empty:
            ax.scatter(
                fixed["speedup"], fixed[metric],
                c=COLOR_FIXED, s=100, edgecolors="black", alpha=0.85,
                zorder=3, label="Fixed",
            )

        # Scatter: adaptive
        if not adaptive.empty:
            ax.scatter(
                adaptive["speedup"], adaptive[metric],
                c=COLOR_ADAPTIVE, s=100, edgecolors="black", alpha=0.85,
                zorder=3, label="Adaptive",
            )

        ax.set_xlabel("Speedup (×)", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(metric.upper(), fontsize=22)
        ax.legend(loc="upper right", fontsize=15)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.5)

        if invert:
            ax.invert_yaxis()

    fig.suptitle(title, fontsize=24, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for cfg in COMBO_CONFIGS:
        print(f"\n--- {cfg['title']} ---")
        df = load_combo(cfg)
        if df.empty:
            print("  No data — skipping")
            continue
        plot_combo(df, cfg["title"], cfg["out_path"])
