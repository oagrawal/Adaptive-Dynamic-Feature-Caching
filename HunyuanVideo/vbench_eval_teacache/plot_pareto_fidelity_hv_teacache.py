"""
Pareto frontier plot for HunyuanVideo TeaCache fidelity vs speedup.

Data sources:
  - Fidelity: fidelity_metrics/*_vs_hunyuan_baseline.json
  - Speedup: avg generation times from generation logs (hardcoded below)
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

FIDELITY_DIR = "/nfs/oagrawal/HunyuanVideo/vbench_eval_teacache/fidelity_metrics"
OUT_DIR      = "/nfs/oagrawal/HunyuanVideo/vbench_eval_teacache"

# Avg seconds/video from generation logs (baseline = denominator for speedup)
BASELINE_TIME_S = 1358.9
AVG_TIMES = {
    "hunyuan_fixed_0.05":                1364.0,
    "hunyuan_fixed_0.1":                  864.4,
    "hunyuan_adaptive":                   651.6,
    "hunyuan_fixed_0.2":                  541.6,
    "hunyuan_fixed_0.3":                  424.0,
    "hunyuan_tc_adaptive_lo0.1_hi0.3":   627.9,
    "hunyuan_tc_adaptive_lo0.15_hi0.3":  554.4,
    "hunyuan_tc_adaptive_lo0.2_hi0.3":   482.9,
    "hunyuan_tc_adaptive_lo0.1_hi0.25":  654.2,
}

# ---------------------------------------------------------------------------
# Load fidelity JSONs
# ---------------------------------------------------------------------------
rows = []
for json_path in sorted(glob.glob(f"{FIDELITY_DIR}/*_vs_hunyuan_baseline.json")):
    with open(json_path) as f:
        d = json.load(f)
    mode = d["mode"]
    if mode not in AVG_TIMES:
        print(f"  [skip] no timing entry for {mode}")
        continue
    rows.append({
        "mode":        mode,
        "speedup":     BASELINE_TIME_S / AVG_TIMES[mode],
        "psnr":        d["psnr"]["mean"],
        "ssim":        d["ssim"]["mean"],
        "lpips":       d["lpips"]["mean"],
        "is_adaptive": "adaptive" in mode.lower(),
    })

df = pd.DataFrame(rows).sort_values("speedup").reset_index(drop=True)
df = df[df["mode"] != "hunyuan_fixed_0.05"].reset_index(drop=True)

print(f"Plotting {len(df)} modes:")
print(df[["mode", "speedup", "psnr", "ssim", "lpips"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
COLORS = {True: "#FF6F61", False: "#6B5B95"}
LABELS = {True: "Adaptive", False: "Fixed threshold"}

METRICS = {
    "psnr":  "PSNR (dB) ↑",
    "ssim":  "SSIM ↑",
    "lpips": "LPIPS ↓  (axis inverted)",
}


def short_label(mode):
    return mode.replace("hunyuan_tc_", "").replace("hunyuan_", "")


# ---------------------------------------------------------------------------
# One figure per metric
# ---------------------------------------------------------------------------
for metric, ylabel in METRICS.items():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Connect fixed-threshold points with a line
    fixed = df[~df["is_adaptive"]].sort_values("speedup")
    ax.plot(fixed["speedup"], fixed[metric],
            linestyle="-", color="#6B5B95", alpha=0.4, zorder=1)

    for is_adapt, group in df.groupby("is_adaptive"):
        ax.scatter(group["speedup"], group[metric],
                   c=COLORS[is_adapt], label=LABELS[is_adapt],
                   s=120, edgecolors="black", alpha=0.85, zorder=3)
        # Labels intentionally omitted from plot; mode order matches printed table above.
        # for _, row in group.iterrows():
        #     ax.annotate(short_label(row["mode"]),
        #                 (row["speedup"], row[metric]),
        #                 textcoords="offset points", xytext=(6, 4),
        #                 fontsize=8.5)

    ax.set_xlabel("Speedup (×)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"HunyuanVideo TeaCache — {metric.upper()} vs Speedup",
                 fontsize=14, pad=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left" if metric == "lpips" else "upper right")
    if metric == "lpips":
        ax.invert_yaxis()

    out_path = os.path.join(OUT_DIR, f"pareto_frontier_hv_teacache_{metric}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
