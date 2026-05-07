import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Baseline latency: 1357s (mean across 33 prompts)
speedups = {
    "easycache_fixed_0.025":                2.060,
    "easycache_fixed_0.0375":               2.417,
    "easycache_fixed_0.050":                2.700,
    "easycache_fixed_0.060":                2.869,
    "easycache_fixed_0.075":                3.115,
    "easycache_adaptive":                   2.420,
    "easycache_adaptive_0.025_0.075":       2.547,
    "easycache_adaptive_0.0375_0.050":      2.529,
    "easycache_adaptive_0.010_0.050":       2.167,
    "easycache_adaptive_0.010_0.075":       2.276,
    "easycache_adaptive_0.010_0.100":       2.323,
    "easycache_adaptive_0.025_0.075_f12l10": 2.384,
    # Phase 3 (2026-04-30): corrected first_steps, varied thresh_low
    "easycache_adaptive_0.025_0.090_f15l4": 2.385,
    "easycache_adaptive_0.025_0.150_f15l4": 2.457,
    "easycache_adaptive_0.010_0.120_f12l4": 2.203,
    "easycache_adaptive_0.050_0.120_f15l6": 2.840,
    # Late-only adaptive modes (2026-05-02): no extra early low-threshold region; protect late tail only
    "easycache_adaptive_0.025_0.050_f4l10": 2.438,
    "easycache_adaptive_0.030_0.050_f4l10": 2.506,
    "easycache_adaptive_0.025_0.045_f4l10": 2.387,
    # Phase 4 (2026-05-03): fill fixed gap 0.0375–0.050; push f4l10 frontier to higher speedups
    "easycache_fixed_0.040":                2.471,
    "easycache_fixed_0.045":                2.554,
    "easycache_adaptive_0.030_0.060_f4l10": 2.599,
    "easycache_adaptive_0.035_0.065_f4l10": 2.719,
}

fidelity_dir = "/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache/fidelity_metrics"
data = []

for filename in os.listdir(fidelity_dir):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join(fidelity_dir, filename)) as f:
        try:
            js = json.load(f)
            mode = js["mode"]
            if mode in speedups:
                data.append({
                    "mode": mode,
                    "speedup": speedups[mode],
                    "psnr": js["psnr"]["mean"],
                    "ssim": js["ssim"]["mean"],
                    "lpips": js["lpips"]["mean"],
                    "is_adaptive": "adaptive" in mode,
                })
        except Exception as e:
            print(f"Warning: Failed to parse {filename}: {e}")

if not data:
    print("Error: No data points found.")
    exit(1)

df = pd.DataFrame(data)
print(f"Loaded {len(df)} points.")
print(df.sort_values("speedup")[["mode", "speedup", "psnr", "ssim", "lpips"]].to_string(index=False))


def pareto_frontier(df, metric, higher_is_better=True):
    """Return boolean mask of non-dominated points (maximize speedup, optimize metric)."""
    pts = df[["speedup", metric]].values
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            spd_dom = pts[j, 0] >= pts[i, 0]
            if higher_is_better:
                met_dom = pts[j, 1] >= pts[i, 1]
            else:
                met_dom = pts[j, 1] <= pts[i, 1]
            if spd_dom and met_dom and (pts[j, 0] > pts[i, 0] or
               (higher_is_better and pts[j, 1] > pts[i, 1]) or
               (not higher_is_better and pts[j, 1] < pts[i, 1])):
                dominated[i] = True
                break
    return ~dominated


def short_label(mode):
    s = mode.replace("easycache_", "")
    if s == "adaptive":
        return "ad:default"
    if s.startswith("fixed_"):
        return "fix:" + s[len("fixed_"):]
    if s.startswith("adaptive_"):
        return "ad:" + s[len("adaptive_"):]
    return s


def add_labels(ax, df, metric):
    # Sort by (speedup, metric) so stagger order is deterministic
    rows = df.sort_values(["speedup", metric]).to_dict("records")
    # Group rows by x proximity (within 0.04 speedup)
    groups = []
    for r in rows:
        placed = False
        for g in groups:
            if abs(g[0]["speedup"] - r["speedup"]) < 0.04:
                g.append(r)
                placed = True
                break
        if not placed:
            groups.append([r])

    for g in groups:
        g = sorted(g, key=lambda r: r[metric])
        n = len(g)
        for i, r in enumerate(g):
            # Alternate above/below within each cluster
            sign = 1 if i % 2 == 0 else -1
            magnitude = 12 + 10 * (i // 2)
            x_off = 6 if i % 2 == 0 else -6
            y_off = sign * magnitude
            ax.annotate(
                short_label(r["mode"]),
                xy=(r["speedup"], r[metric]),
                xytext=(x_off, y_off),
                textcoords="offset points",
                fontsize=6.5,
                color="#333333",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
                ha="left" if x_off > 0 else "right",
                va="center",
            )


metrics = ["psnr", "ssim", "lpips"]
metric_titles = {
    "psnr": "PSNR (dB) ↑",
    "ssim": "SSIM ↑",
    "lpips": "LPIPS ↓",
}
higher_is_better = {"psnr": True, "ssim": True, "lpips": False}

out_dir = "/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache"

for metric in metrics:
    fig, ax = plt.subplots(figsize=(14, 7))

    fixed = df[~df["is_adaptive"]].sort_values("speedup")
    adaptive = df[df["is_adaptive"]]

    # Fixed threshold curve
    ax.plot(
        fixed["speedup"], fixed[metric],
        linestyle="-", color="#6B5B95", alpha=0.4,
        linewidth=1.5, zorder=1,
    )

    # Pareto frontier line
    mask = pareto_frontier(df, metric, higher_is_better[metric])
    frontier = df[mask].sort_values("speedup")
    ax.plot(
        frontier["speedup"], frontier[metric],
        linestyle="--", color="#2ecc71", linewidth=2,
        alpha=0.85, zorder=2, label="Pareto Frontier",
    )

    # Fixed points
    ax.scatter(
        fixed["speedup"], fixed[metric],
        c="#6B5B95", s=130, edgecolors="black",
        alpha=0.9, zorder=4, label="Fixed Threshold",
    )

    # Adaptive mode naming:
    # - f15l4 / f12l4 / f15l6 are phase-3 boundary sweeps.
    # - f4l10 modes are late-only adaptive: first 5 steps are already forced full,
    #   so first_steps=4 adds no extra early low-threshold decision region.
    # For plot readability, all adaptive variants share one visual style.
    if not adaptive.empty:
        ax.scatter(
            adaptive["speedup"], adaptive[metric],
            c="#FF6F61", s=120, edgecolors="black",
            alpha=0.85, zorder=5, label="Adaptive Threshold",
            marker="o",
        )

    ax.set_xlabel("Speedup (×)", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} — {metric_titles[metric]}", fontsize=12)
    ax.set_title(
        f"HunyuanVideo EasyCache: {metric.upper()} vs Speedup",
        fontsize=14, pad=14,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left" if metric == "lpips" else "upper right", fontsize=10)

    add_labels(ax, df, metric)

    if metric == "lpips":
        ax.invert_yaxis()

    out_path = os.path.join(out_dir, f"pareto_frontier_hv_easycache_{metric}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
