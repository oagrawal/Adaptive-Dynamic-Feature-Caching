import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Speedups computed from per-video latency; baseline = 1265s (dicache_baseline, delta=0, no skipping)
speedups = {
    "dicache_fixed_0.05":          1.92,
    "dicache_fixed_0.07":          2.02,
    "dicache_fixed_0.08":          2.22,
    "dicache_fixed_0.10":          3.02,
    "dicache_fixed_0.15":          3.65,
    "dicache_fixed_0.20":          4.42,
    "dicache_fixed_0.25":          5.05,
    "dicache_fixed_0.30":          5.59,
    "dicache_fixed_0.35":          6.25,
    "dicache_fixed_0.40":          6.79,
    "dicache_fixed_0.60":          7.69,
    "dicache_adaptive_0.05_0.10":  2.04,
    "dicache_adaptive_0.05_0.15":  2.23,
    "dicache_adaptive_0.05_0.20":  2.72,
    "dicache_adaptive_0.05_0.25":  2.39,
    "dicache_adaptive_0.05_0.30":  2.39,
    "dicache_adaptive_0.05_0.35":  2.50,
    "dicache_adaptive_0.05_0.40":  2.52,
    "dicache_adaptive_0.10_0.30":  3.87,
    "dicache_adaptive_0.15_0.40":  4.60,
}

fidelity_dir = "/nfs/oagrawal/HunyuanVideo/dicache_results/fidelity_metrics"
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
print(f"Plotting {len(df)} points.")

colors = {True: "#FF6F61", False: "#6B5B95"}
labels = {True: "Adaptive Modes", False: "Fixed Modes"}

metrics = ["psnr", "ssim", "lpips"]
metric_titles = {
    "psnr": "PSNR (dB) — Higher is Better",
    "ssim": "SSIM — Higher is Better",
    "lpips": "LPIPS — Lower is Better (Inverted Axis)",
}

out_dir = "/nfs/oagrawal/HunyuanVideo/dicache_results"

for metric in metrics:
    plt.figure(figsize=(10, 6))

    fixed_group = df[~df["is_adaptive"]].sort_values("speedup")
    plt.plot(
        fixed_group["speedup"], fixed_group[metric],
        linestyle="-", color="#6B5B95", alpha=0.5,
        label="Fixed Threshold Curve", zorder=1,
    )

    for is_adapt, group in df.groupby("is_adaptive"):
        plt.scatter(
            group["speedup"], group[metric],
            c=colors[is_adapt], label=labels[is_adapt],
            s=120, edgecolors="black", alpha=0.8, zorder=3,
        )

    plt.xlabel("Speedup (x)", fontsize=12)
    plt.ylabel(metric_titles[metric], fontsize=12)
    plt.title(
        f"HunyuanVideo DiCache Pareto: {metric.upper()} vs Speedup (Top-Right is Better)",
        fontsize=14, pad=20,
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    if metric == "lpips":
        plt.gca().invert_yaxis()

    out_path = os.path.join(out_dir, f"pareto_frontier_hv_dicache_{metric}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
