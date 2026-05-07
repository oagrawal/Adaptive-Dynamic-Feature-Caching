import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Speedups from per-video latency in generation_log_*.json; baseline mean = 1019s.
speedups = {
    "cogvideo_fixed_0.1":                      1.41,
    "cogvideo_fixed_0.2":                      1.75,
    "cogvideo_fixed_0.22":                     1.76,
    "cogvideo_fixed_0.25":                     1.90,
    "cogvideo_fixed_0.3":                      2.19,
    "cogvideo_adaptive_0.1_17_0.3":            1.94,
    "cogvideo_adaptive_0.1_20_0.3":            1.87,
    "cogvideo_adaptive3_0.30_17_0.35_48":      2.31,
    "cogvideo_adaptive3_0.30_17_0.40_48":      2.31,
}

fidelity_dir = "/nfs/oagrawal/CogVideo/vbench_eval_teacache/fidelity_metrics"
data = []

for filename in os.listdir(fidelity_dir):
    if not filename.endswith(".json") or filename == "all_fidelity_results.json":
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

out_dir = "/nfs/oagrawal/CogVideo/vbench_eval_teacache"

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
        f"CogVideo TeaCache Pareto: {metric.upper()} vs Speedup (Top-Right is Better)",
        fontsize=14, pad=20,
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    if metric == "lpips":
        plt.gca().invert_yaxis()

    out_path = os.path.join(out_dir, f"pareto_frontier_cog_teacache_{metric}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
