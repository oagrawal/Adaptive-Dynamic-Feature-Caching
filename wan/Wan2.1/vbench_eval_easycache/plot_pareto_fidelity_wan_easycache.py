import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Speedups: baseline latency = 216s (Wan 2.1 T2V-1.3B, 50 steps, 832x480, 81f)
# Fixed-mode speedups provided by user; adaptive speedups recovered from
# results_comparison.json (deleted in d6bcb90) via avg_time / 216s baseline.
speedups = {
    "wan_ec_fixed_0.020":          1.73,
    "wan_ec_fixed_0.025":          1.86,
    "wan_ec_fixed_0.040":          2.12,
    "wan_ec_fixed_0.050":          2.28,
    "wan_ec_fixed_0.060":          2.40,
    "wan_ec_fixed_0.070":          2.49,
    "wan_ec_fixed_0.080":          2.57,
    "wan_ec_fixed_0.090":          2.62,
    "wan_ec_fixed_0.125":          2.89,
    "wan_ec_fixed_0.150":          2.98,
    "wan_ec_fixed_0.175":          3.09,
    "wan_ec_fixed_0.200":          3.15,
    "wan_ec_fixed_0.225":          3.25,
    "wan_ec_fixed_0.250":          3.28,
    "wan_ec_adaptive_16_020040":   1.92,
    "wan_ec_adaptive_16_025_020":  2.45,
    "wan_ec_adaptive_16_050_020":  2.68,
    "wan_ec_adaptive_16_075_020":  2.86,
    "wan_ec_adaptive_16_100_020":  2.93,
    "wan_ec_adaptive":             2.10,
}

fidelity_dir = "/nfs/oagrawal/wan/Wan2.1/vbench_eval_easycache/fidelity_metrics"
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

out_dir = "/nfs/oagrawal/wan/Wan2.1/vbench_eval_easycache"

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
        f"Wan 2.1 EasyCache Pareto: {metric.upper()} vs Speedup (Top-Right is Better)",
        fontsize=14, pad=20,
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    if metric == "lpips":
        plt.gca().invert_yaxis()

    out_path = os.path.join(out_dir, f"pareto_frontier_wan_easycache_{metric}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
