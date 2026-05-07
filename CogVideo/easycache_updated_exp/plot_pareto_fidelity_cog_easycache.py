import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Speedups from vbench_scores_table.csv; baseline mean = 1010.6s.
# cog_ec_fixed_0.025 excluded — threshold so low no steps are cached (speedup=1.00x, PSNR=100).
speedups = {
    "cog_ec_fixed_0.05":                      1.61,
    "cog_ec_fixed_0.075":                     2.05,
    "cog_ec_fixed_0.10":                      2.32,
    "cog_ec_fixed_0.125":                     2.72,
    "cog_ec_fixed_0.15":                      3.03,
    "cog_ec_fixed_0.20":                      3.43,
    "cog_ec_adaptive_hi0.10_lo0.075_f9_l6":   2.27,
    "cog_ec_adaptive_hi0.10_lo0.075_f13_l8":  2.21,
    "cog_ec_adaptive_hi0.125_lo0.075_f9_l6":  2.55,
    "cog_ec_adaptive_hi0.125_lo0.075_f13_l8": 2.39,
    "cog_ec_adaptive_hi0.15_lo0.075_f9_l6":   2.72,
    "cog_ec_adaptive_hi0.15_lo0.075_f13_l8":  2.56,
    "cog_ec_adaptive_hi0.20_lo0.075_f9_l6":   2.84,
    "cog_ec_adaptive_hi0.20_lo0.075_f13_l8":  2.65,
}

fidelity_dir = "/nfs/oagrawal/CogVideo/easycache_updated_exp/fidelity_metrics"
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

out_dir = "/nfs/oagrawal/CogVideo/easycache_updated_exp"

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
        f"CogVideo EasyCache Pareto: {metric.upper()} vs Speedup (Top-Right is Better)",
        fontsize=14, pad=20,
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    if metric == "lpips":
        plt.gca().invert_yaxis()

    out_path = os.path.join(out_dir, f"pareto_frontier_cog_easycache_{metric}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
