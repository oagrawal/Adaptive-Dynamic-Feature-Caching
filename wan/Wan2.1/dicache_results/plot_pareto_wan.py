import pandas as pd
import matplotlib.pyplot as plt
import os

# Data provided by the user for Wan2.1 + DiCache
data = [
    {"mode": "wan_dc_baseline", "latency": 224, "vbench": 0.8306},
    {"mode": "wan_dc_fixed_0.05", "latency": 126, "vbench": 0.8150},
    {"mode": "wan_dc_fixed_0.10", "latency": 87, "vbench": 0.8007},
    {"mode": "wan_dc_fixed_0.15", "latency": 71, "vbench": 0.8081},
    {"mode": "wan_dc_fixed_0.20", "latency": 61, "vbench": 0.8036},
    {"mode": "wan_dc_fixed_0.225", "latency": 56, "vbench": 0.7687},
    {"mode": "wan_dc_fixed_0.25", "latency": 55, "vbench": 0.7331},
    {"mode": "wan_dc_adaptive_hi0.225_lo0.05", "latency": 98, "vbench": 0.8164},
    {"mode": "wan_dc_adaptive_hi0.225_lo0.10", "latency": 77, "vbench": 0.7837},
    {"mode": "wan_dc_adaptive_hi0.225_lo0.15", "latency": 67, "vbench": 0.8030},
    {"mode": "wan_dc_adaptive_hi0.25_lo0.05", "latency": 98, "vbench": 0.8107},
    {"mode": "wan_dc_adaptive_hi0.25_lo0.10", "latency": 76, "vbench": 0.7643},
    {"mode": "wan_dc_adaptive_hi0.25_lo0.15", "latency": 66, "vbench": 0.7887},
]

df = pd.DataFrame(data)

# Separate baseline/fixed from adaptive
df["is_adaptive"] = df["mode"].str.contains("adaptive")

# Define colors
colors = {True: "#FF6F61", False: "#6B5B95"} # Adaptive: Coral/Red-ish, Fixed/Baseline: Purple/Blue-ish
labels = {True: "Adaptive Modes", False: "Baseline & Fixed Modes"}

plt.figure(figsize=(10, 6))

# Plot the connecting line for Baseline & Fixed Modes
fixed_group = df[~df["is_adaptive"]].sort_values("latency")
plt.plot(fixed_group["latency"], fixed_group["vbench"], 
         linestyle="-", color="#6B5B95", alpha=0.5, label="Fixed Threshold Curve", zorder=1)

# Scatter points
for is_adapt, group in df.groupby("is_adaptive"):
    plt.scatter(group["latency"], group["vbench"], 
                c=colors[is_adapt], label=labels[is_adapt], s=120, edgecolors="black", alpha=0.8, zorder=3)

plt.xlabel("Latency (seconds)", fontsize=12)
plt.ylabel("VBench Score (Aggregated)", fontsize=12)
plt.title("Wan2.1 DiCache Pareto Frontier: Quality vs. Latency", fontsize=14, pad=20)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower left")

# Save the plot
output_path = "/nfs/oagrawal/wan/Wan2.1/dicache_results/pareto_frontier_wan.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output_path}")
