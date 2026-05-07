import json
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

HERE     = Path(__file__).resolve().parent
VIDEOS   = HERE / "videos"
FIDELITY = HERE / "fidelity_metrics"

# ---------------------------------------------------------------------------
# Load timing from generation log JSONs
# ---------------------------------------------------------------------------
times = defaultdict(list)
for log in glob.glob(str(VIDEOS / "generation_log_*.json")):
    for r in json.loads(Path(log).read_text())["runs"]:
        if "time_seconds" in r:
            times[r["mode"]].append(r["time_seconds"])

if "wan_baseline" not in times:
    raise RuntimeError("No baseline timing found in generation logs — run baseline generation first.")

baseline_time = sum(times["wan_baseline"]) / len(times["wan_baseline"])
print(f"Baseline (T2V-1.3B): {baseline_time:.1f}s  (n={len(times['wan_baseline'])})")

# ---------------------------------------------------------------------------
# Build data points from fidelity JSONs
# ---------------------------------------------------------------------------
data = []
for fid_path in sorted(FIDELITY.glob("*_vs_wan_baseline.json")):
    mode = fid_path.stem.replace("_vs_wan_baseline", "")
    if mode not in times:
        print(f"WARNING: no timing data for {mode}")
        continue
    fid = json.loads(fid_path.read_text())
    avg_time = sum(times[mode]) / len(times[mode])
    speedup  = baseline_time / avg_time
    data.append({
        "mode":        mode,
        "speedup":     speedup,
        "psnr":        fid["psnr"]["mean"],
        "ssim":        fid["ssim"]["mean"],
        "lpips":       fid["lpips"]["mean"],
        "is_adaptive": "adaptive" in mode,
    })

df = pd.DataFrame(data)
print(f"\nPlotting {len(df)} points:")
print(df[["mode", "speedup", "psnr", "ssim", "lpips"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {
    "fixed":    "#6B5B95",   # purple
    "adaptive": "#FF6F61",   # coral
}
MARKERS = {
    "fixed":    "o",
    "adaptive": "^",
}

def short_label(mode):
    return (mode
        .replace("wan_tc_adaptive_", "adap ")
        .replace("wan_tc_fixed_",    "fixed ")
        .replace("wan_adaptive",     "adap (orig)")
        .replace("wan_fixed_",       "fixed ")
        .replace("hi", "h")
        .replace("lo", "l"))

metrics = ["psnr", "ssim", "lpips"]
metric_titles = {
    "psnr":  "PSNR (dB) — Higher is Better",
    "ssim":  "SSIM — Higher is Better",
    "lpips": "LPIPS — Lower is Better (Inverted Axis)",
}

for metric in metrics:
    fig, ax = plt.subplots(figsize=(11, 6))

    # Connect fixed-threshold points with a curve
    fixed = df[~df["is_adaptive"]].sort_values("speedup")
    if not fixed.empty:
        ax.plot(fixed["speedup"], fixed[metric],
                linestyle="-", color="#6B5B95", alpha=0.4, zorder=1)

    # Scatter all points
    seen_labels = set()
    for _, row in df.iterrows():
        kind = "adaptive" if row["is_adaptive"] else "fixed"
        c    = COLORS[kind]
        lbl  = kind.capitalize() if kind not in seen_labels else "_nolegend_"
        seen_labels.add(kind)
        ax.scatter(row["speedup"], row[metric],
                   c=c, s=130, edgecolors="black", alpha=0.9,
                   zorder=3, label=lbl, marker=MARKERS[kind])

    ax.set_xlabel("Speedup (×)", fontsize=12)
    ax.set_ylabel(metric_titles[metric], fontsize=12)
    ax.set_title(f"Wan2.1 T2V-1.3B TeaCache — {metric.upper()} vs Speedup", fontsize=14, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    if metric == "lpips":
        ax.invert_yaxis()

    out = HERE / f"pareto_frontier_wan_teacache_{metric}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
