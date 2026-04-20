import os
import json

latencies = {
    "wan_dc_baseline": ("224s", "1.00x"),
    "wan_dc_fixed_0.05": ("126s", "1.78x"),
    "wan_dc_fixed_0.10": ("87s", "2.58x"),
    "wan_dc_fixed_0.15": ("71s", "3.16x"),
    "wan_dc_fixed_0.20": ("61s", "3.66x"),
    "wan_dc_fixed_0.225": ("56s", "3.97x"),
    "wan_dc_fixed_0.25": ("55s", "4.03x"),
    "wan_dc_adaptive_hi0.225_lo0.05": ("98s", "2.28x"),
    "wan_dc_adaptive_hi0.225_lo0.10": ("77s", "2.89x"),
    "wan_dc_adaptive_hi0.225_lo0.15": ("67s", "3.33x"),
    "wan_dc_adaptive_hi0.25_lo0.05": ("98s", "2.29x"),
    "wan_dc_adaptive_hi0.25_lo0.10": ("76s", "2.94x"),
    "wan_dc_adaptive_hi0.25_lo0.15": ("66s", "3.39x"),
}

fidelity_dir = "/nfs/oagrawal/wan/Wan2.1/dicache_results/fidelity_metrics"
results = []

def sort_key(mode):
    if "baseline" in mode: return (0, 0)
    if "fixed" in mode: 
        try: return (1, float(mode.split("_")[-1]))
        except: return (1, 0)
    return (2, mode)

for mode in sorted(latencies.keys(), key=sort_key):
    lat, speedup = latencies[mode]
    if mode == "wan_dc_baseline":
        results.append((mode, lat, speedup, "Inf", "1.0000", "0.0000"))
        continue
    
    path = os.path.join(fidelity_dir, mode + "_vs_wan_dc_baseline.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            js = json.load(f)
            psnr = "%.4f" % js["psnr"]["mean"]
            ssim = "%.4f" % js["ssim"]["mean"]
            lpips = "%.4f" % js["lpips"]["mean"]
            results.append((mode, lat, speedup, psnr, ssim, lpips))
    else:
        results.append((mode, lat, speedup, "N/A", "N/A", "N/A"))

print("| Mode | Latency | Speedup | PSNR | SSIM | LPIPS |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")
for r in results:
    print("| " + " | ".join(r) + " |")
