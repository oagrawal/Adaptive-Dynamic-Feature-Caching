# HunyuanVideo + DiCache Latency and Fidelity Metrics

Latency is seconds per video. PSNR/SSIM are higher-is-better; LPIPS is lower-is-better. All rows use 33 videos.

| mode | policy | low_thresh | high/fixed_thresh | threshold steps | latency_s | speedup_x | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| dicache_baseline | baseline |  |  |  | 1265.0 | 1.00 | 100.0000 | 1.0000 | 0.0000 |
| dicache_fixed_0.05 | fixed |  | 0.05 | all steps | 658.9 | 1.92 | 30.9574 | 0.9037 | 0.0875 |
| dicache_fixed_0.07 | fixed |  | 0.07 | all steps | 627.5 | 2.02 | 27.9486 | 0.8584 | 0.1396 |
| dicache_adaptive_0.05_0.10 | adaptive | 0.05 | 0.10 | high: 8-39; low: <8, >=40 | 619.6 | 2.04 | 29.4186 | 0.8881 | 0.1052 |
| dicache_fixed_0.08 | fixed |  | 0.08 | all steps | 570.6 | 2.22 | 20.2057 | 0.7290 | 0.3221 |
| dicache_adaptive_0.05_0.15 | adaptive | 0.05 | 0.15 | high: 8-39; low: <8, >=40 | 567.9 | 2.23 | 28.1893 | 0.8742 | 0.1228 |
| dicache_adaptive_0.05_0.30 | adaptive | 0.05 | 0.30 | high: 8-39; low: <8, >=40 | 529.2 | 2.39 | 26.1520 | 0.8435 | 0.1694 |
| dicache_adaptive_0.05_0.25 | adaptive | 0.05 | 0.25 | high: 8-39; low: <8, >=40 | 528.3 | 2.39 | 26.6643 | 0.8521 | 0.1550 |
| dicache_adaptive_0.05_0.35 | adaptive | 0.05 | 0.35 | high: 8-39; low: <8, >=40 | 505.8 | 2.50 | 25.5513 | 0.8316 | 0.1953 |
| dicache_adaptive_0.05_0.40 | adaptive | 0.05 | 0.40 | high: 8-39; low: <8, >=40 | 502.4 | 2.52 | 25.1963 | 0.8244 | 0.2061 |
| dicache_adaptive_0.05_0.20 | adaptive | 0.05 | 0.20 | low: first 7 + last 10; high: rest | 465.1 | 2.72 | 27.0806 | 0.8586 | 0.1442 |
| dicache_fixed_0.10 | fixed |  | 0.10 | all steps | 418.9 | 3.02 | 20.1463 | 0.7197 | 0.3279 |
| dicache_fixed_0.15 | fixed |  | 0.15 | all steps | 346.6 | 3.65 | 20.1790 | 0.7194 | 0.3298 |
| dicache_adaptive_0.10_0.30 | adaptive | 0.10 | 0.30 | low: first 7 + last 10; high: rest | 326.9 | 3.87 | 20.2306 | 0.7236 | 0.3375 |
| dicache_fixed_0.20 | fixed |  | 0.20 | all steps | 286.2 | 4.42 | 20.1070 | 0.7159 | 0.3401 |
| dicache_adaptive_0.15_0.40 | adaptive | 0.15 | 0.40 | low: first 7 + last 10; high: rest | 275.0 | 4.60 | 19.8306 | 0.7068 | 0.3801 |
| dicache_fixed_0.25 | fixed |  | 0.25 | all steps | 250.5 | 5.05 | 19.7915 | 0.7044 | 0.3635 |
| dicache_fixed_0.30 | fixed |  | 0.30 | all steps | 226.3 | 5.59 | 18.9684 | 0.6864 | 0.4103 |
| dicache_fixed_0.35 | fixed |  | 0.35 | all steps | 202.4 | 6.25 | 18.7283 | 0.6804 | 0.4315 |
| dicache_fixed_0.40 | fixed |  | 0.40 | all steps | 186.3 | 6.79 | 17.8538 | 0.6599 | 0.4877 |
| dicache_fixed_0.60 | fixed |  | 0.60 | all steps | 164.5 | 7.69 | 16.2992 | 0.6200 | 0.6406 |

Notes:
- All adaptive modes use `ret_ratio=0.0` and schedule: low threshold for `step < 8` and `step >= 40`, high threshold for `step 8–39`.
- `dicache_fixed_0.08` hits the quality cliff (PSNR drops from ~28 to ~20); `dicache_adaptive_0.05_0.10` bridges the cliff by keeping the high threshold below it.
- `dicache_adaptive_0.05_0.10` and `dicache_adaptive_0.05_0.15` are both Pareto-optimal new additions.
