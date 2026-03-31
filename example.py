"""
Example script mirroring man/examples/ML_example.R from the emery R package.
"""
import numpy as np
from emery import (
    aggregate_boot_ML,
    bin_auc,
    boot_ML,
    estimate_ML,
    generate_multimethod_data,
    plot_ML,
    unique_obs_summary,
)

# ---------------------------------------------------------------------------
# 1. Generate simulated binary data
# ---------------------------------------------------------------------------
np.random.seed(42)

sim = generate_multimethod_data(
    "binary",
    n_obs=20000,
    n_method=4,
    se=[0.87, 0.92, 0.79, 0.95],
    sp=[0.85, 0.93, 0.94, 0.80],
    method_names=["alpha", "beta", "gamma", "delta"],
)

print("=== Simulated data (first 10 rows) ===")
data = sim["generated_data"]
print(data[:10])
print(f"\nShape: {data.shape}")

params = sim["params"]
print(f"\nTrue prevalence : {params['prev']:.3f}")
print(f"True se         : {params['se']}")
print(f"True sp         : {params['sp']}")

# ---------------------------------------------------------------------------
# 2. Estimate ML accuracy values
# ---------------------------------------------------------------------------
result = estimate_ML("binary", data=data, save_progress=True)

print("\n=== ML Estimates ===")
print(result)

print(f"\nAUC per method: {bin_auc(result.results['se_est'], result.results['sp_est'])}")

# ---------------------------------------------------------------------------
# 3. Using unique_obs_summary to compress repeated rows
# ---------------------------------------------------------------------------
summary = unique_obs_summary(data)
print(f"\nUnique observation patterns : {len(summary['unique_obs'])}")
print(f"Frequencies (first 5)       : {summary['obs_freqs'][:5]}")

result_compressed = estimate_ML(
    "binary",
    data=summary["unique_obs"],
    freqs=summary["obs_freqs"],
    save_progress=False,
)
print("\n=== Estimates from compressed data (should match) ===")
print(result_compressed)

# ---------------------------------------------------------------------------
# 4. Plots (saved to files)
# ---------------------------------------------------------------------------
plots = plot_ML(result, params=params)

for name, fig in plots.items():
    fname = f"plot_{name}.png"
    fig.savefig(fname, dpi=100, bbox_inches="tight")
    print(f"Saved {fname}")

import matplotlib.pyplot as plt
plt.close("all")

# ---------------------------------------------------------------------------
# 5. Bootstrap confidence intervals
# ---------------------------------------------------------------------------
print("\n=== Bootstrap (100 replicates) ===")
boot = boot_ML("binary", data=data, n_boot=100, seed=0)
agg = aggregate_boot_ML(boot)

for stat in ("prev_est", "se_est", "sp_est"):
    vals = agg[stat]["values"]
    q10, q50, q90 = np.quantile(vals, [0.10, 0.50, 0.90], axis=0)
    print(f"\n{stat}")
    print(f"  10th pct: {q10}")
    print(f"  Median  : {q50}")
    print(f"  90th pct: {q90}")
