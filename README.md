# estimate-maximum-likelihood

A Python port of the R package [`emery`](https://github.com/cdkuempel/emery) (Corie Drake).

Produces maximum likelihood estimates of accuracy statistics — sensitivity, specificity, AUC, and prevalence — for multiple measurement methods when **no gold standard is available** (the *imperfect gold standard* problem).  Estimation uses an expectation-maximization (EM) algorithm based on:

> Zhou, Obuchowski & McClish (2011). *Statistical Methods in Diagnostic Medicine*, 2nd ed.

> Walter (1988). *Biometrics*, 44(3), 601–607.  
> Zhou (2005). *Biometrics*, 61(2), 456–463.  
> Hsieh, Su & Zhou (2011). *Biometrics*, 67(4), 1197–1206.

Supports all three measurement paradigms from the original R package:

| Type | Data | Key statistics |
|---|---|---|
| **binary** | pass/fail (0/1) | sensitivity, specificity, prevalence |
| **ordinal** | Likert-scale integers | CMF, AUC (Wilcoxon), prevalence |
| **continuous** | real-valued | group means/covariances, AUC (normal), prevalence |

---

## Installation

```bash
pip install .
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`

---

## Quick start

```python
import numpy as np
from emery import generate_multimethod_data, estimate_ML, plot_ML, bin_auc

# 1. Simulate binary data from 4 methods (no gold standard needed for estimation)
sim = generate_multimethod_data(
    "binary",
    n_obs=75,
    n_method=4,
    se=[0.87, 0.92, 0.79, 0.95],
    sp=[0.85, 0.93, 0.94, 0.80],
    method_names=["alpha", "beta", "gamma", "delta"],
)

data   = sim["generated_data"]   # shape (75, 4), NaN = missing
params = sim["params"]           # true parameter values

# 2. Estimate sensitivity, specificity, and prevalence
result = estimate_ML("binary", data=data)
print(result)
# MultiMethodMLEstimate(type='binary', iter=22)
#   prev_est: 0.539498
#   se_est: [0.876779, 0.857923, 0.744353, 0.967533]
#   sp_est: [0.724259, 0.962754, 0.945517, 0.830582]

# 3. AUC per method
print(bin_auc(result.results["se_est"], result.results["sp_est"]))

# 4. Diagnostic plots (returns dict of matplotlib Figure objects)
plots = plot_ML(result, params=params)
plots["se_sp"].savefig("se_sp_path.png")
```

---

## Core API

### Data generation

```python
generate_multimethod_data(type, n_method, n_obs, prev, se, sp, ...)
generate_multimethod_binary(n_method, n_obs, prev, se, sp, ...)
```

Simulates paired multi-method measurements from a population with known prevalence.  An optional censoring pattern (`n_method_subset`, `first_reads_all`) can be applied so only a subset of methods produces a result per observation.

### Estimation

```python
result = estimate_ML("binary",     data=data)
result = estimate_ML("ordinal",    data=data)
result = estimate_ML("continuous", data=data)
```

Runs the EM algorithm and returns a `MultiMethodMLEstimate` object.

**Common attributes**

| Attribute | Description |
|---|---|
| `result.results["prev_est"]` | Estimated prevalence |
| `result.iter` | Iterations until convergence |
| `result.prog` | Per-iteration history (when `save_progress=True`) |

**Binary-specific** (`result.results`)

| Key | Description |
|---|---|
| `se_est` | Sensitivity per method |
| `sp_est` | Specificity per method |
| `qk_est` | Posterior P(D=1) per observation |

**Ordinal-specific** (`result.results`)

| Key | Description |
|---|---|
| `A_i_est` | AUC per method (Wilcoxon) |
| `phi_1ij_est` | PMF for positive state (n_level × n_method) |
| `phi_0ij_est` | PMF for negative state (n_level × n_method) |
| `q_k1_est` | Posterior P(D=1) per observation |

**Continuous-specific** (`result.results`)

| Key | Description |
|---|---|
| `mu_i1_est` / `mu_i0_est` | Group mean vectors |
| `sigma_i1_est` / `sigma_i0_est` | Group covariance matrices |
| `eta_j_est` | Standardised mean difference per method |
| `A_j_est` | AUC = Φ(η_j) per method |
| `z_k1_est` | Posterior P(D=1) per observation |

### Initialisation

```python
# Data-driven starting values (default when init is not provided)
init = pollinate_ML("binary", data=data)

# Random starting values (useful for checking local optima)
init = random_start("binary", n_method=4)

# Pass to estimate_ML
result = estimate_ML("binary", data=data, init=init)
```

### Compressed data

Repeated observation patterns can be summarised before estimation for a speed-up on large datasets:

```python
from emery import unique_obs_summary

summary = unique_obs_summary(data)
result = estimate_ML(
    "binary",
    data=summary["unique_obs"],
    freqs=summary["obs_freqs"],
)
```

### Bootstrap confidence intervals

```python
from emery import boot_ML, aggregate_boot_ML
import numpy as np

boot = boot_ML("binary", data=data, n_boot=200, seed=0)
agg  = aggregate_boot_ML(boot)

q10, q50, q90 = np.quantile(agg["se_est"]["values"], [0.10, 0.50, 0.90], axis=0)
```

### Plots

`plot_ML(result, params=sim["params"])` returns a dict of `matplotlib.figure.Figure` objects.  Pass `params` to overlay true values on simulation data.

**Binary** — keys: `prev`, `se`, `sp`, `qk`, `qk_hist`, `se_sp`  
**Ordinal** — keys: `ROC`, `q_k1`, `q_k0`, `q_k1_hist`, `phi_d`  
**Continuous** — keys: `ROC`, `z_k1`, `z_k0`, `z_k1_hist`

### AUC

```python
from emery import bin_auc

auc = bin_auc(se=[0.9, 0.85], sp=[0.88, 0.90])
# AUC = (se + sp) / 2 for a single operating point
```

---

## Package structure

```
emery/
├── __init__.py         # Public API
├── classes.py          # MultiMethodMLEstimate, BootML
├── utils.py            # name_thing, define_disease_state, unique_obs_summary, …
├── binary.py           # Binary EM functions and plotting
├── ordinal.py          # Ordinal EM functions and plotting
├── continuous.py       # Continuous EM functions and plotting
└── core.py             # Dispatch functions (estimate_ML, boot_ML, …)
```

---

## Running tests

```bash
pip install pytest
pytest tests/
```
