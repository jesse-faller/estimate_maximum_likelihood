"""
Continuous-method functions for emery.

Implements EM-based maximum likelihood estimation of AUC, means, and
covariance matrices for continuous measurements from multiple methods when
no gold standard is available.

Reference: Hsieh, Su & Zhou (2011), *Biometrics*, 67(4), 1197–1206.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm

from .classes import MultiMethodMLEstimate
from .utils import censor_data, define_disease_state, name_thing


# ---------------------------------------------------------------------------
# Multivariate normal density
# ---------------------------------------------------------------------------


def _dmvnorm(x: np.ndarray, mu, sigma: np.ndarray) -> np.ndarray:
    """
    Multivariate normal density for each row of *x*.

    Mirrors the R utility ``dmvnorm()`` in emery.  Rows containing NaN
    receive a density of NaN (continuous EM assumes complete data).

    Parameters
    ----------
    x : ndarray, shape (n_obs, n_method)
    mu : array-like, length n_method
    sigma : ndarray, shape (n_method, n_method)

    Returns
    -------
    ndarray, shape (n_obs,)
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float)
    n_obs, k = x.shape

    try:
        sigma_inv = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)
        if det_sigma <= 0:
            return np.full(n_obs, 1e-300)
    except np.linalg.LinAlgError:
        return np.full(n_obs, 1e-300)

    norm_const = 1.0 / np.sqrt((2.0 * np.pi) ** k * det_sigma)
    x_c = x - mu                             # (n_obs, k)
    mah = np.sum((x_c @ sigma_inv) * x_c, axis=1)   # (n_obs,)
    return norm_const * np.exp(-0.5 * mah)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_multimethod_continuous(
    n_method: int = 2,
    n_obs: int = 100,
    prev: float = 0.5,
    D=None,
    mu_i1=None,
    sigma_i1=None,
    mu_i0=None,
    sigma_i0=None,
    method_names=None,
    obs_names=None,
    n_method_subset: Optional[int] = None,
    first_reads_all: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate continuous measurement data from multiple methods.

    Positive and negative observations are drawn from multivariate normal
    distributions with separate means and covariances.

    Parameters
    ----------
    n_method : int
        Number of measurement methods.
    n_obs : int
        Number of observations.
    prev : float
        Disease prevalence.
    D : array-like or None
        True disease-state vector (1 = positive, 0 = negative).
    mu_i1 : array-like, length n_method, or None
        Mean vector for positive observations.  Defaults to all-12.
    sigma_i1 : ndarray, shape (n_method, n_method), or None
        Covariance for positive observations.  Defaults to identity.
    mu_i0 : array-like or None
        Mean vector for negative observations.  Defaults to all-10.
    sigma_i0 : ndarray or None
        Covariance for negative observations.  Defaults to identity.
    method_names, obs_names : list of str or None
        Optional names.
    n_method_subset : int or None
        Number of methods observed per observation.
    first_reads_all : bool
        If ``True``, method 0 always has a result.
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys ``generated_data`` and ``params``.
    """
    if seed is not None:
        np.random.seed(seed)

    if n_method_subset is None:
        n_method_subset = n_method

    if method_names is None:
        method_names = name_thing("method", n_method)
    if obs_names is None:
        obs_names = name_thing("obs", n_obs)

    if mu_i1 is None:
        mu_i1 = np.full(n_method, 12.0)
    if mu_i0 is None:
        mu_i0 = np.full(n_method, 10.0)
    if sigma_i1 is None:
        sigma_i1 = np.eye(n_method)
    if sigma_i0 is None:
        sigma_i0 = np.eye(n_method)

    mu_i1 = np.asarray(mu_i1, dtype=float)
    mu_i0 = np.asarray(mu_i0, dtype=float)
    sigma_i1 = np.asarray(sigma_i1, dtype=float)
    sigma_i0 = np.asarray(sigma_i0, dtype=float)

    dis = define_disease_state(D=D, n_obs=n_obs, prev=prev)
    n_obs = dis["n_obs"]

    X = np.random.multivariate_normal(mu_i1, sigma_i1, size=dis["pos"])
    Y = np.random.multivariate_normal(mu_i0, sigma_i0, size=dis["neg"])

    subset_matrix = censor_data(
        n_obs=n_obs,
        first_reads_all=first_reads_all,
        n_method_subset=n_method_subset,
        n_method=n_method,
    )

    generated_data = np.vstack([X, Y]) * subset_matrix

    params = {
        "n_method": n_method,
        "n_obs": n_obs,
        "prev": dis["prev"],
        "D": dict(zip(obs_names, dis["D"])),
        "mu_i1": mu_i1,
        "sigma_i1": sigma_i1,
        "mu_i0": mu_i0,
        "sigma_i0": sigma_i0,
        "method_names": method_names,
        "obs_names": obs_names,
    }

    return {"generated_data": generated_data, "params": params}


# ---------------------------------------------------------------------------
# EM initialisation
# ---------------------------------------------------------------------------


def pollinate_ML_continuous(
    data,
    freqs=None,
    prev: float = 0.5,
    q_seeds=None,
    high_pos: bool = True,
    **kwargs,
) -> dict:
    """
    Generate quantile-based starting values for the continuous EM algorithm.

    Parameters
    ----------
    data : array-like, shape (n_obs, n_method)
        Observed continuous measurements (NaN for missing).
    freqs : array-like or None
        Accepted for API consistency; not used.
    prev : float
        Approximate prevalence, used to choose default quantile seeds.
    q_seeds : sequence of two floats or None
        Quantiles at which to seed the two group means.  Defaults to
        ``[(1-prev)/2, 1 - prev/2]``.
    high_pos : bool
        If ``True`` (default), higher values indicate positive disease state,
        so the high-quantile seed is used for the positive group mean.

    Returns
    -------
    dict with keys ``prev_1``, ``mu_i1_1``, ``sigma_i1_1``,
    ``mu_i0_1``, ``sigma_i0_1``.
    """
    data = np.asarray(data, dtype=float)

    if q_seeds is None:
        q_seeds = [(1.0 - prev) / 2.0, 1.0 - prev / 2.0]

    q_seeds = sorted(q_seeds, reverse=high_pos)  # descending when high_pos

    mu_mat = np.nanquantile(data, q_seeds, axis=0)   # (2, n_method)
    mu_i1_1 = mu_mat[0]
    mu_i0_1 = mu_mat[1]

    # Initial covariance: diagonal with half the mean as variance
    sigma_i1_1 = np.diag(np.abs(mu_i1_1) / 2.0)
    sigma_i0_1 = np.diag(np.abs(mu_i0_1) / 2.0)

    return {
        "prev_1": prev,
        "mu_i1_1": mu_i1_1,
        "sigma_i1_1": sigma_i1_1,
        "mu_i0_1": mu_i0_1,
        "sigma_i0_1": sigma_i0_1,
    }


# ---------------------------------------------------------------------------
# EM algorithm
# ---------------------------------------------------------------------------


def estimate_ML_continuous(
    data,
    freqs=None,
    init: Optional[dict] = None,
    max_iter: int = 100,
    tol: float = 1e-7,
    save_progress: bool = True,
) -> MultiMethodMLEstimate:
    """
    Estimate AUC, group means, and covariances for continuous multi-method
    data via EM algorithm.

    .. note::
       The continuous EM assumes approximately complete data.  The
       ``freqs`` parameter is accepted for API consistency but is **not**
       used in the EM iterations.

    Parameters
    ----------
    data : array-like, shape (n_obs, n_method)
        Observed continuous measurements (NaN for missing).
    freqs : array-like or None
        Accepted but not used in the continuous EM.
    init : dict or None
        Initial values with keys ``prev_1``, ``mu_i1_1``, ``sigma_i1_1``,
        ``mu_i0_1``, ``sigma_i0_1``.  Auto-generated via
        ``pollinate_ML_continuous`` when ``None``.
    max_iter : int
        Maximum EM iterations (default 100).
    tol : float
        Convergence tolerance on the conditional log-likelihood change.
    save_progress : bool
        Store per-iteration diagnostics.

    Returns
    -------
    MultiMethodMLEstimate
        ``results`` keys:

        * ``prev_est``     – estimated prevalence
        * ``mu_i1_est``    – estimated positive-group means (length n_method)
        * ``sigma_i1_est`` – estimated positive-group covariance matrix
        * ``mu_i0_est``    – estimated negative-group means
        * ``sigma_i0_est`` – estimated negative-group covariance matrix
        * ``eta_j_est``    – standardised mean difference per method
        * ``A_j_est``      – AUC = Φ(η_j) per method
        * ``z_k1_est``     – P(D=1 | data_k) per observation
        * ``z_k0_est``     – P(D=0 | data_k) per observation
    """
    # ------------------------------------------------------------------
    # Coerce input
    # ------------------------------------------------------------------
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            _method_names = list(data.columns)
            _obs_names = list(data.index.astype(str))
            data = data.values.astype(float)
        else:
            _method_names = None
            _obs_names = None
    except ImportError:
        _method_names = None
        _obs_names = None

    data = np.asarray(data, dtype=float)
    n_obs, n_method = data.shape

    method_names = _method_names or name_thing("method", n_method)
    obs_names = _obs_names or name_thing("obs", n_obs)

    if freqs is None:
        freqs = np.ones(n_obs)
    freqs = np.asarray(freqs, dtype=float)

    not_missing = ~np.isnan(data)   # (n_obs, n_method)

    # ------------------------------------------------------------------
    # Initial values
    # ------------------------------------------------------------------
    _need_init = (
        init is None
        or not isinstance(init, dict)
        or not all(k in init for k in
                   ("prev_1", "mu_i1_1", "sigma_i1_1", "mu_i0_1", "sigma_i0_1"))
        or any(init.get(k) is None
               for k in ("prev_1", "mu_i1_1", "sigma_i1_1", "mu_i0_1", "sigma_i0_1"))
    )
    if _need_init:
        init = pollinate_ML_continuous(data)

    prev_m = float(init["prev_1"])
    mu_i1_m = np.asarray(init["mu_i1_1"], dtype=float).flatten()
    sigma_i1_m = np.asarray(init["sigma_i1_1"], dtype=float)
    mu_i0_m = np.asarray(init["mu_i0_1"], dtype=float).flatten()
    sigma_i0_m = np.asarray(init["sigma_i0_1"], dtype=float)

    # ------------------------------------------------------------------
    # Inner helpers
    # ------------------------------------------------------------------

    def _calc_z_kd(d, f_X, f_Y):
        num = (prev_m * f_X) * d + ((1.0 - prev_m) * f_Y) * (1.0 - d)
        denom = prev_m * f_X + (1.0 - prev_m) * f_Y
        return num / np.maximum(denom, 1e-300)

    def _calc_l_cond(z_k1, z_k0, f_X, f_Y):
        return float(np.nansum(
            z_k1 * np.log(np.maximum(prev_m * f_X, 1e-300))
            + z_k0 * np.log(np.maximum((1.0 - prev_m) * f_Y, 1e-300))
        ))

    def _calc_next_prev(z_k1):
        return float(np.mean(z_k1))

    def _calc_next_mu(z_kd):
        num = np.nansum(z_kd[:, np.newaxis] * data, axis=0)   # (n_method,)
        denom = np.sum(z_kd[:, np.newaxis] * not_missing, axis=0)
        return num / np.maximum(denom, 1e-300)

    def _calc_next_sigma(z_kd, mu_id):
        sigma = np.zeros((n_method, n_method))
        for i in range(n_method):
            for j in range(i + 1):
                mask = not_missing[:, i] & not_missing[:, j]
                if np.any(mask):
                    diff_i = data[mask, i] - mu_id[i]
                    diff_j = data[mask, j] - mu_id[j]
                    num = np.sum(z_kd[mask] * diff_i * diff_j)
                else:
                    num = 0.0
                denom = np.sum(z_kd * not_missing[:, i] * not_missing[:, j])
                sigma[i, j] = num / max(denom, 1e-300)
                sigma[j, i] = sigma[i, j]
        # Ensure positive-definiteness via small diagonal nudge if needed
        min_eig = np.min(np.linalg.eigvalsh(sigma))
        if min_eig <= 0:
            sigma += np.eye(n_method) * (abs(min_eig) + 1e-6)
        return sigma

    def _calc_eta_j(mu1, mu0, sig1, sig0):
        return (mu1 - mu0) / np.sqrt(np.diag(sig1) + np.diag(sig0))

    # ------------------------------------------------------------------
    # EM iterations
    # ------------------------------------------------------------------
    hist_prev: list = []
    hist_mu_i1: list = []
    hist_sigma_i1: list = []
    hist_mu_i0: list = []
    hist_sigma_i0: list = []
    hist_eta_j: list = []
    hist_A_j: list = []
    hist_z_k1: list = []
    hist_z_k0: list = []
    hist_l_cond: list = []

    final_iter = max_iter

    for it in range(1, max_iter + 1):
        f_X_m = np.maximum(_dmvnorm(data, mu_i1_m, sigma_i1_m), 1e-300)
        f_Y_m = np.maximum(_dmvnorm(data, mu_i0_m, sigma_i0_m), 1e-300)

        eta_j_m = _calc_eta_j(mu_i1_m, mu_i0_m, sigma_i1_m, sigma_i0_m)
        A_j_m = norm.cdf(eta_j_m)
        z_k1_m = _calc_z_kd(1, f_X_m, f_Y_m)
        z_k0_m = _calc_z_kd(0, f_X_m, f_Y_m)
        l_cond_m = _calc_l_cond(z_k1_m, z_k0_m, f_X_m, f_Y_m)

        hist_prev.append(prev_m)
        hist_mu_i1.append(mu_i1_m.copy())
        hist_sigma_i1.append(sigma_i1_m.copy())
        hist_mu_i0.append(mu_i0_m.copy())
        hist_sigma_i0.append(sigma_i0_m.copy())
        hist_eta_j.append(eta_j_m.copy())
        hist_A_j.append(A_j_m.copy())
        hist_z_k1.append(z_k1_m.copy())
        hist_z_k0.append(z_k0_m.copy())
        hist_l_cond.append(l_cond_m)

        if it > 1 and abs(hist_l_cond[-1] - hist_l_cond[-2]) < tol:
            final_iter = it
            break

        prev_m = _calc_next_prev(z_k1_m)
        mu_i1_m = _calc_next_mu(z_k1_m)
        sigma_i1_m = _calc_next_sigma(z_k1_m, mu_i1_m)
        mu_i0_m = _calc_next_mu(z_k0_m)
        sigma_i0_m = _calc_next_sigma(z_k0_m, mu_i0_m)
    else:
        final_iter = max_iter

    results = {
        "prev_est": prev_m,
        "mu_i1_est": mu_i1_m,
        "sigma_i1_est": sigma_i1_m,
        "mu_i0_est": mu_i0_m,
        "sigma_i0_est": sigma_i0_m,
        "eta_j_est": eta_j_m,
        "A_j_est": A_j_m,
        "z_k1_est": z_k1_m,
        "z_k0_est": z_k0_m,
    }

    out = MultiMethodMLEstimate(
        results=results,
        data=data,
        freqs=freqs,
        names={"method_names": method_names, "obs_names": obs_names},
        iter=final_iter,
        type="continuous",
    )

    if save_progress:
        out.prog = {
            "prev": np.array(hist_prev),
            "mu_i1": np.vstack(hist_mu_i1),       # (iter, n_method)
            "mu_i0": np.vstack(hist_mu_i0),
            "sigma_i1": hist_sigma_i1,             # list of (n_method, n_method)
            "sigma_i0": hist_sigma_i0,
            "eta_j": np.vstack(hist_eta_j),
            "A_j": np.vstack(hist_A_j),
            "z_k1": np.vstack(hist_z_k1),          # (iter, n_obs)
            "z_k0": np.vstack(hist_z_k0),
            "l_cond": np.array(hist_l_cond),
        }

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ML_continuous(
    ML_est: MultiMethodMLEstimate,
    params: Optional[dict] = None,
) -> dict:
    """
    Create diagnostic plots for a continuous EM estimation result.

    Parameters
    ----------
    ML_est : MultiMethodMLEstimate
        Result from ``estimate_ML_continuous`` (requires ``save_progress=True``
        for progress plots).
    params : dict or None
        Optional ground-truth parameters.  ``params["D"]`` colours traces.

    Returns
    -------
    dict of matplotlib Figure objects with keys:
    ``ROC``, ``z_k1``, ``z_k0``, ``z_k1_hist``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    if params is None:
        params = {}

    method_names = ML_est.names.get("method_names", [])
    obs_names = ML_est.names.get("obs_names", [])
    n_method = len(method_names)
    n_obs = len(obs_names)
    n_iter = ML_est.iter

    _colours = [
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
        "#66A61E", "#E6AB02", "#A6761D", "#666666",
    ]
    method_colours = [_colours[i % len(_colours)] for i in range(n_method)]

    true_D = params.get("D")
    if true_D is not None:
        if isinstance(true_D, dict):
            true_D_arr = np.array([true_D.get(n, np.nan) for n in obs_names])
        else:
            true_D_arr = np.asarray(list(true_D), dtype=float)
    else:
        true_D_arr = np.full(n_obs, np.nan)

    # ------------------------------------------------------------------
    # 1. Empirical ROC curves
    # ------------------------------------------------------------------
    data = ML_est.data
    z_k1 = ML_est.results["z_k1_est"]
    z_k0 = ML_est.results["z_k0_est"]
    A_j = ML_est.results["A_j_est"]

    fig_roc, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)

    auc_lines = []
    for i, (mname, colour) in enumerate(zip(method_names, method_colours)):
        col = data[:, i]
        not_na = ~np.isnan(col)
        col_valid = col[not_na]
        z1_valid = z_k1[not_na]
        z0_valid = z_k0[not_na]

        z1_total = np.sum(z1_valid)
        z0_total = np.sum(z0_valid)

        # Sort descending; cumulative sum gives P(T >= threshold)
        sort_idx = np.argsort(col_valid)[::-1]
        tpr_curve = np.concatenate([[0], np.cumsum(z1_valid[sort_idx]) / z1_total])
        fpr_curve = np.concatenate([[0], np.cumsum(z0_valid[sort_idx]) / z0_total])

        ax.plot(fpr_curve, tpr_curve, color=colour, alpha=0.8, linewidth=1.2)
        auc_lines.append(f"{mname}: {A_j[i]:.3f}")

    auc_text = "AUC\n" + "\n".join(auc_lines)
    ax.text(0.65, 0.05, auc_text, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    handles = [mpatches.Patch(color=c, label=m)
               for m, c in zip(method_names, method_colours)]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("ROC Curves (empirical, weighted by z_k)")
    fig_roc.tight_layout()

    # ------------------------------------------------------------------
    # 2 & 3. z_k1 and z_k0 progress plots
    # ------------------------------------------------------------------
    def _progress_plot(prog_key, ylabel):
        fig, ax = plt.subplots(figsize=(7, 4))
        if not ML_est.prog:
            ax.text(0.5, 0.5, "No progress data\n(re-run with save_progress=True)",
                    ha="center", va="center", transform=ax.transAxes)
            return fig
        arr = ML_est.prog[prog_key]   # (n_iter, n_obs)
        iters = np.arange(1, arr.shape[0] + 1)
        for k in range(n_obs):
            d = true_D_arr[k]
            colour = "gray" if np.isnan(d) else (_colours[0] if d == 1 else _colours[1])
            ax.plot(iters, arr[:, k], color=colour, alpha=0.4, linewidth=0.7)
        handles = []
        if not np.all(np.isnan(true_D_arr)):
            handles.append(mlines.Line2D([], [], color=_colours[0], label="Class 1"))
            handles.append(mlines.Line2D([], [], color=_colours[1], label="Class 0"))
        else:
            handles.append(mlines.Line2D([], [], color="gray", label="Unknown"))
        ax.legend(handles=handles)
        ax.set_xlim(1, max(n_iter, 2)); ax.set_ylim(0, 1)
        ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
        ax.grid(color="gray", alpha=0.3)
        ax.set_title(f"{ylabel} over iterations")
        fig.tight_layout()
        return fig

    fig_zk1 = _progress_plot("z_k1", "z_k1")
    fig_zk0 = _progress_plot("z_k0", "z_k0")

    # ------------------------------------------------------------------
    # 4. z_k1 histogram
    # ------------------------------------------------------------------
    fig_zk1_hist, ax = plt.subplots(figsize=(7, 4))
    zk1_final = ML_est.results["z_k1_est"]
    if not np.all(np.isnan(true_D_arr)):
        for d_val, label, colour in [(1.0, "Class 1", _colours[0]),
                                     (0.0, "Class 0", _colours[1])]:
            mask = true_D_arr == d_val
            if np.any(mask):
                ax.hist(zk1_final[mask], bins=40, color=colour, alpha=0.6, label=label)
        ax.legend()
    else:
        ax.hist(zk1_final, bins=40, color="gray", alpha=0.6, label="Unknown")
        ax.legend()
    ax.set_xlim(0, 1)
    ax.set_xlabel("z_k1"); ax.set_ylabel("Count")
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Histogram of z_k1 (posterior P(D=1))")
    fig_zk1_hist.tight_layout()

    return {
        "ROC": fig_roc,
        "z_k1": fig_zk1,
        "z_k0": fig_zk0,
        "z_k1_hist": fig_zk1_hist,
    }
