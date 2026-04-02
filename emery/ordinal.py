"""
Ordinal-method functions for emery.

Implements EM-based maximum likelihood estimation of AUC, cumulative mass
functions (CMF), and prevalence for ordinal (e.g. Likert-scale) measurements
from multiple methods when no gold standard is available.

Reference: Zhou (2005), *Biometrics*, 61(2), 456–463.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist

from .classes import MultiMethodMLEstimate
from .utils import censor_data, define_disease_state, name_thing


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_multimethod_ordinal(
    n_method: int = 3,
    n_obs: int = 100,
    prev: float = 0.5,
    D=None,
    n_level: int = 5,
    pmf_pos=None,
    pmf_neg=None,
    method_names=None,
    level_names=None,
    obs_names=None,
    n_method_subset: Optional[int] = None,
    first_reads_all: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate ordinal measurement data from multiple methods.

    Each observation is rated on a scale of 1 to ``n_level`` by each method,
    according to method-specific probability mass functions (PMFs) for the
    positive and negative disease states.

    Parameters
    ----------
    n_method : int
        Number of measurement methods.
    n_obs : int
        Number of observations.
    prev : float
        Disease prevalence, used when ``D`` is ``None``.
    D : array-like or None
        True disease-state vector (1 = positive, 0 = negative).
    n_level : int
        Number of ordinal levels (default 5).
    pmf_pos : array-like, shape (n_method, n_level) or None
        Probability mass functions for positive observations.  Each row
        corresponds to one method and will be normalised to sum to 1.
        Defaults to linearly increasing weights ``[0, 1, …, n_level-1]``
        (higher levels more likely when positive).
    pmf_neg : array-like, shape (n_method, n_level) or None
        PMF for negative observations.  Defaults to linearly decreasing
        weights ``[n_level-1, …, 1, 0]``.
    method_names, level_names, obs_names : list of str or None
        Optional names.  Auto-generated when ``None``.
    n_method_subset : int or None
        Number of methods that produce a result per observation.
    first_reads_all : bool
        If ``True``, method 0 always produces a result.
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys ``generated_data`` (ndarray, shape n_obs × n_method) and
    ``params`` (dict of simulation parameters).
    """
    if seed is not None:
        np.random.seed(seed)

    if n_method_subset is None:
        n_method_subset = n_method

    if method_names is None:
        method_names = name_thing("method", n_method)
    if level_names is None:
        level_names = name_thing("level", n_level)
    if obs_names is None:
        obs_names = name_thing("obs", n_obs)

    # Default PMFs: linearly increasing / decreasing weights
    if pmf_pos is None:
        pmf_pos = np.tile(np.arange(n_level, dtype=float), (n_method, 1))
    if pmf_neg is None:
        pmf_neg = np.tile(np.arange(n_level - 1, -1, -1, dtype=float), (n_method, 1))

    pmf_pos = np.asarray(pmf_pos, dtype=float)
    pmf_neg = np.asarray(pmf_neg, dtype=float)

    # Normalise rows to valid PMFs
    pmf_pos = pmf_pos / pmf_pos.sum(axis=1, keepdims=True)
    pmf_neg = pmf_neg / pmf_neg.sum(axis=1, keepdims=True)

    dis = define_disease_state(D=D, n_obs=n_obs, prev=prev)
    n_obs = dis["n_obs"]

    subset_matrix = censor_data(
        n_obs=n_obs,
        first_reads_all=first_reads_all,
        n_method_subset=n_method_subset,
        n_method=n_method,
    )

    levels = np.arange(1, n_level + 1)

    pos_data = np.column_stack(
        [np.random.choice(levels, size=dis["pos"], replace=True, p=pmf_pos[i])
         for i in range(n_method)]
    ).astype(float)
    neg_data = np.column_stack(
        [np.random.choice(levels, size=dis["neg"], replace=True, p=pmf_neg[i])
         for i in range(n_method)]
    ).astype(float)

    generated_data = np.vstack([pos_data, neg_data]) * subset_matrix

    D_vec = dis["D"]
    pos_mask = D_vec == 1
    neg_mask = D_vec == 0

    # Observed se/sp at each threshold level
    se_obs = np.column_stack(
        [np.nanmean(generated_data[pos_mask, :] > x, axis=0) for x in range(1, n_level + 1)]
    )  # shape (n_method, n_level)
    sp_obs = np.column_stack(
        [np.nanmean(generated_data[neg_mask, :] < x, axis=0) for x in range(1, n_level + 1)]
    )

    params = {
        "n_method": n_method,
        "n_level": n_level,
        "n_obs": n_obs,
        "prev": dis["prev"],
        "D": dict(zip(obs_names, D_vec)),
        "pmf_pos": pmf_pos,
        "pmf_neg": pmf_neg,
        "se_observed": se_obs,
        "sp_observed": sp_obs,
        "method_names": method_names,
        "level_names": level_names,
        "obs_names": obs_names,
    }

    return {"generated_data": generated_data, "params": params}


# ---------------------------------------------------------------------------
# EM initialisation
# ---------------------------------------------------------------------------


def pollinate_ML_ordinal(
    data,
    freqs=None,
    n_level: Optional[int] = None,
    threshold_level: Optional[int] = None,
    level_names=None,
    **kwargs,
) -> dict:
    """
    Generate data-driven starting values for the ordinal EM algorithm.

    Uses majority-vote classification (relative to ``threshold_level``) to
    estimate initial PMFs and prevalence.

    Parameters
    ----------
    data : array-like, shape (n_obs, n_method)
        Observed ordinal measurements (integer levels, NaN for missing).
    freqs : array-like or None
        Observation frequencies.
    n_level : int or None
        Number of ordinal levels.  Inferred from data when ``None``.
    threshold_level : int or None
        Threshold used to define positive (>= threshold) vs negative.
        Defaults to ``ceil(n_level / 2)``.
    level_names : list of str or None
        Level names.

    Returns
    -------
    dict with keys ``pi_1_1``, ``phi_1ij_1``, ``phi_0ij_1``, ``n_level``.
    """
    data = np.asarray(data, dtype=float)
    n_obs, n_method = data.shape

    method_names = name_thing("method", n_method)

    if freqs is None:
        freqs = np.ones(n_obs)
    freqs = np.asarray(freqs, dtype=float)

    if n_level is None:
        unique_vals = np.unique(data[~np.isnan(data)])
        n_level = int(len(unique_vals))

    if threshold_level is None:
        threshold_level = int(np.ceil(n_level / 2))

    if level_names is None:
        level_names = name_thing("level", n_level)

    # Majority vote: round mean across methods, classify vs threshold
    jitter = np.random.uniform(-1e-6, 1e-6, n_obs)
    D_majority = (np.round(np.nanmean(data, axis=1) + jitter) >= threshold_level).astype(float)

    pi_1_1 = float(np.average(D_majority, weights=freqs))

    pos_mask = D_majority == 1
    neg_mask = D_majority == 0

    data_pos = data[pos_mask, :]
    data_neg = data[neg_mask, :]
    freqs_pos = freqs[pos_mask]
    freqs_neg = freqs[neg_mask]

    def _calc_phi(data_subset, freqs_subset):
        total = np.sum(freqs_subset)
        phi = np.zeros((n_level, n_method))
        for j_idx in range(n_level):
            j_val = j_idx + 1
            matches = (data_subset == j_val)  # NaN == j_val → False
            phi[j_idx, :] = np.sum(matches * freqs_subset[:, np.newaxis], axis=0) / total
        return np.clip(phi, 1e-10, 1 - 1e-10)

    phi_1ij_1 = _calc_phi(data_pos, freqs_pos)
    phi_0ij_1 = _calc_phi(data_neg, freqs_neg)

    return {
        "pi_1_1": pi_1_1,
        "phi_1ij_1": phi_1ij_1,
        "phi_0ij_1": phi_0ij_1,
        "n_level": n_level,
    }


# ---------------------------------------------------------------------------
# EM algorithm
# ---------------------------------------------------------------------------


def estimate_ML_ordinal(
    data,
    freqs=None,
    init: Optional[dict] = None,
    level_names=None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    save_progress: bool = True,
) -> MultiMethodMLEstimate:
    """
    Estimate AUC and CMFs for ordinal multi-method data via EM algorithm.

    Parameters
    ----------
    data : array-like, shape (n_obs, n_method)
        Observed ordinal levels (integer values 1 … n_level, NaN for missing).
        Column names are used as method names if available.
    freqs : array-like or None
        Observation frequencies.
    init : dict or None
        Initial values with keys ``pi_1_1`` (float), ``phi_1ij_1``
        (n_level × n_method array), ``phi_0ij_1`` (n_level × n_method array),
        ``n_level`` (int).  Auto-generated via ``pollinate_ML_ordinal`` when
        ``None``.
    level_names : list of str or None
        Ordered names for each ordinal level.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the conditional log-likelihood change.
    save_progress : bool
        Store per-iteration diagnostics in the returned object.

    Returns
    -------
    MultiMethodMLEstimate
        ``results`` keys:

        * ``prev_est``     – estimated prevalence
        * ``A_i_est``      – AUC per method (ndarray, length n_method)
        * ``phi_1ij_est``  – estimated positive CMF (n_level × n_method)
        * ``phi_0ij_est``  – estimated negative CMF (n_level × n_method)
        * ``q_k1_est``     – P(D=1 | data_k) per observation
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

    missing_obs = np.isnan(data)
    not_missing = ~missing_obs

    # ------------------------------------------------------------------
    # Initial values
    # ------------------------------------------------------------------
    _need_init = (
        init is None
        or not isinstance(init, dict)
        or not all(k in init for k in ("pi_1_1", "phi_1ij_1", "phi_0ij_1", "n_level"))
        or any(init.get(k) is None for k in ("pi_1_1", "phi_1ij_1", "phi_0ij_1", "n_level"))
    )
    if _need_init:
        n_level_inferred = int(len(np.unique(data[~np.isnan(data)])))
        init = pollinate_ML_ordinal(
            data, freqs=freqs, n_level=n_level_inferred, level_names=level_names
        )

    n_level = int(init["n_level"])
    if level_names is None:
        level_names = name_thing("level", n_level)

    p_t = float(init["pi_1_1"])
    phi_1ij_t = np.asarray(init["phi_1ij_1"], dtype=float)   # (n_level, n_method)
    phi_0ij_t = np.asarray(init["phi_0ij_1"], dtype=float)   # (n_level, n_method)

    # ------------------------------------------------------------------
    # Precompute y_k indicator tensor: y_k[j, i, k] = 1 if obs k,
    # method i was rated level j+1 and was not missing.
    # Shape: (n_level, n_method, n_obs)
    # ------------------------------------------------------------------
    y_k = np.zeros((n_level, n_method, n_obs), dtype=float)
    for j_idx in range(n_level):
        j_val = j_idx + 1
        y_k[j_idx, :, :] = ((data == j_val) & not_missing).T

    # ------------------------------------------------------------------
    # Inner helpers
    # ------------------------------------------------------------------

    def _calc_g_d(phi_dij):
        """P(data_k | D=d) for each observation k."""
        log_phi = np.log(np.maximum(phi_dij, 1e-300))  # (n_level, n_method)
        # sum_j sum_i y_k[j,i,k] * log_phi[j,i] → (n_obs,)
        log_g = np.einsum("ji,jik->k", log_phi, y_k)
        return np.maximum(np.exp(log_g), 1e-300)

    def _calc_q_kd(d, g_1, g_0):
        """Posterior P(D=d | data_k)."""
        num = (p_t * g_1) * d + ((1.0 - p_t) * g_0) * (1.0 - d)
        denom = (1.0 - p_t) * g_0 + p_t * g_1
        return num / denom

    def _calc_l_cond(q_k0, q_k1, g_0, g_1):
        """Conditional log-likelihood."""
        return float(np.nansum(
            q_k1 * freqs * np.log(np.maximum(g_1, 1e-300))
            + q_k0 * freqs * np.log(np.maximum(g_0, 1e-300))
        ))

    def _calc_A_i(phi_1ij, phi_0ij):
        """AUC per method (Wilcoxon / trapezoidal)."""
        n_lev = phi_1ij.shape[0]
        outer = np.zeros(n_method)
        for j in range(n_lev - 1):
            inner = np.sum(phi_1ij[j + 1:, :], axis=0)
            outer += phi_0ij[j, :] * inner
        return outer + 0.5 * np.sum(phi_1ij * phi_0ij, axis=0)

    def _calc_next_prev(q_k1):
        return float(np.dot(q_k1, freqs) / np.sum(freqs))

    def _calc_next_phi_dij(q_kd):
        """M-step update for phi_dij."""
        w = q_kd * freqs  # (n_obs,)
        # Numerator: sum_k y_k[j,i,k] * w[k] → (n_level, n_method)
        numerator = np.einsum("jik,k->ji", y_k, w)
        # Denominator: sum_k w[k] * not_missing[k,i] → (n_method,)
        denom = w @ not_missing  # (n_method,)
        phi_new = numerator / np.maximum(denom[np.newaxis, :], 1e-300)
        return np.maximum(phi_new, 1e-300)

    # ------------------------------------------------------------------
    # EM iterations
    # ------------------------------------------------------------------
    hist_prev: list = []
    hist_phi_1: list = []
    hist_phi_0: list = []
    hist_A_i: list = []
    hist_g_1: list = []
    hist_g_0: list = []
    hist_q_k1: list = []
    hist_q_k0: list = []
    hist_l_cond: list = []

    final_iter = max_iter

    for it in range(1, max_iter + 1):
        A_i = _calc_A_i(phi_1ij_t, phi_0ij_t)
        g_1_t = _calc_g_d(phi_1ij_t)
        g_0_t = _calc_g_d(phi_0ij_t)
        q_k1_t = _calc_q_kd(1, g_1_t, g_0_t)
        q_k0_t = _calc_q_kd(0, g_1_t, g_0_t)
        l_cond_t = _calc_l_cond(q_k0_t, q_k1_t, g_0_t, g_1_t)

        hist_prev.append(p_t)
        hist_phi_1.append(phi_1ij_t.copy())
        hist_phi_0.append(phi_0ij_t.copy())
        hist_A_i.append(A_i.copy())
        hist_g_1.append(g_1_t.copy())
        hist_g_0.append(g_0_t.copy())
        hist_q_k1.append(q_k1_t.copy())
        hist_q_k0.append(q_k0_t.copy())
        hist_l_cond.append(l_cond_t)

        if it > 1 and abs(hist_l_cond[-1] - hist_l_cond[-2]) < tol:
            final_iter = it
            break

        p_t = _calc_next_prev(q_k1_t)
        phi_1ij_t = _calc_next_phi_dij(q_k1_t)
        phi_0ij_t = _calc_next_phi_dij(q_k0_t)
    else:
        final_iter = max_iter

    results = {
        "prev_est": p_t,
        "A_i_est": A_i,
        "phi_1ij_est": phi_1ij_t,
        "phi_0ij_est": phi_0ij_t,
        "q_k1_est": q_k1_t,
    }

    out = MultiMethodMLEstimate(
        results=results,
        data=data,
        freqs=freqs,
        names={
            "method_names": method_names,
            "obs_names": obs_names,
            "level_names": level_names,
        },
        iter=final_iter,
        type="ordinal",
    )

    if save_progress:
        out.prog = {
            "prev": np.array(hist_prev),
            "phi_1ij": hist_phi_1,       # list of (n_level, n_method) arrays
            "phi_0ij": hist_phi_0,
            "A_i": np.vstack(hist_A_i),  # (iter, n_method)
            "g_1": np.vstack(hist_g_1),  # (iter, n_obs)
            "g_0": np.vstack(hist_g_0),
            "q_k1": np.vstack(hist_q_k1),  # (iter, n_obs)
            "q_k0": np.vstack(hist_q_k0),
            "l_cond": np.array(hist_l_cond),
        }

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ML_ordinal(
    ML_est: MultiMethodMLEstimate,
    params: Optional[dict] = None,
) -> dict:
    """
    Create diagnostic plots for an ordinal EM estimation result.

    Parameters
    ----------
    ML_est : MultiMethodMLEstimate
        Result from ``estimate_ML_ordinal`` (requires ``save_progress=True``
        for progress plots).
    params : dict or None
        Optional ground-truth parameters.  ``params["D"]`` (dict or array)
        is used to colour observation traces.

    Returns
    -------
    dict of matplotlib Figure objects with keys:
    ``ROC``, ``q_k1``, ``q_k0``, ``q_k1_hist``, ``phi_d``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if params is None:
        params = {}

    method_names = ML_est.names.get("method_names", [])
    obs_names = ML_est.names.get("obs_names", [])
    level_names = ML_est.names.get("level_names", [])
    n_method = len(method_names)
    n_obs = len(obs_names)
    n_level = len(level_names)
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
    # 1. ROC curves
    # ------------------------------------------------------------------
    phi_1 = ML_est.results["phi_1ij_est"]   # (n_level, n_method)
    phi_0 = ML_est.results["phi_0ij_est"]

    def _survival(phi):
        # P(T >= level_l | D) for each threshold l → (n_method, n_level)
        result = np.zeros((n_method, n_level))
        for l in range(n_level):
            result[:, l] = np.clip(np.sum(phi[l:, :], axis=0), 0, 1)
        return result

    tpr = _survival(phi_1)   # (n_method, n_level)
    fpr = _survival(phi_0)

    A_i = ML_est.results["A_i_est"]

    fig_roc, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    auc_lines = []
    for i, (mname, colour) in enumerate(zip(method_names, method_colours)):
        xs = np.concatenate([[0], fpr[i]])
        ys = np.concatenate([[0], tpr[i]])
        ax.plot(xs, ys, color=colour, marker="o", markersize=4)
        for l, lname in enumerate(level_names):
            ax.annotate(lname, (fpr[i, l], tpr[i, l]),
                        fontsize=7, ha="left", va="bottom", color=colour)
        auc_lines.append(f"{mname}: {A_i[i]:.3f}")

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
    ax.set_title("ROC Curves")
    fig_roc.tight_layout()

    # ------------------------------------------------------------------
    # 2 & 3. q_k1 and q_k0 progress plots
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
        import matplotlib.lines as mlines
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

    fig_qk1 = _progress_plot("q_k1", "q_k1")
    fig_qk0 = _progress_plot("q_k0", "q_k0")

    # ------------------------------------------------------------------
    # 4. q_k1 histogram
    # ------------------------------------------------------------------
    fig_qk1_hist, ax = plt.subplots(figsize=(7, 4))
    qk1_final = ML_est.results["q_k1_est"]
    if not np.all(np.isnan(true_D_arr)):
        for d_val, label, colour in [(1.0, "Class 1", _colours[0]),
                                     (0.0, "Class 0", _colours[1])]:
            mask = true_D_arr == d_val
            if np.any(mask):
                ax.hist(qk1_final[mask], bins=40, color=colour, alpha=0.6, label=label)
        ax.legend()
    else:
        ax.hist(qk1_final, bins=40, color="gray", alpha=0.6, label="Unknown")
        ax.legend()
    ax.set_xlim(0, 1)
    ax.set_xlabel("q_k1"); ax.set_ylabel("Count")
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Histogram of q_k1 (posterior P(D=1))")
    fig_qk1_hist.tight_layout()

    # ------------------------------------------------------------------
    # 5. Stacked bar plot of phi_d (estimated CMFs)
    # ------------------------------------------------------------------
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, n_level))

    fig_phi, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (phi, d_label) in zip(axes,
                                   [(phi_1, "d = 1 (positive)"),
                                    (phi_0, "d = 0 (negative)")]):
        bottoms = np.zeros(n_method)
        for j in range(n_level):
            heights = phi[j, :]   # (n_method,)
            ax.bar(method_names, heights, bottom=bottoms,
                   color=blues[j], edgecolor="black", linewidth=0.5,
                   label=level_names[j], alpha=0.85)
            bottoms += heights
        ax.set_ylim(0, 1)
        ax.set_title(d_label)
        ax.set_xlabel("Method")
        ax.grid(axis="y", color="gray", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Probability")
    handles = [mpatches.Patch(color=blues[j], label=level_names[j])
               for j in range(n_level)]
    fig_phi.legend(handles=handles, title="Level", loc="center right",
                   bbox_to_anchor=(1.0, 0.5))
    fig_phi.suptitle("Estimated PMF by disease state")
    fig_phi.tight_layout()

    return {
        "ROC": fig_roc,
        "q_k1": fig_qk1,
        "q_k0": fig_qk0,
        "q_k1_hist": fig_qk1_hist,
        "phi_d": fig_phi,
    }
