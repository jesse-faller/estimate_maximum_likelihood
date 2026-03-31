"""
Binary-method functions for emery.

Implements EM-based maximum likelihood estimation of sensitivity, specificity,
and prevalence for binary (pass/fail) measurements from multiple methods when
no gold standard is available.

Reference: Zhou, Obuchowski & McClish (2011), *Statistical Methods in
Diagnostic Medicine*, 2nd ed., Chapter 8.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist

from .classes import MultiMethodMLEstimate
from .utils import censor_data, define_disease_state, name_thing


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_multimethod_binary(
    n_method: int = 3,
    n_obs: int = 100,
    prev: float = 0.5,
    D=None,
    se=None,
    sp=None,
    method_names=None,
    obs_names=None,
    n_method_subset: Optional[int] = None,
    first_reads_all: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate binary (0/1) measurement data from multiple methods.

    Each observation is classified as positive (1) or negative (0) by each
    method according to the method's sensitivity and specificity.  An optional
    censoring pattern can be applied so that only a random subset of methods
    produces a result for each observation.

    Parameters
    ----------
    n_method : int
        Number of measurement methods.
    n_obs : int
        Number of observations.
    prev : float
        Disease prevalence (fraction positive), used when ``D`` is ``None``.
    D : array-like or None
        True disease-state vector (1 = positive, 0 = negative).  When
        provided, ``n_obs`` and ``prev`` are derived from it.
    se : array-like or None
        Sensitivity for each method, length ``n_method``.  Defaults to 0.9
        for every method.
    sp : array-like or None
        Specificity for each method, length ``n_method``.  Defaults to 0.9
        for every method.
    method_names : list of str or None
        Names for each method.  Auto-generated if ``None``.
    obs_names : list of str or None
        Names for each observation.  Auto-generated if ``None``.
    n_method_subset : int or None
        Number of methods that produce a result per observation.  Defaults to
        ``n_method`` (all methods observed for every observation).
    first_reads_all : bool
        If ``True``, method 0 always produces a result for every observation.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:

    ``generated_data`` : np.ndarray, shape (n_obs, n_method)
        Simulated binary results (0 / 1 / NaN).  Column names (as a list
        stored in ``params``) correspond to methods; row names to
        observations.
    ``params`` : dict
        All parameters used, including the true disease vector ``D`` and
        observed sensitivity / specificity.
    """
    if seed is not None:
        np.random.seed(seed)

    if se is None:
        se = np.full(n_method, 0.9)
    if sp is None:
        sp = np.full(n_method, 0.9)
    se = np.asarray(se, dtype=float)
    sp = np.asarray(sp, dtype=float)

    if n_method_subset is None:
        n_method_subset = n_method

    if method_names is None:
        method_names = name_thing("method", n_method)
    if obs_names is None:
        obs_names = name_thing("obs", n_obs)

    dis = define_disease_state(D=D, n_obs=n_obs, prev=prev)
    D_vec = dis["D"]
    n_obs = dis["n_obs"]

    subset_matrix = censor_data(
        n_obs=n_obs,
        first_reads_all=first_reads_all,
        n_method_subset=n_method_subset,
        n_method=n_method,
    )

    # Generate binary outcomes: P(positive) = se*D + (1-sp)*(1-D)
    generated_data = np.column_stack(
        [
            np.random.binomial(1, se[i] * D_vec + (1 - sp[i]) * (1 - D_vec))
            for i in range(n_method)
        ]
    ).astype(float)

    # Apply censoring mask (multiplying by NaN propagates NaN)
    generated_data = generated_data * subset_matrix

    # Observed se/sp (may differ from true values due to random sampling)
    pos_idx = D_vec == 1
    neg_idx = D_vec == 0

    se_observed = np.nanmean(generated_data[pos_idx, :], axis=0)
    sp_observed = 1.0 - np.nanmean(generated_data[neg_idx, :], axis=0)

    params = {
        "n_method": n_method,
        "n_obs": n_obs,
        "prev": dis["prev"],
        "se": se,
        "sp": sp,
        "D": dict(zip(obs_names, D_vec)),
        "se_observed": dict(zip(method_names, se_observed)),
        "sp_observed": dict(zip(method_names, sp_observed)),
        "method_names": method_names,
        "obs_names": obs_names,
    }

    return {"generated_data": generated_data, "params": params}


# ---------------------------------------------------------------------------
# EM initialisation (seeding)
# ---------------------------------------------------------------------------


def pollinate_ML_binary(
    data,
    freqs=None,
    **kwargs,
) -> dict:
    """
    Generate data-driven starting values for the binary EM algorithm.

    Uses majority-vote classification across methods to estimate initial
    prevalence, sensitivity, and specificity.

    Parameters
    ----------
    data : array-like, shape (n_obs, n_method)
        Observed binary measurements (NaN for missing).
    freqs : array-like or None
        Observation frequencies.  Defaults to all-ones.

    Returns
    -------
    dict with keys ``prev_1``, ``se_1``, ``sp_1``.
    """
    data = np.asarray(data, dtype=float)
    n_obs, n_method = data.shape

    method_names = name_thing("method", n_method)

    if freqs is None:
        freqs = np.ones(n_obs)
    freqs = np.asarray(freqs, dtype=float)

    not_missing = ~np.isnan(data)

    # Majority-vote estimate of disease state per observation
    D_majority = np.nanmean(data, axis=1)

    # Initial prevalence: weighted mean of majority-vote estimates
    total_w = np.sum(freqs)
    prev_1 = float(np.dot(D_majority, freqs) / total_w)

    # Initial se: weighted fraction of positive results among "positive" obs
    data_tmp_se = np.where(np.isnan(data), 0.0, data)
    w_pos = D_majority * freqs  # shape (n_obs,)
    se_1 = (w_pos @ data_tmp_se) / (w_pos @ not_missing)

    # Initial sp: weighted fraction of negative results among "negative" obs
    data_tmp_sp = np.where(np.isnan(data), 1.0, data)
    w_neg = (1.0 - D_majority) * freqs  # shape (n_obs,)
    sp_1 = (w_neg @ (1.0 - data_tmp_sp)) / (w_neg @ not_missing)

    return {"prev_1": prev_1, "se_1": se_1, "sp_1": sp_1}


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------


def random_start_binary(
    n_method: Optional[int] = None,
    method_names=None,
) -> dict:
    """
    Generate random starting values for the binary EM algorithm.

    Sensitivity and specificity are drawn from Beta(3, 1) (right-skewed toward
    higher diagnostic accuracy).  Prevalence is drawn from Uniform(0, 1).
    If ``se + sp < 1`` for any method, both values are reflected (``1 - value``)
    to ensure the test discriminates better than chance.

    Parameters
    ----------
    n_method : int
        Number of methods.
    method_names : list of str or None
        Method names.  Auto-generated if ``None``.

    Returns
    -------
    dict with keys ``prev_1``, ``se_1``, ``sp_1``.
    """
    if n_method is None:
        raise ValueError("n_method must be provided.")
    if method_names is None:
        method_names = name_thing("method", n_method)

    se_1 = beta_dist.rvs(3, 1, size=n_method)
    sp_1 = beta_dist.rvs(3, 1, size=n_method)
    prev_1 = float(np.random.uniform())

    # Ensure se + sp >= 1 (test better than random)
    comp_index = (se_1 + sp_1) < 1.0
    se_1[comp_index] = 1.0 - se_1[comp_index]
    sp_1[comp_index] = 1.0 - sp_1[comp_index]

    return {"prev_1": prev_1, "se_1": se_1, "sp_1": sp_1}


# ---------------------------------------------------------------------------
# EM algorithm
# ---------------------------------------------------------------------------


def estimate_ML_binary(
    data,
    freqs=None,
    init: Optional[dict] = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    save_progress: bool = True,
) -> MultiMethodMLEstimate:
    """
    Estimate sensitivity, specificity, and prevalence via EM algorithm.

    No gold standard is required.  The algorithm iterates between:

    * **E-step** – compute posterior probability ``q_k`` that each
      observation is disease-positive given current parameter estimates.
    * **M-step** – update ``se``, ``sp``, and ``prev`` to maximise the
      expected complete-data log-likelihood.

    Parameters
    ----------
    data : array-like, shape (n_obs, n_method)
        Observed binary measurements (0 / 1).  Use ``NaN`` for missing values.
        If a pandas DataFrame is passed, column names are used as method names
        and index values as observation names.
    freqs : array-like or None
        Observation frequencies (for pre-summarised data).  Defaults to
        all-ones (each row is one observation).
    init : dict or None
        Initial values with keys ``prev_1`` (float), ``se_1`` (array-like,
        length n_method), ``sp_1`` (array-like, length n_method).  If
        ``None`` or incomplete, ``pollinate_ML_binary`` is called.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance: stop when the maximum absolute change across
        all parameters is below this value.
    save_progress : bool
        If ``True``, store parameter values at every iteration in the
        ``prog`` slot of the returned object.

    Returns
    -------
    MultiMethodMLEstimate
        Object with ``results`` containing:

        * ``prev_est`` – estimated prevalence (float)
        * ``se_est``   – estimated sensitivities (ndarray, length n_method)
        * ``sp_est``   – estimated specificities (ndarray, length n_method)
        * ``qk_est``   – posterior probabilities (ndarray, length n_obs)
    """
    # ------------------------------------------------------------------
    # Coerce input
    # ------------------------------------------------------------------
    # Support pandas DataFrames by extracting names before converting.
    try:
        import pandas as pd  # optional
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

    missing_obs = np.isnan(data)
    not_missing = ~missing_obs

    if freqs is None:
        freqs = np.ones(n_obs)
    freqs = np.asarray(freqs, dtype=float)

    # ------------------------------------------------------------------
    # Validate / generate initial values
    # ------------------------------------------------------------------
    _need_init = (
        init is None
        or not isinstance(init, dict)
        or not all(k in init for k in ("prev_1", "se_1", "sp_1"))
        or any(init.get(k) is None for k in ("prev_1", "se_1", "sp_1"))
    )
    if _need_init:
        init = pollinate_ML_binary(data, freqs=freqs)

    BOUND_LO = 1e-13
    BOUND_HI = 1.0 - 1e-13

    se_m = np.clip(np.asarray(init["se_1"], dtype=float), BOUND_LO, BOUND_HI)
    sp_m = np.clip(np.asarray(init["sp_1"], dtype=float), BOUND_LO, BOUND_HI)
    prev_m = float(init["prev_1"])

    # ------------------------------------------------------------------
    # Inner helpers (closures over data / missing_obs / freqs)
    # ------------------------------------------------------------------

    def _calc_A2(se, prev):
        """P(data_k | D=1) * prev for each observation k."""
        log_like = np.nansum(
            data * np.log(se) + (1.0 - data) * np.log(1.0 - se), axis=1
        )
        return np.exp(log_like) * prev

    def _calc_B2(sp, prev):
        """P(data_k | D=0) * (1-prev) for each observation k."""
        log_like = np.nansum(
            data * np.log(1.0 - sp) + (1.0 - data) * np.log(sp), axis=1
        )
        return np.exp(log_like) * (1.0 - prev)

    def _calc_qk(A2, B2):
        """Posterior probability P(D=1 | data_k)."""
        return A2 / (A2 + B2)

    def _calc_next_se(qk):
        """M-step update for sensitivity."""
        dat_mat = np.where(missing_obs, 0.0, data)
        w = qk * freqs  # shape (n_obs,)
        se_new = (w @ dat_mat) / (w @ not_missing)
        return np.clip(se_new, BOUND_LO, BOUND_HI)

    def _calc_next_sp(qk):
        """M-step update for specificity."""
        dat_mat = np.where(missing_obs, 0.0, 1.0 - data)
        w = (1.0 - qk) * freqs  # shape (n_obs,)
        sp_new = (w @ dat_mat) / (w @ not_missing)
        return np.clip(sp_new, BOUND_LO, BOUND_HI)

    def _calc_next_prev(qk):
        """M-step update for prevalence."""
        return float(np.dot(qk, freqs) / np.sum(freqs))

    # ------------------------------------------------------------------
    # EM iterations
    # ------------------------------------------------------------------
    hist_se: list = []
    hist_sp: list = []
    hist_prev: list = []
    hist_A2: list = []
    hist_B2: list = []
    hist_qk: list = []

    final_iter = max_iter

    for it in range(1, max_iter + 1):
        # E-step
        A2_m = _calc_A2(se_m, prev_m)
        B2_m = _calc_B2(sp_m, prev_m)
        qk_m = _calc_qk(A2_m, B2_m)

        # Store current parameters (pre-M-step values)
        hist_se.append(se_m.copy())
        hist_sp.append(sp_m.copy())
        hist_prev.append(prev_m)
        hist_A2.append(A2_m.copy())
        hist_B2.append(B2_m.copy())
        hist_qk.append(qk_m.copy())

        # Convergence check (after the first iteration)
        if it > 1:
            max_change = max(
                np.max(np.abs(hist_se[-1] - hist_se[-2])),
                np.max(np.abs(hist_sp[-1] - hist_sp[-2])),
                abs(hist_prev[-1] - hist_prev[-2]),
            )
            if max_change < tol:
                final_iter = it
                break

        # M-step
        se_m = _calc_next_se(qk_m)
        sp_m = _calc_next_sp(qk_m)
        prev_m = _calc_next_prev(qk_m)
    else:
        final_iter = max_iter

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    results = {
        "prev_est": prev_m,
        "se_est": se_m,
        "sp_est": sp_m,
        "qk_est": qk_m,
    }

    out = MultiMethodMLEstimate(
        results=results,
        data=data,
        freqs=freqs,
        names={"method_names": method_names, "obs_names": obs_names},
        iter=final_iter,
        type="binary",
    )

    if save_progress:
        out.prog = {
            "prev": np.array(hist_prev),           # shape (iter,)
            "se": np.vstack(hist_se),               # shape (iter, n_method)
            "sp": np.vstack(hist_sp),               # shape (iter, n_method)
            "A2": np.vstack(hist_A2),               # shape (iter, n_obs)
            "B2": np.vstack(hist_B2),               # shape (iter, n_obs)
            "qk": np.vstack(hist_qk),               # shape (iter, n_obs)
        }

    return out


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


def bin_auc(se, sp) -> np.ndarray:
    """
    Calculate AUC for one or more sensitivity / specificity pairs.

    For a single operating point (se, sp), the AUC of the implied ROC curve
    is ``(se + sp) / 2``.  This is derived from the signed area of the
    triangle formed by the point, (0, 0), and (1, 1).

    Parameters
    ----------
    se : float or array-like
        Sensitivity value(s).
    sp : float or array-like
        Specificity value(s).

    Returns
    -------
    np.ndarray
        AUC value(s).
    """
    se = np.atleast_1d(np.asarray(se, dtype=float))
    sp = np.atleast_1d(np.asarray(sp, dtype=float))
    auc = np.empty(len(se))
    for i in range(len(se)):
        mat = np.array(
            [
                [se[i], 1.0 - sp[i], 1.0],
                [1.0,   1.0,          1.0],
                [0.0,   0.0,          1.0],
            ]
        )
        auc[i] = np.linalg.det(mat) / 2.0 + 0.5
    return auc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ML_binary(
    ML_est: MultiMethodMLEstimate,
    params: Optional[dict] = None,
) -> dict:
    """
    Create diagnostic plots for a binary EM estimation result.

    Requires ``save_progress=True`` when calling ``estimate_ML_binary``
    (the default).

    Parameters
    ----------
    ML_est : MultiMethodMLEstimate
        Estimation result from ``estimate_ML_binary``.
    params : dict or None
        Optional ground-truth parameters (from ``generate_multimethod_binary``
        ``params`` key).  The following sub-keys are used if present:

        * ``se`` – true sensitivity values
        * ``sp`` – true specificity values
        * ``D``  – true disease state (dict or array-like of 0/1)

    Returns
    -------
    dict of matplotlib Figure objects with keys:
    ``prev``, ``se``, ``sp``, ``qk``, ``qk_hist``, ``se_sp``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not ML_est.prog:
        raise ValueError(
            "No progress data found.  Re-run estimate_ML_binary with "
            "save_progress=True."
        )

    if params is None:
        params = {}

    method_names = ML_est.names.get("method_names", [])
    obs_names = ML_est.names.get("obs_names", [])
    n_method = len(method_names)
    n_obs = len(obs_names)
    n_iter = ML_est.iter

    prog = ML_est.prog
    iters = np.arange(1, len(prog["prev"]) + 1)

    # Colour cycle (Dark2-like)
    _colours = [
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
        "#66A61E", "#E6AB02", "#A6761D", "#666666",
    ]
    method_colours = [_colours[i % len(_colours)] for i in range(n_method)]

    # ------------------------------------------------------------------
    # 1. Prevalence progress
    # ------------------------------------------------------------------
    fig_prev, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, prog["prev"], color=_colours[0], label="Estimate")
    if "prev" in params and params["prev"] is not None:
        ax.axhline(params["prev"], color="gray", linestyle="--", label="Truth")
    ax.set_xlim(1, max(n_iter, 2))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("prev")
    ax.legend()
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Prevalence estimate over iterations")
    fig_prev.tight_layout()

    # ------------------------------------------------------------------
    # 2. Sensitivity progress
    # ------------------------------------------------------------------
    fig_se, ax = plt.subplots(figsize=(7, 4))
    for i, (mname, colour) in enumerate(zip(method_names, method_colours)):
        ax.plot(iters, prog["se"][:, i], color=colour, label=mname)
    ax.set_xlim(1, max(n_iter, 2))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("se")
    ax.legend()
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Sensitivity estimates over iterations")
    fig_se.tight_layout()

    # ------------------------------------------------------------------
    # 3. Specificity progress
    # ------------------------------------------------------------------
    fig_sp, ax = plt.subplots(figsize=(7, 4))
    for i, (mname, colour) in enumerate(zip(method_names, method_colours)):
        ax.plot(iters, prog["sp"][:, i], color=colour, label=mname)
    ax.set_xlim(1, max(n_iter, 2))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("sp")
    ax.legend()
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Specificity estimates over iterations")
    fig_sp.tight_layout()

    # ------------------------------------------------------------------
    # 4. Posterior probabilities (q_k) progress
    # ------------------------------------------------------------------
    # Colour by true disease state if available
    true_D = params.get("D")
    if true_D is not None:
        # Support dict (obs_name → 0/1) or array-like
        if isinstance(true_D, dict):
            true_D_arr = np.array([true_D.get(name, np.nan) for name in obs_names])
        else:
            true_D_arr = np.asarray(list(true_D), dtype=float)
    else:
        true_D_arr = np.full(n_obs, np.nan)

    fig_qk, ax = plt.subplots(figsize=(7, 4))
    for k in range(n_obs):
        d = true_D_arr[k]
        if np.isnan(d):
            colour = "gray"
        elif d == 1:
            colour = _colours[0]
        else:
            colour = _colours[1]
        ax.plot(iters, prog["qk"][:, k], color=colour, alpha=0.4, linewidth=0.7)

    # Legend proxies
    import matplotlib.lines as mlines
    handles = []
    if not np.all(np.isnan(true_D_arr)):
        handles.append(mlines.Line2D([], [], color=_colours[0], label="Class 1"))
        handles.append(mlines.Line2D([], [], color=_colours[1], label="Class 0"))
    else:
        handles.append(mlines.Line2D([], [], color="gray", label="Unknown"))
    ax.legend(handles=handles)
    ax.set_xlim(1, max(n_iter, 2))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("qk")
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Posterior probabilities over iterations")
    fig_qk.tight_layout()

    # ------------------------------------------------------------------
    # 5. Histogram of final q_k values
    # ------------------------------------------------------------------
    fig_qk_hist, ax = plt.subplots(figsize=(7, 4))
    qk_final = ML_est.results["qk_est"]
    obs_freqs = ML_est.freqs

    if not np.all(np.isnan(true_D_arr)):
        for d_val, label, colour in [
            (1.0, "Class 1", _colours[0]),
            (0.0, "Class 0", _colours[1]),
        ]:
            mask = true_D_arr == d_val
            if np.any(mask):
                ax.hist(
                    qk_final[mask],
                    bins=40,
                    weights=obs_freqs[mask],
                    color=colour,
                    alpha=0.6,
                    label=label,
                    density=True,
                )
        ax.legend()
    else:
        ax.hist(
            qk_final,
            bins=40,
            weights=obs_freqs,
            color="gray",
            alpha=0.6,
            density=True,
            label="Unknown",
        )
        ax.legend()

    ax.set_xlim(0, 1)
    ax.set_xlabel("q_k (posterior probability)")
    ax.set_ylabel("Density")
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Histogram of final posterior probabilities")
    fig_qk_hist.tight_layout()

    # ------------------------------------------------------------------
    # 6. Se/Sp path plot
    # ------------------------------------------------------------------
    fig_se_sp, ax = plt.subplots(figsize=(6, 6))

    for i, (mname, colour) in enumerate(zip(method_names, method_colours)):
        sp_path = prog["sp"][:, i]
        se_path = prog["se"][:, i]

        # Path through se/sp space
        ax.plot(sp_path, se_path, color=colour, alpha=0.7)

        # Final estimate (filled circle)
        ax.scatter(
            ML_est.results["sp_est"][i],
            ML_est.results["se_est"][i],
            color=colour,
            marker="o",
            s=80,
            zorder=5,
            label=mname,
        )

        # True value if provided (open circle) + dashed connector
        true_se = params.get("se")
        true_sp = params.get("sp")
        if true_se is not None and true_sp is not None:
            true_se_arr = np.asarray(true_se, dtype=float)
            true_sp_arr = np.asarray(true_sp, dtype=float)
            ax.scatter(
                true_sp_arr[i],
                true_se_arr[i],
                color=colour,
                marker="o",
                s=80,
                facecolors="none",
                zorder=5,
            )
            ax.plot(
                [ML_est.results["sp_est"][i], true_sp_arr[i]],
                [ML_est.results["se_est"][i], true_se_arr[i]],
                color=colour,
                linestyle="--",
                alpha=0.5,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right")
    ax.grid(color="gray", alpha=0.3)
    ax.set_title("Se/Sp path (filled = estimate, open = truth)")
    fig_se_sp.tight_layout()

    return {
        "prev": fig_prev,
        "se": fig_se,
        "sp": fig_sp,
        "qk": fig_qk,
        "qk_hist": fig_qk_hist,
        "se_sp": fig_se_sp,
    }
