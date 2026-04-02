"""
Dispatch functions and bootstrap for emery.

The top-level functions ``generate_multimethod_data``, ``estimate_ML``,
``pollinate_ML``, ``random_start``, and ``plot_ML`` route to the
appropriate method-specific implementation based on the ``type`` argument.

``boot_ML`` and ``aggregate_boot_ML`` provide non-parametric bootstrap
confidence intervals for any supported data type.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

from .binary import (
    bin_auc,
    estimate_ML_binary,
    generate_multimethod_binary,
    plot_ML_binary,
    pollinate_ML_binary,
    random_start_binary,
)
from .classes import BootML, MultiMethodMLEstimate
from .continuous import (
    estimate_ML_continuous,
    generate_multimethod_continuous,
    plot_ML_continuous,
    pollinate_ML_continuous,
)
from .ordinal import (
    estimate_ML_ordinal,
    generate_multimethod_ordinal,
    plot_ML_ordinal,
    pollinate_ML_ordinal,
)


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

_BINARY = "binary"
_ORDINAL = "ordinal"
_CONTINUOUS = "continuous"
_TYPES = (_BINARY, _ORDINAL, _CONTINUOUS)


def _check_type(type_: str) -> str:
    t = type_.lower().strip()
    if t not in _TYPES:
        raise ValueError(f"type must be one of {_TYPES}; got '{type_}'.")
    return t


def _not_implemented(type_: str, fn: str):
    raise NotImplementedError(
        f"'{fn}' is not yet implemented for type='{type_}'."
    )


# ---------------------------------------------------------------------------
# generate_multimethod_data
# ---------------------------------------------------------------------------


def generate_multimethod_data(
    type: str = "binary",
    n_method: int = 3,
    n_obs: int = 100,
    prev: float = 0.5,
    D=None,
    method_names=None,
    obs_names=None,
    **kwargs,
) -> dict:
    """
    Simulate paired multi-method measurement data.

    Parameters
    ----------
    type : str
        ``"binary"`` (currently the only supported type).
    n_method : int
        Number of measurement methods.
    n_obs : int
        Number of observations.
    prev : float
        Disease prevalence.
    D : array-like or None
        Optional true disease-state vector.
    method_names : list of str or None
        Method names (auto-generated when ``None``).
    obs_names : list of str or None
        Observation names (auto-generated when ``None``).
    **kwargs
        Additional arguments forwarded to the type-specific function.

    Returns
    -------
    dict with keys ``generated_data`` and ``params``.

    See Also
    --------
    generate_multimethod_binary
    """
    t = _check_type(type)
    if t == _BINARY:
        return generate_multimethod_binary(
            n_method=n_method, n_obs=n_obs, prev=prev, D=D,
            method_names=method_names, obs_names=obs_names, **kwargs,
        )
    if t == _ORDINAL:
        return generate_multimethod_ordinal(
            n_method=n_method, n_obs=n_obs, prev=prev, D=D,
            method_names=method_names, obs_names=obs_names, **kwargs,
        )
    if t == _CONTINUOUS:
        return generate_multimethod_continuous(
            n_method=n_method, n_obs=n_obs, prev=prev, D=D,
            method_names=method_names, obs_names=obs_names, **kwargs,
        )
    _not_implemented(t, "generate_multimethod_data")


# ---------------------------------------------------------------------------
# estimate_ML
# ---------------------------------------------------------------------------


def estimate_ML(
    type: str = "binary",
    data=None,
    freqs=None,
    init: Optional[dict] = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    save_progress: bool = True,
    **kwargs,
) -> MultiMethodMLEstimate:
    """
    Estimate maximum likelihood accuracy statistics via EM algorithm.

    Parameters
    ----------
    type : str
        ``"binary"`` (currently the only supported type).
    data : array-like, shape (n_obs, n_method)
        Observed measurements.  Use ``NaN`` for missing values.
    freqs : array-like or None
        Observation frequencies (for pre-summarised data).
    init : dict or None
        Initial parameter values.  See ``estimate_ML_binary`` for details.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    save_progress : bool
        Whether to store per-iteration diagnostics.
    **kwargs
        Additional arguments forwarded to the type-specific function.

    Returns
    -------
    MultiMethodMLEstimate

    See Also
    --------
    estimate_ML_binary
    """
    if data is None:
        raise ValueError("data must be provided.")
    t = _check_type(type)
    if t == _BINARY:
        return estimate_ML_binary(
            data=data, freqs=freqs, init=init,
            max_iter=max_iter, tol=tol, save_progress=save_progress, **kwargs,
        )
    if t == _ORDINAL:
        return estimate_ML_ordinal(
            data=data, freqs=freqs, init=init,
            max_iter=max_iter, tol=tol, save_progress=save_progress, **kwargs,
        )
    if t == _CONTINUOUS:
        return estimate_ML_continuous(
            data=data, freqs=freqs, init=init,
            max_iter=max_iter, tol=tol, save_progress=save_progress, **kwargs,
        )
    _not_implemented(t, "estimate_ML")


# ---------------------------------------------------------------------------
# pollinate_ML
# ---------------------------------------------------------------------------


def pollinate_ML(
    type: str = "binary",
    data=None,
    freqs=None,
    **kwargs,
) -> dict:
    """
    Generate data-driven starting values for the EM algorithm.

    Parameters
    ----------
    type : str
        ``"binary"`` (currently the only supported type).
    data : array-like
        Observed measurements.
    freqs : array-like or None
        Observation frequencies.

    Returns
    -------
    dict of initial parameter values.

    See Also
    --------
    pollinate_ML_binary
    """
    if data is None:
        raise ValueError("data must be provided.")
    t = _check_type(type)

    data = np.asarray(data, dtype=float)
    # Warn if any method has zero variance
    col_std = np.nanstd(data, axis=0)
    if np.any(col_std == 0.0):
        warnings.warn(
            "Data from one or more methods has zero variance.",
            stacklevel=2,
        )

    if t == _BINARY:
        return pollinate_ML_binary(data=data, freqs=freqs, **kwargs)
    if t == _ORDINAL:
        return pollinate_ML_ordinal(data=data, freqs=freqs, **kwargs)
    if t == _CONTINUOUS:
        return pollinate_ML_continuous(data=data, freqs=freqs, **kwargs)
    _not_implemented(t, "pollinate_ML")


# ---------------------------------------------------------------------------
# random_start
# ---------------------------------------------------------------------------


def random_start(
    type: str = "binary",
    n_method: Optional[int] = None,
    method_names=None,
) -> dict:
    """
    Generate random starting values for the EM algorithm.

    Parameters
    ----------
    type : str
        ``"binary"`` (currently the only supported type).
    n_method : int
        Number of methods.
    method_names : list of str or None
        Method names.

    Returns
    -------
    dict of initial parameter values.

    See Also
    --------
    random_start_binary
    """
    t = _check_type(type)
    if t == _BINARY:
        return random_start_binary(n_method=n_method, method_names=method_names)
    _not_implemented(t, "random_start")  # ordinal/continuous not yet supported


# ---------------------------------------------------------------------------
# plot_ML
# ---------------------------------------------------------------------------


def plot_ML(
    ML_est: MultiMethodMLEstimate,
    params: Optional[dict] = None,
) -> dict:
    """
    Create diagnostic plots for a ``MultiMethodMLEstimate`` object.

    Parameters
    ----------
    ML_est : MultiMethodMLEstimate
        Estimation result.
    params : dict or None
        Optional ground-truth parameters for simulation comparisons.

    Returns
    -------
    dict of matplotlib Figure objects.

    See Also
    --------
    plot_ML_binary
    """
    t = _check_type(ML_est.type)
    if t == _BINARY:
        return plot_ML_binary(ML_est, params=params)
    if t == _ORDINAL:
        return plot_ML_ordinal(ML_est, params=params)
    if t == _CONTINUOUS:
        return plot_ML_continuous(ML_est, params=params)
    _not_implemented(t, "plot_ML")


# ---------------------------------------------------------------------------
# Bootstrap worker (module-level so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------


def _boot_replicate(args: tuple):
    """Run a single bootstrap replicate.  Called by ``boot_ML``."""
    type_, data, boot_freqs, method_names, n_method, randomize_init, \
        max_iter, tol, rand_seed, extra_kwargs = args

    if randomize_init:
        if rand_seed is not None:
            np.random.seed(rand_seed)
        init_vals = random_start(type=type_, n_method=n_method,
                                 method_names=method_names)
    else:
        init_vals = pollinate_ML(type=type_, data=data, freqs=boot_freqs)

    rep = estimate_ML(
        type=type_, data=data, freqs=boot_freqs, init=init_vals,
        max_iter=max_iter, tol=tol, save_progress=False, **extra_kwargs,
    )
    return rep.get_results()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def boot_ML(
    type: str = "binary",
    data=None,
    freqs=None,
    n_boot: int = 100,
    n_study: Optional[int] = None,
    randomize_init: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-7,
    seed: Optional[int] = None,
    verbose: bool = False,
    n_jobs: int = 1,
    **kwargs,
) -> BootML:
    """
    Generate bootstrap replicates of ML estimates.

    For each bootstrap iteration, observations are sampled with replacement
    (proportional to ``freqs``) and ``estimate_ML`` is re-run on the
    resampled data.  This is useful for constructing non-parametric
    confidence intervals.

    Parameters
    ----------
    type : str
        Data type (``"binary"``).
    data : array-like, shape (n_obs, n_method)
        Observed measurements.
    freqs : array-like or None
        Observation frequencies.
    n_boot : int
        Number of bootstrap iterations.
    n_study : int or None
        Sample size per bootstrap replicate.  Defaults to the total sample
        size implied by ``freqs`` (or ``n_obs`` when ``freqs`` is ``None``).
    randomize_init : bool
        If ``True``, use ``random_start`` for each replicate instead of
        ``pollinate_ML``.  Useful for detecting local optima.
    max_iter : int
        Maximum EM iterations per replicate.
    tol : float
        Convergence tolerance per replicate.
    seed : int or None
        Global random seed.
    verbose : bool
        Print progress at each 10 % increment.
    n_jobs : int
        Number of parallel worker processes.  ``1`` (default) runs
        sequentially; ``-1`` uses all available CPU cores.
    **kwargs
        Additional arguments forwarded to ``estimate_ML``.

    Returns
    -------
    BootML
        Object containing ``v_0`` (original-data estimate) and ``v_star``
        (list of per-replicate ``results`` dicts).
    """
    if data is None:
        raise ValueError("data must be provided.")
    if seed is not None:
        np.random.seed(seed)

    t = _check_type(type)
    data = np.asarray(data, dtype=float)
    n_obs, n_method = data.shape

    if freqs is None:
        freqs = np.ones(n_obs)
    freqs = np.asarray(freqs, dtype=float)

    if n_study is None:
        n_study = int(np.sum(freqs))

    # Estimate on original data (progress not needed)
    v_0 = estimate_ML(
        type=t,
        data=data,
        freqs=freqs,
        max_iter=max_iter,
        tol=tol,
        save_progress=False,
        **kwargs,
    )

    method_names = v_0.names.get("method_names")
    prob = freqs / np.sum(freqs)

    # Pre-generate all bootstrap samples and per-replicate seeds in the main
    # process so results are reproducible regardless of n_jobs.
    all_boot_freqs = [
        np.bincount(
            np.random.choice(n_obs, size=n_study, replace=True, p=prob),
            minlength=n_obs,
        ).astype(float)
        for _ in range(n_boot)
    ]
    rand_seeds = (
        np.random.randint(0, 2**31, size=n_boot).tolist()
        if randomize_init else [None] * n_boot
    )

    arg_list = [
        (t, data, all_boot_freqs[i], method_names, n_method,
         randomize_init, max_iter, tol, rand_seeds[i], kwargs)
        for i in range(n_boot)
    ]

    workers = os.cpu_count() if n_jobs == -1 else n_jobs
    v_star: list[dict] = [None] * n_boot  # type: ignore[list-item]
    _last_pct_printed = -1

    if workers == 1:
        for i, args in enumerate(arg_list):
            v_star[i] = _boot_replicate(args)
            if verbose:
                pct = (i + 1) * 100 // n_boot
                if pct // 10 > _last_pct_printed // 10:
                    print(f"\rBootstrap: {i + 1}/{n_boot} [{pct}%]",
                          end="", flush=True)
                    _last_pct_printed = pct
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_boot_replicate, a): i
                       for i, a in enumerate(arg_list)}
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                v_star[idx] = future.result()
                completed += 1
                if verbose:
                    pct = completed * 100 // n_boot
                    if pct // 10 > _last_pct_printed // 10:
                        print(f"\rBootstrap: {completed}/{n_boot} [{pct}%]",
                              end="", flush=True)
                        _last_pct_printed = pct

    if verbose:
        print()  # move past the \r line

    params = {
        "type": t,
        "data": data,
        "freqs": freqs,
        "n_boot": n_boot,
        "n_study": n_study,
        "max_iter": max_iter,
        "tol": tol,
        "n_obs": n_obs,
        "seed": seed,
    }

    return BootML(v_0=v_0, v_star=v_star, params=params)


# ---------------------------------------------------------------------------
# Aggregate bootstrap results
# ---------------------------------------------------------------------------


def aggregate_boot_ML(boot_result: BootML) -> dict:
    """
    Aggregate bootstrap results by statistic.

    Rearranges the per-replicate ``results`` dicts from ``boot_ML`` into
    per-statistic arrays, making it easy to compute quantiles.

    Parameters
    ----------
    boot_result : BootML
        Output from ``boot_ML``.

    Returns
    -------
    dict
        Keys are statistic names (``"prev_est"``, ``"se_est"``, etc.).
        Each value is a dict with:

        * ``"values"`` – 2-D array, shape (n_boot, …).  Scalars become
          shape (n_boot, 1).
        * ``"boot_ids"`` – integer array ``[1, 2, …, n_boot]``.
    """
    stat_names = list(boot_result.v_0.results.keys())
    aggregated: dict = {}

    for stat in stat_names:
        rows = [
            np.atleast_1d(rep[stat]) for rep in boot_result.v_star
        ]
        aggregated[stat] = {
            "values": np.vstack(rows),           # shape (n_boot, ...)
            "boot_ids": np.arange(1, len(rows) + 1),
        }

    return aggregated


# ---------------------------------------------------------------------------
# Bootstrap plot
# ---------------------------------------------------------------------------


def plot_boot_ML(
    boot_result: BootML,
    probs=(0.10, 0.50, 0.90),
    stats_to_plot=("prev_est", "se_est", "sp_est"),
) -> dict:
    """
    Plot bootstrap distributions for selected accuracy statistics.

    Parameters
    ----------
    boot_result : BootML
        Output from ``boot_ML``.
    probs : sequence of float
        Quantile probabilities to mark with vertical lines.
    stats_to_plot : sequence of str
        Statistic names to plot (must be keys in the ``results`` dict).

    Returns
    -------
    dict of matplotlib Figure objects, keyed by statistic name.
    """
    import matplotlib.pyplot as plt

    agg = aggregate_boot_ML(boot_result)
    method_names = boot_result.v_0.names.get("method_names", [])

    _colours = [
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
        "#66A61E", "#E6AB02", "#A6761D", "#666666",
    ]

    figures: dict = {}

    for stat in stats_to_plot:
        if stat not in agg:
            continue

        values = agg[stat]["values"]   # shape (n_boot, k) or (n_boot, 1)
        n_boot, k = values.shape

        fig, axes = plt.subplots(k, 1, figsize=(7, 3 * k), squeeze=False)

        for j in range(k):
            ax = axes[j, 0]
            col_vals = values[:, j]
            colour = _colours[j % len(_colours)]

            ax.hist(col_vals, bins=100, color=colour, alpha=0.6)

            # Quantile lines
            qs = np.quantile(col_vals, probs)
            for q, p in zip(qs, probs):
                ax.axvline(q, color=colour, linestyle="--", linewidth=1.2,
                           label=f"p={p:.2f}: {q:.4f}")

            label = method_names[j] if j < len(method_names) else str(j)
            ax.set_title(f"{stat} – {label}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)
            ax.grid(color="gray", alpha=0.3)

        fig.suptitle(f"Bootstrap distribution: {stat}", y=1.01)
        fig.tight_layout()
        figures[stat] = fig

    return figures
