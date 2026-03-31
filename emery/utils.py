"""
Utility functions for emery.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def name_thing(thing: str = "", n: int = 1) -> list[str]:
    """
    Create zero-padded unique names for a set of items.

    Parameters
    ----------
    thing : str
        Prefix string (e.g. ``"method"`` or ``"obs"``).
    n : int
        Number of names to generate.

    Returns
    -------
    list of str
        e.g. ``name_thing("method", 12)`` → ``["method01", ..., "method12"]``.
    """
    width = len(str(n))
    return [f"{thing}{str(i).zfill(width)}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Disease-state helper
# ---------------------------------------------------------------------------


def define_disease_state(
    D=None,
    n_obs: Optional[int] = None,
    prev: Optional[float] = None,
) -> dict:
    """
    Define the true disease state of a simulated sample.

    Parameters
    ----------
    D : array-like or None
        True binary classification vector (1 = positive, 0 = negative).
        If ``None``, generated from ``n_obs`` and ``prev``.
    n_obs : int or None
        Number of observations (required when ``D`` is ``None``).
    prev : float or None
        Disease prevalence in [0, 1] (required when ``D`` is ``None``).

    Returns
    -------
    dict with keys ``D``, ``n_obs``, ``prev``, ``pos``, ``neg``.
    """
    if D is None:
        if prev is None or n_obs is None:
            raise ValueError(
                "Either D or both n_obs and prev must be provided."
            )
        pos = round(n_obs * prev)
        neg = n_obs - pos
        D = np.array([1.0] * pos + [0.0] * neg)
    else:
        D = np.asarray(D, dtype=float)
        n_obs = len(D)
        pos = int(np.sum(D))
        neg = int(np.sum(1.0 - D))
        prev = float(np.mean(D))

    return {"D": D, "n_obs": n_obs, "prev": prev, "pos": pos, "neg": neg}


# ---------------------------------------------------------------------------
# Censoring helper
# ---------------------------------------------------------------------------


def censor_data(
    n_obs: int,
    first_reads_all: bool,
    n_method_subset: int,
    n_method: int,
) -> np.ndarray:
    """
    Create a random censoring mask for multi-method data.

    For each observation, ``n_method_subset`` of the ``n_method`` methods are
    selected at random to produce a result; the rest are set to ``NaN``.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    first_reads_all : bool
        If ``True``, method 0 (the first method) is always observed.
    n_method_subset : int
        Number of methods with results per observation.
    n_method : int
        Total number of methods.

    Returns
    -------
    np.ndarray, shape (n_obs, n_method)
        Mask with 1.0 where observed and ``NaN`` where censored.
    """
    result = np.empty((n_obs, n_method))
    for i in range(n_obs):
        if not first_reads_all:
            mask = np.array(
                [1.0] * n_method_subset + [np.nan] * (n_method - n_method_subset)
            )
            np.random.shuffle(mask)
        else:
            rest = np.array(
                [1.0] * (n_method_subset - 1)
                + [np.nan] * (n_method - n_method_subset)
            )
            np.random.shuffle(rest)
            mask = np.concatenate([[1.0], rest])
        result[i] = mask
    return result


# ---------------------------------------------------------------------------
# Unique-observation summary
# ---------------------------------------------------------------------------


def unique_obs_summary(data) -> dict:
    """
    Reduce data by identifying unique rows and their frequencies.

    Useful for compressing repeated observations before calling
    ``estimate_ML_binary`` with a ``freqs`` argument.

    Parameters
    ----------
    data : array-like, shape (n_obs, n_method)
        Observation matrix (NaN allowed).

    Returns
    -------
    dict with keys:

    ``unique_obs`` : np.ndarray
        Matrix of unique rows (NaN preserved).
    ``duplicate_obs`` : list of list of int
        For each unique row, the original row indices that match it.
    ``obs_freqs`` : np.ndarray
        Count of how many original rows match each unique row.
    """
    data = np.asarray(data, dtype=float)

    # NaN != NaN in numpy comparisons, so convert to string for grouping.
    data_str = np.where(np.isnan(data), "nan", data.astype(str))

    unique_rows_str, inverse = np.unique(data_str, axis=0, return_inverse=True)
    obs_freqs = np.bincount(inverse)

    # Convert string rows back to float (restoring NaN).
    def _parse_row(row):
        return np.array(
            [np.nan if x == "nan" else float(x) for x in row], dtype=float
        )

    unique_obs = np.vstack([_parse_row(row) for row in unique_rows_str])
    duplicate_obs = [
        np.where(inverse == i)[0].tolist() for i in range(len(unique_rows_str))
    ]

    return {
        "unique_obs": unique_obs,
        "duplicate_obs": duplicate_obs,
        "obs_freqs": obs_freqs,
    }
