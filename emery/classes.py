"""
Data classes for emery: MultiMethodMLEstimate and BootML.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MultiMethodMLEstimate:
    """
    Result of multi-method maximum likelihood estimation via EM algorithm.

    Attributes
    ----------
    results : dict
        Estimated accuracy statistics.  For binary methods the keys are
        ``prev_est`` (float), ``se_est`` (ndarray, shape n_method),
        ``sp_est`` (ndarray, shape n_method), ``qk_est`` (ndarray, shape n_obs).
    data : np.ndarray
        Raw input data, shape (n_obs, n_method), NaN for missing values.
    freqs : np.ndarray
        Observation frequencies, shape (n_obs,).
    names : dict
        ``method_names`` and ``obs_names`` lists.
    iter : int
        Number of EM iterations until convergence.
    prog : dict
        Progress data from each iteration when ``save_progress=True``.
        Keys match ``results`` but each value is a 2-D array with one row
        per iteration.
    type : str
        Data type: ``"binary"``, ``"ordinal"``, or ``"continuous"``.
    """

    results: dict = field(default_factory=dict)
    data: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    names: dict = field(default_factory=dict)
    iter: int = 0
    prog: dict = field(default_factory=dict)
    type: str = ""

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        # Mirror the R show() method: hide per-observation posterior arrays
        # (qk_est, q_k1_est, z_k1_est, z_k0_est, …)
        import re
        display = {
            k: v for k, v in self.results.items()
            if not re.search(r"q_?k|z_?k", k)
        }
        lines = [
            f"MultiMethodMLEstimate(type='{self.type}', iter={self.iter})"
        ]
        for k, v in display.items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim > 1:
                lines.append(f"  {k}: {arr.shape} matrix")
            elif arr.size == 1:
                lines.append(f"  {k}: {float(arr.flat[0]):.6f}")
            else:
                vals = ", ".join(f"{float(x):.6f}" for x in arr.flat[:8])
                suffix = ", ..." if arr.size > 8 else ""
                lines.append(f"  {k}: [{vals}{suffix}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Accessor / mutator helpers (mirrors R S4 generics)
    # ------------------------------------------------------------------

    def get_results(self) -> dict:
        """Return the ``results`` dict."""
        return self.results

    def get_names(self, name: str):
        """Return a named element from the ``names`` dict."""
        return self.names.get(name)

    def set_freqs(self, freqs=None) -> "MultiMethodMLEstimate":
        """
        Return a copy of this object with updated ``freqs``.

        Parameters
        ----------
        freqs : array-like or None
            New frequency vector.  If ``None``, defaults to all-ones.
        """
        obj = copy.copy(self)
        if freqs is None:
            freqs = np.ones(len(self.data))
        obj.freqs = np.asarray(freqs, dtype=float)
        return obj


@dataclass
class BootML:
    """
    Bootstrap ML results container.

    Attributes
    ----------
    v_0 : MultiMethodMLEstimate
        Estimate from the original data.
    v_star : list of dict
        Each element is the ``results`` dict from one bootstrap replicate.
    params : dict
        Parameters used when generating the bootstrap samples.
    """

    v_0: Optional[MultiMethodMLEstimate] = None
    v_star: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.v_star)
        return (
            f"BootML(n_boot={n}, n_study={self.params.get('n_study')}, "
            f"type='{self.params.get('type', '')}')"
        )
