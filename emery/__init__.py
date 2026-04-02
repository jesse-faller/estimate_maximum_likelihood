"""
emery – Maximum likelihood estimation of accuracy statistics for multiple
measurement methods when no gold standard is available.

Python port of the R package `emery` (Corie Drake).

Currently supported: **binary** methods (sensitivity, specificity, prevalence).

Quick start
-----------
>>> from emery import generate_multimethod_data, estimate_ML
>>>
>>> sim = generate_multimethod_data(
...     "binary",
...     n_obs=75,
...     n_method=4,
...     se=[0.87, 0.92, 0.79, 0.95],
...     sp=[0.85, 0.93, 0.94, 0.80],
... )
>>> result = estimate_ML("binary", data=sim["generated_data"])
>>> print(result)
"""

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
from .core import (
    aggregate_boot_ML,
    boot_ML,
    estimate_ML,
    generate_multimethod_data,
    plot_ML,
    plot_boot_ML,
    pollinate_ML,
    random_start,
)
from .ordinal import (
    estimate_ML_ordinal,
    generate_multimethod_ordinal,
    plot_ML_ordinal,
    pollinate_ML_ordinal,
)
from .utils import (
    censor_data,
    define_disease_state,
    name_thing,
    unique_obs_summary,
)

__version__ = "0.1.0"

__all__ = [
    # Classes
    "MultiMethodMLEstimate",
    "BootML",
    # Dispatch functions
    "generate_multimethod_data",
    "estimate_ML",
    "pollinate_ML",
    "random_start",
    "plot_ML",
    "boot_ML",
    "aggregate_boot_ML",
    "plot_boot_ML",
    # Binary-specific
    "generate_multimethod_binary",
    "estimate_ML_binary",
    "pollinate_ML_binary",
    "random_start_binary",
    "plot_ML_binary",
    "bin_auc",
    # Ordinal-specific
    "generate_multimethod_ordinal",
    "estimate_ML_ordinal",
    "pollinate_ML_ordinal",
    "plot_ML_ordinal",
    # Continuous-specific
    "generate_multimethod_continuous",
    "estimate_ML_continuous",
    "pollinate_ML_continuous",
    "plot_ML_continuous",
    # Utilities
    "name_thing",
    "define_disease_state",
    "censor_data",
    "unique_obs_summary",
]
