"""Top‑level package for the QAOA research codebase.

This package groups together modules for graph generation, QAOA circuit
construction and benchmarking.  See the README in the repository root
for an overview and usage examples.
"""

from . import graphs  # noqa: F401
from . import qaoa   # noqa: F401
from . import bench  # noqa: F401

__all__ = ["graphs", "qaoa", "bench"]