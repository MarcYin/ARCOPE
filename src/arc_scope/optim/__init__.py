"""Parameter optimization extension for SCOPE simulations.

Provides utilities for optimising SCOPE parameters (e.g., fluorescence
quantum efficiency, soil resistances) against observed SIF, thermal, or
flux data.  ARC-retrieved biophysical parameters are held fixed while
the optimiser adjusts parameters that cannot be retrieved from reflectance.
"""

from arc_scope.optim.parameters import ParameterSet, ParameterSpec

__all__ = ["ParameterSet", "ParameterSpec"]
