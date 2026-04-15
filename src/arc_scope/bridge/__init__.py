"""ARC-to-SCOPE parameter bridge: maps ARC retrieval outputs to SCOPE input format."""

from arc_scope.bridge.convert import arc_arrays_to_scope_inputs, arc_npz_to_scope_inputs
from arc_scope.bridge.parameter_map import (
    ARC_BIO_INDICES,
    ARC_BIO_NAMES,
    ARC_SOIL_INDICES,
    BIO_BANDS,
    BIO_SCALES,
    SCALE_BANDS,
)

__all__ = [
    "arc_arrays_to_scope_inputs",
    "arc_npz_to_scope_inputs",
    "ARC_BIO_INDICES",
    "ARC_BIO_NAMES",
    "ARC_SOIL_INDICES",
    "BIO_BANDS",
    "BIO_SCALES",
    "SCALE_BANDS",
]
