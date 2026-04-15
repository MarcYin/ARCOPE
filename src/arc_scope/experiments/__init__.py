"""Self-contained showcase experiments for ARC-SCOPE."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ShowcaseExperimentResult",
    "ShowcaseSummary",
    "run_showcase_experiment",
    "write_showcase_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from arc_scope.experiments import showcase

        return getattr(showcase, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
