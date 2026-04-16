"""Self-contained showcase experiments for ARC-SCOPE."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DualWorkflowExperimentResult",
    "ShowcaseExperimentResult",
    "ShowcaseSummary",
    "WorkflowExperimentResult",
    "run_full_experiment",
    "run_dual_workflow_experiment",
    "run_showcase_experiment",
    "write_full_run_artifacts",
    "write_dual_workflow_artifacts",
    "write_showcase_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        if name in {
            "DualWorkflowExperimentResult",
            "WorkflowExperimentResult",
            "run_full_experiment",
            "run_dual_workflow_experiment",
            "write_full_run_artifacts",
            "write_dual_workflow_artifacts",
        }:
            from arc_scope.experiments import dual_workflow

            if name == "WorkflowExperimentResult":
                return dual_workflow.DualWorkflowExperimentResult
            if name == "run_full_experiment":
                return dual_workflow.run_dual_workflow_experiment
            if name == "write_full_run_artifacts":
                return dual_workflow.write_dual_workflow_artifacts
            return getattr(dual_workflow, name)
        from arc_scope.experiments import showcase

        return getattr(showcase, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
