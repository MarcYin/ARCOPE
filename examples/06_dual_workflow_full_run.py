"""Example 06: Real ARC -> SCOPE full run.

This thin wrapper delegates to the packaged CLI/module entry point so the
recommended command also works from an installed package:

    python -m arc_scope.experiments.dual_workflow --help

The module path retains the historical ``dual_workflow`` name for
compatibility, but the documented default run is the real ARC retrieval plus
SCOPE reflectance experiment:

    python -m arc_scope.experiments.dual_workflow \
        --scope-root-path /path/to/SCOPE \
        --dtype float32 \
        --output-dir ./full-run-output
"""

from __future__ import annotations

from arc_scope.experiments.dual_workflow import main


if __name__ == "__main__":
    main()
