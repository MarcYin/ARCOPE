"""Example 05: core-only showcase experiment for ARC-SCOPE.

This example assembles a SCOPE-style input stack from synthetic ARC-like
retrieval outputs, a bundled field boundary, bundled local weather, and
computed observation geometry. It then fits a clearly labelled proxy
fluorescence signal to demonstrate the optimisation stack without claiming a
full ``scope-rtm`` run.

Requirements:
    pip install arc-scope

Usage:
    python3 -m arc_scope.experiments.showcase --output-dir docs/assets/showcase
    python3 examples/05_showcase_experiment.py --output-dir docs/assets/showcase
"""

from __future__ import annotations

from arc_scope.experiments.showcase import main


if __name__ == "__main__":
    main()
