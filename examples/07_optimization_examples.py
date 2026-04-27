"""Example 07: Generate optimisation guide artifacts.

This script runs deterministic proxy optimisation examples for SIF, thermal,
and coupled energy-balance fitting. It exercises ARC-SCOPE's optimisation
machinery without requiring live ARC retrieval or scope-rtm runtime assets.
"""

from arc_scope.experiments.optimization_examples import main


if __name__ == "__main__":
    main()
