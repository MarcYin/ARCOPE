---
title: ARC-SCOPE
icon: material/home-analytics
hide:
  - toc
---

# ARC-SCOPE

Bridge ARC crop-state retrieval with SCOPE canopy radiative-transfer simulation for real field-season experiments.

[Open the real full run](full-run-example.md){ .md-button .md-button--primary }
[Quick start](quickstart.md){ .md-button }
[Installation](installation.md){ .md-button }

!!! success "What is real here"
    The primary showcase path is a real ARC retrieval for the bundled Belgium field in 2021, paired with real forcing, real observation geometry, saved SCOPE outputs, and an interactive explorer generated from the run artifacts.

!!! info "What is checked into the site"
    The GitHub Pages bundle is a compact interactive slice of that real run: the explorer payload is downsampled for responsiveness, while the figure suite and CSV inventories still come from the saved experiment outputs.

!!! info "What is lightweight"
    The core showcase keeps a dependency-light path for users who want to understand the bridge, weather alignment, radiation partitioning, and optimization mechanics without installing ARC or SCOPE.

<div class="grid cards" markdown>

-   :material-satellite-variant: __Real Full Run__

    ---

    Walk from real retrieval to simulation, then inspect pixel/date/band/spectrum behaviour in the interactive explorer across reflectance, SIF, and thermal outputs.

    [:octicons-arrow-right-24: Open the full example](full-run-example.md)

-   :material-flask-outline: __Core Showcase__

    ---

    Explore the repo-native walkthrough with bundled weather, synthetic ARC-like inputs, and proxy optimization.

    [:octicons-arrow-right-24: Open the showcase](showcase-experiment.md)

-   :material-cog-outline: __Build Your Own__

    ---

    Start from the bridge and pipeline APIs if you want your own field boundary, weather source, or experiment settings.

    [:octicons-arrow-right-24: Open the quick start](quickstart.md)

</div>

## Why ARC-SCOPE Exists

<div class="grid cards" markdown>

-   :material-database-search: __Model stage 1: retrieval__

    ---

    ARC reconstructs crop biophysical state from Sentinel-2 acquisitions over a real field and date range.

-   :material-weather-partly-cloudy: __Model stage 2: forcing__

    ---

    Weather and observation geometry provide the atmospheric and sun-sensor state needed for physically grounded forward simulation.

-   :material-chart-line-variant: __Model stage 3: simulation__

    ---

    SCOPE turns the retrieved crop state into reflectance, SIF, and thermal-oriented outputs that can be inspected as seasonal trajectories, spectra, and spatial maps.

</div>

## Choose A Path

=== "Heavy runtime"

    Use this when you want the full validated story:

    - real ARC retrieval
    - real forcing and observation geometry
    - saved SCOPE outputs plus interactive explorer
    - figure-rich docs assets and report metadata

    ```bash
    pixi install
    pixi run fetch-scope-upstream
    pixi run check-runtime
    pixi run real-experiment-docs
    ```

=== "Core-only"

    Use this when you want fast onboarding without the heavy dependencies:

    - no ARC install
    - no SCOPE install
    - bundled local weather
    - direct bridge and optimization surfaces

    ```bash
    pip install arc-scope
    python3 -m arc_scope.experiments.showcase --output-dir ./showcase-output
    ```

## What You Get

| Surface | What it shows | Best page |
| --- | --- | --- |
| Real full run | End-to-end ARC retrieval, weather, geometry, saved SCOPE outputs, and interactive explorer | [Real Full Run](full-run-example.md) |
| Core showcase | Dependency-light walkthrough of the bridge and optimization surfaces | [Core Showcase](showcase-experiment.md) |
| Step-by-step usage | Lower-level bridge and pipeline usage patterns | [Quick Start](quickstart.md) |
| Runtime setup | Installation routes for ARC, SCOPE, GDAL, and ERA5 access | [Installation](installation.md) |

## At A Glance

<div class="grid cards" markdown>

-   __Primary validated scenario__

    ---

    Belgium/Flanders test field, wheat crop type, 2021 growing season, real ARC retrieval, then `reflectance`, `fluorescence`, and `thermal` outputs with an explorer generated from the run bundle.

-   __Checked-in docs bundle__

    ---

    The docs ship the generated figure set and a compact explorer payload from the real run, so GitHub Pages stays small and fast without pretending the browser layer is the full-resolution artifact.

-   __Material site features__

    ---

    This site uses Material navigation, buttons, tabs, icons, admonitions, and figure galleries instead of plain Markdown-only pages.

</div>
