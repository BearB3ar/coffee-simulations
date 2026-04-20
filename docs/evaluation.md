---
layout: default
title: Development and Evaluation
---

# Development Process and Evaluation

## Development Approach

The model was developed iteratively:

1. Start from a minimal advection-diffusion baseline on a small control volume.
2. Add realism incrementally (network generation quality, geometry constraints, extraction logic, thermal coupling, fines behavior).
3. Tune uncertain approximation parameters using experimental benchmark data.

## Key Design Iterations

- Switched to a watershed-based image-to-network conversion for better structure fidelity.
- Explored multi-solute extraction, then simplified to a unified-solute model for tractability and fit reliability.
- Tested transient solver alternatives; retained numerically stable workflow.
- Replaced simpler size assumptions with mixed lognormal distributions.
- Upgraded extraction from constant-rate to concentration-aware, saturation-limited behavior.
- Added V60 cone geometry and wall-effect porosity adjustments.
- Added dynamic thermal coupling to account for temperature-dependent transport properties.
- Upgraded from single-factor to two-stage extraction approximation.

## Parameter Tuning

A parameter sweep was conducted over key approximation coefficients, including:

- extraction factors
- fast-fraction terms
- saturation concentration terms

Candidate parameter sets were compared against experimental concentration trends, and selected by minimizing a cost function balancing:

- absolute concentration error
- profile-shape agreement
- realistic extraction-yield behavior

## Evaluation Summary

### Strengths

- Captures coupled hydraulic, thermal, and mass transport behavior in a unified framework.
- Supports physically interpretable parameter studies.
- Lower complexity and runtime expectations than very high-fidelity direct numerical approaches.

### Limitations

- Fines migration requires approximate rules rather than direct geometric remeshing.
- Model fidelity depends on tuned empirical coefficients.
- Some microscale phenomena remain outside standard PNM resolution.

## Takeaway

This project demonstrates that PNM can be a useful and computationally practical framework for coffee extraction simulation when paired with carefully chosen approximations and calibration against experiments.
