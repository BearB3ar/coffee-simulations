---
layout: default
title: Coffee Extraction Simulation
---

# Coffee Extraction Simulation with a Pore Network Model

This site presents my CDE2701A aspirational project: building and evaluating a pore network model (PNM) workflow for simulating coffee extraction, with a focus on V60-style pourover brewing.

## Quick Links

- [Project Overview](./overview.md)
- [Implementation and Methodology](./methodology.md)
- [Development Process and Evaluation](./evaluation.md)
- [Personal Reflection](./reflection.md)
- [References](./references.md)

## At a Glance

- **Core question:** Can a PNM-based approach capture coffee extraction behavior with useful accuracy at lower computational cost?
- **Simulation stack:** Python, OpenPNM, and PoreSpy.
- **Physics modeled:** Hydraulic flow, thermal transport, solute extraction and transport, and fines migration effects.
- **Target brew context:** V60 pourover geometry with configurable brew parameters.

## Why This Project Matters

Coffee extraction is a porous-media transport problem with strong practical interest and challenging multiscale physics. Existing simulation approaches can be computationally intensive or highly specialized. This project explores whether a simpler and faster PNM framework can still provide meaningful predictive power for brewing analysis and parameter tuning.

## Repository

- Source code and project files: [github.com/BearB3ar/coffee-simulations](https://github.com/BearB3ar/coffee-simulations)
