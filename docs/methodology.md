---
layout: default
title: Implementation and Methodology
---

# Implementation and Methodology

## Tooling

- **Language:** Python
- **Libraries:** OpenPNM (physics models and solvers), PoreSpy (network generation)

The simulation runs in discrete time steps and solves coupled transport processes on a generated pore-throat network.

## 1) Initialization

The model starts from user-defined geometry and porosity inputs:

- 3D domain shape in voxels
- target porosity
- voxel scale (10 microns per pixel)

### Particle and pore structure assumptions

- Coffee grind sizes are represented using a bimodal mixture of lognormal distributions.
- A representative setup uses target peaks around coarse and fine regimes (for example, approximately 325 microns and 50 microns) with weighted mixing.
- The resulting particle packing is converted into pore space and then transformed into a pore network via a watershed approach.

### Geometry shaping

- The bed is trimmed to a conical geometry approximating a V60 brewer.
- A wall-effect porosity boost is applied near boundaries with exponential decay into the bed interior.

## 2) Physics Setup

After network generation, geometric and transport properties are assigned:

- pore/throat dimensions, areas, and volumes
- hydraulic, diffusive, and thermal conductance factors
- temperature-dependent fluid properties (viscosity, diffusivity, conductivity)

## 3) Brew Time-Stepping Workflow

For each time step, the model performs:

1. **Hydraulic solve:** pressure and velocity field.
2. **Thermal solve:** temperature distribution update.
3. **Solute introduction:** extraction into pores based on local conditions and remaining extractable mass.
4. **Solute transport solve:** advection-diffusion update of concentration.
5. **Fines migration update:** stochastic fines movement and clogging adjustments.

## Hydraulic Flow Model

- Inlet pressure is computed using hydrostatic and dynamic components from pour conditions.
- Outlet is set with a Dirichlet-type pressure boundary.
- Stokes-flow assumptions are used for slow, viscous-dominated bed flow.

## Thermal Transport

- Uses inlet-water and initial-bed temperature conditions.
- Solves transient thermal advection-diffusion over the same network used for hydraulic transport.

## Solute Extraction and Transport

- Solute is represented as a unified compound for practical tractability.
- Extraction is modeled with concentration-dependent and saturation-limited behavior.
- Transport is solved using advection-diffusion across the network.

## Fines Migration Approximation

Direct pore restructuring for each fines movement is computationally expensive in standard PNM workflows, so this project uses a probabilistic fines migration/clogging approximation:

- fines move stochastically with flow and local concentration effects
- small throats preferentially retain fines
- throat conductance is reduced as retained fines accumulate
- heavily depleted elite source pores can transition to higher local conductance states

This introduces physically motivated resistance evolution without full geometry regeneration each time step.
