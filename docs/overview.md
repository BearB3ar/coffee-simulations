---
layout: default
title: Project Overview
---

# Project Overview

## Background

Coffee quality is commonly linked to measurable extraction metrics, especially:

- **Total Dissolved Solids (TDS):** concentration of dissolved coffee compounds.
- **Extraction Yield:** mass fraction of dry coffee extracted into the beverage.

Traditional studies and brewing control tools typically optimize these quantities empirically. This project instead investigates the underlying transport physics using pore-scale simulation.

## Problem Statement

Pore Network Modelling (PNM) is widely used for flow in porous media, but has not been widely applied to coffee extraction workflows compared with approaches such as lattice Boltzmann, smoothed particle hydrodynamics, and direct pore-scale numerical simulations.

The project asks:

> Can PNM provide enough fidelity for coffee extraction simulation while remaining computationally efficient?

## Scope

- **Brewing method focus:** V60 pourover
- **Model dimensionality:** 3D pore-throat network
- **Primary outputs:** pressure, flow, temperature, and solute concentration over time
- **Secondary mechanism:** fines migration and clogging effects

Although developed around pourover geometry, the framework is designed to be extensible to other brew styles.

## Objectives

- Build an end-to-end pore-network coffee extraction simulator.
- Capture key physical effects relevant to brewing outcomes.
- Tune approximation parameters against available experimental concentration data.
- Assess strengths and limitations of PNM for this application.
