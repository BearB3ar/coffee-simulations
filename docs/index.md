---
layout: home
title: Coffee Extraction Simulation
---

## Navigate the Project

<div class="card-grid">
  <div class="card">
    <h3><a href="./overview">Project Overview</a></h3>
    <p>Research question, background, scope, and objectives.</p>
  </div>
  <div class="card">
    <h3><a href="./methodology">Methodology</a></h3>
    <p>Simulation design, physics models, and the time-step brew workflow.</p>
  </div>
  <div class="card">
    <h3><a href="./evaluation">Development &amp; Evaluation</a></h3>
    <p>Iterative design choices, parameter tuning, and evaluation outcomes.</p>
  </div>
  <div class="card">
    <h3><a href="./reflection">Personal Reflection</a></h3>
    <p>Key lessons learned and directions for future work.</p>
  </div>
  <div class="card">
    <h3><a href="./references">References</a></h3>
    <p>Full bibliography of literature cited throughout the project.</p>
  </div>
</div>

## At a Glance

<div class="glance-strip">
  <div class="glance-item">
    <span class="glance-label">Core Question</span>
    <span class="glance-value">Can PNM accurately simulate coffee extraction?</span>
  </div>
  <div class="glance-item">
    <span class="glance-label">Language</span>
    <span class="glance-value">Python</span>
  </div>
  <div class="glance-item">
    <span class="glance-label">Libraries</span>
    <span class="glance-value">OpenPNM &amp; PoreSpy</span>
  </div>
  <div class="glance-item">
    <span class="glance-label">Brew Target</span>
    <span class="glance-value">V60 Pourover</span>
  </div>
  <div class="glance-item">
    <span class="glance-label">Physics</span>
    <span class="glance-value">Hydraulic · Thermal · Solute · Fines</span>
  </div>
</div>

## Why This Project Matters

Coffee extraction is fundamentally a porous-media transport problem. Existing simulation approaches (LBM, SPH, DPNS) are powerful but computationally intensive. This project explores whether a **Pore Network Model** — a simpler, lower-cost framework — can still deliver meaningful predictive power for brewing analysis and parameter optimisation.

> Can PNM provide enough fidelity for coffee extraction simulation while remaining computationally efficient?

The results are generalised across brewing methods but focused on the V60 pourover geometry.
