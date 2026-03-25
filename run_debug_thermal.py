import base_realistic_run
import numpy as np


sim = base_realistic_run.Simulation(
    domain_shape=[265, 265, 205],
    porosity=0.44,
    temperature=95,
    particle_size_dist="twin_lognormal",
)

# Based on the tube-particle diameter ratio, this is the expected porosity increase over packing
# unaffected by wall effect (same as run1.py).
wall_porosity_boost = 1.74 / (250 / (650e-6 / 1e-4) + 1.14) ** 2

sim.generate_coffee_bed()
sim.wall_effect(wall_porosity_boost=wall_porosity_boost, decay_width=60)

sim.extract_network()
sim.add_geometry_models()
sim.phase()
sim.add_physics_models()

# Fewer time steps: we only need early/mid/late behavior for thermal diagnostics.
sim.brew(brew_time=120, pour_rate=2, time_steps=6, shrink_factor=1)

# Summarize bottom region at the last thermal step
coords_z = sim.pn["pore.coords"][:, 2]
bottom_mask = coords_z <= float(np.quantile(coords_z, 0.2))

T_uncl_last = sim.temperature_fields["unclipped"][-1]
T_cl_last = sim.temperature_fields["clipped"][-1]

print("=== Bottom thermal summary (last step) ===")
print(f"mean_all_uncl={np.mean(T_uncl_last):.2f}C mean_all_cl={np.mean(T_cl_last):.2f}C")
print(f"mean_bottom_uncl={np.mean(T_uncl_last[bottom_mask]):.2f}C min_bottom_uncl={np.min(T_uncl_last[bottom_mask]):.2f}C")
print(f"mean_bottom_cl={np.mean(T_cl_last[bottom_mask]):.2f}C min_bottom_cl={np.min(T_cl_last[bottom_mask]):.2f}C")

