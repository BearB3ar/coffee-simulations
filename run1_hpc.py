import base_realistic_run 
import pyvista as pv
pv.start_xvfb() # Start virtual framebuffer for pyvista rendering in HPC environments

sim = base_realistic_run.Simulation(
    # Full V60 size is approximately [500,500,410]
    domain_shape=[577,577,500],
    porosity = 0.44,
    temperature = 92
)

# Based on the tube-particle diameter ratio, this is the expected porosity increase over packing unaffected by wall effect
wall_porosity_boost = 1.74/(1e-2/325e-6 +1.14)**2

sim.generate_coffee_bed()
sim.wall_effect(wall_porosity_boost=wall_porosity_boost, decay_width=60)
#sim.plot_coffee_bed()
#sim.plot_size_distribution_line()

sim.extract_network()
sim.add_geometry_models()
sim.phase()
sim.add_physics_models()

sim.brew(
    brew_time = 120,
    pour_rate = 2,
    time_steps = 120,
    shrink_factor = 1, # Choose 1 to neglect swelling effects
    fines_rng_seed = 0
)
sim.generate_brewing_animation()
sim.generate_pressure_animation()
sim.generate_temperature_animation()
sim.plot_results()
sim.print_statistics()