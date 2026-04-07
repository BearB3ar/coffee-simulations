import random
import numpy as np
import base_realistic_run 

np.random.seed(0)
random.seed(0)

sim = base_realistic_run.Simulation(
    # Full V60 size is approximately [500,500,410]
    domain_shape=[265,265,205],
    porosity = 0.44,
    temperature = 92,
    particle_size_dist = 'twin_lognormal'
)

# Based on the tube-particle diameter ratio, this is the expected porosity increase over packing unaffected by wall effect
wall_porosity_boost = 1.74/(250/(650e-6/1e-4) +1.14)**2

sim.generate_coffee_bed()
sim.wall_effect(wall_porosity_boost=wall_porosity_boost, decay_width=60)
#sim.plot_coffee_bed()

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
#sim.generate_brewing_animation()
#sim.generate_pressure_animation()
#sim.generate_temperature_animation()
sim.plot_results()
sim.print_statistics()