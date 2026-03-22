import base_realistic_run 

sim = base_realistic_run.Simulation(
    # Full V60 size is [1000,1000,820]
    domain_shape=[250,250,205],
    porosity = 0.44,
    temperature = 95,
    particle_size_dist = 'twin_lognormal'
)

sim.generate_coffee_bed()
sim.wall_effect(wall_porosity_boost=0.5, decay_width=30)
#sim.plot_coffee_bed()

sim.extract_network()
sim.add_geometry_models()
sim.phase()
sim.add_physics_models()

sim.brew(
    brew_time = 120,
    pour_rate = 2,
    time_steps = 120,
    shrink_factor = 1 # Choose 1 to neglect swelling effects
)
sim.generate_brewing_animation()
sim.plot_results()
sim.print_statistics()
#print(sim.mass_balance())