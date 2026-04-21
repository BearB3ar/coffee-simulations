import random
import numpy as np
import base_realistic_run 
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

def plot_size_distribution_line(simulation, n_bins=40):
    """Plot the particle size distribution used to generate the coffee bed."""

    def _smooth_curve(values, sigma_bins=0.5):
        radius = int(np.ceil(4 * sigma_bins))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(values, kernel, mode="same")

    x_axis, probs = simulation.get_particle_size_distribution()
    size_um = x_axis * 100.0

    # Weight by sphere volume (∝ r³) to convert number distribution → volume fraction.
    # This is purely analytical; no 3-D image data is needed.
    volume_weights = probs * x_axis ** 3
    volume_frac = volume_weights / volume_weights.sum()
    smoothed = _smooth_curve(volume_frac)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(size_um, 100.0 * smoothed, linewidth=2.5, color="blue", label="Particles")
    ax.set_xscale("log")
    ax.set_xlabel("Particle diameter (microns)")
    ax.set_ylabel("Volume fraction (%)")
    ax.set_title("Coffee particle size distribution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

sim = base_realistic_run.Simulation(
    # Full V60 size is approximately [500,500,410]
    domain_shape=[300,300,235],
    porosity = 0.44,
    temperature = 92,
    particle_size_dist = 'twin_lognormal'
)

# Based on the tube-particle diameter ratio, this is the expected porosity increase over packing unaffected by wall effect
wall_porosity_boost = 1.74/(1e-2/325e-6 +1.14)**2

sim.generate_coffee_bed()
sim.wall_effect(wall_porosity_boost=wall_porosity_boost, decay_width=60)
sim.plot_coffee_bed()
plot_size_distribution_line(sim)

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