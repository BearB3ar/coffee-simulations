import random
import numpy as np
import base_realistic_run 
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)

def plot_size_distribution_line(simulation, n_bins=40):
    """Plot smooth pore/throat size distribution using volume-fraction weighting."""
    pore_diam_um = simulation.pn['pore.diameter'] * 1e6
    throat_diam_um = simulation.pn['throat.diameter'] * 1e6
    pore_volumes = simulation.pn['pore.volume']
    throat_volumes = simulation.pn['throat.volume']

    positive_diams = np.concatenate([
        pore_diam_um[pore_diam_um > 0],
        throat_diam_um[throat_diam_um > 0],
    ])
    if positive_diams.size == 0:
        return

    min_d = np.min(positive_diams)
    max_d = np.max(positive_diams)
    if min_d == max_d:
        # Keep log-space bins valid even if all diameters collapse to one value.
        min_d *= 0.9
        max_d *= 1.1

    bins = np.logspace(np.log10(min_d), np.log10(max_d), n_bins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])

    pore_hist, _ = np.histogram(pore_diam_um, bins=bins, weights=pore_volumes)
    throat_hist, _ = np.histogram(throat_diam_um, bins=bins, weights=throat_volumes)

    pore_total = np.sum(pore_hist)
    throat_total = np.sum(throat_hist)
    pore_frac = (pore_hist / pore_total) if pore_total > 0 else np.zeros_like(pore_hist)
    throat_frac = (throat_hist / throat_total) if throat_total > 0 else np.zeros_like(throat_hist)

    # Smooth in log-space with a Gaussian kernel to get a continuous line profile.
    def _smooth_curve(values, sigma_bins=1.6):
        radius = int(np.ceil(4 * sigma_bins))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(values, kernel, mode="same")

    pore_smooth = _smooth_curve(pore_frac)
    throat_smooth = _smooth_curve(throat_frac)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, 100.0 * pore_smooth, linewidth=2.5, color="blue", label="Pores")
    #ax.plot(centers, 100.0 * throat_smooth, linewidth=2.5, color="red", label="Throats")
    ax.set_xscale("log")
    ax.set_xlabel("Diameter (microns)")
    ax.set_ylabel("Volume fraction (%)")
    ax.set_title("Pore and throat size distribution (volume-weighted)")
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
wall_porosity_boost = 1.74/(300/(650e-6/1e-4) +1.14)**2

sim.generate_coffee_bed()
sim.wall_effect(wall_porosity_boost=wall_porosity_boost, decay_width=60)
#sim.plot_coffee_bed()

sim.extract_network()
sim.add_geometry_models()
plot_size_distribution_line(sim)
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