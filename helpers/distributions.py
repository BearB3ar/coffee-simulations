import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.csgraph import connected_components
import scipy.stats
from scipy.interpolate import griddata
from scipy.sparse.linalg import spsolve as scipy_spsolve
from pypardiso import spsolve as pypardiso_spsolve
from scipy.sparse import spdiags
from scipy.sparse import diags
import scipy.ndimage as spim
import math

"""Assume SI units unless otherwise specified"""

class Simulation:
    viscosity_ref_table = {20:0.0010016, # Units of C: Pa s
                           25:0.0008900,
                           30:0.0007972,
                           40:0.0006527,
                           50:0.0005465,
                           60:0.0004660,
                           70:0.0004035,
                           80:0.0003540,
                           90:0.0003142,
                           100:0.0002816}  
    
    coarse_grind = {
    1.0: 0.1,
    2: 0.15,
    5: 0.25,
    10: 0.4,
    15: 0.6,
    25: 0.8,
    40: 0.7,
    60: 0.55,
    80: 0.65,
    100: 0.75,
    150: 0.9,
    200: 0.7,
    250: 1.0,
    300: 1.8,
    400: 3.5,
    500: 5.5,
    600: 7.2,
    700: 8.6,
    800: 9.4,
    900: 9.7,
    1000: 9.7,
    1100: 9.2,
    1200: 8.0,
    1300: 6.0,
    1400: 4.0,
    1500: 2.5,
    1700: 1.0,
    2000: 0.0
    }

    fine_grind = {
    0.5: 0.05,
    1: 0.2,
    5: 0.4,
    10: 0.6,
    15: 0.95,
    20: 1.25,
    25: 1.4,
    40: 1.2,
    60: 1.05,
    70: 1.0,
    100: 0.8,
    150: 0.9,
    200: 1.6,
    250: 2.8,
    300: 4.2,
    400: 6.8,
    500: 7.8,
    600: 7.5,
    700: 6.2,
    800: 4.8,
    1000: 2.2,
    1200: 0.8,
    1500: 0.05,
    1800: 0.0
    }

    def __init__(self, domain_shape=[300,300,300], porosity = 0.46, temperature = 95, solute_classes=None):
        self.shape = domain_shape # Cuboidal control volume, trimming to cone shape done later (units of voxels)
        self.porosity = porosity # Target porosity, actual porosity depends on porespy image generation + wall effect 
        self.temperature = temperature # Initial temperature
        self.time_steps = [] # For results graphs
        self.concentrations = { # Stores concentration of the entire bed at every time step
            'acids': [],
        }
        self.temperature_variation = { # Stores mean temperature of the bed at every time step, before and after clipping
            "final": [],
            "unclipped": []
        }
        # Stores full pore temperature fields for debugging the T solver.
        # Each entry is an array of shape (Np,) in the same pore ordering as `self.pn`.
        self.temperature_fields = {
            "clipped": [],
            "unclipped": []
        }
        self.pressures = []  # Filled in brew(): one snapshot per step after fines + Stokes (see generate_pressure_animation)
        self.velocities = []
        self.total_extracted = 0.0 # Cumulative measure for how much solute leaves outlet_pores
        self.total_extracted_by_solute = {}
        self.extracted_mass_history_by_solute = {}
        self.initial_extractable_mass_by_solute = {}
        self.yield_by_solute = {}
        if solute_classes is None:
            self.solute_classes = {
                # TODO: Tune amount of coffee present initially and other parameters
                'acids': {
                    'k_fast': 0.5,
                    'k_slow': 0.02,
                    'f_fast': 0.25, # Fraction of initial mass that is in the fast-extracting class vs slow-extracting class; this is a simple way to capture the common observation of a fast initial extraction followed by slower extraction later on
                    'concentration': 8.4e3,
                    'c_sat': 30,
                }, # Target initial mass is 2.041e-3
            }
        else:
            self.solute_classes = solute_classes

    def get_particle_size_distribution(self):
        x_axis = np.arange(0.05, 100, 0.5)  # voxels

        sizes_um_coarse = np.array(sorted(self.coarse_grind.keys()), dtype=float)
        vol_fracs_coarse = np.array([self.coarse_grind[s] for s in sizes_um_coarse], dtype=float)

        sizes_um_fine = np.array(sorted(self.fine_grind.keys()), dtype=float)
        vol_fracs_fine = np.array([self.fine_grind[s] for s in sizes_um_fine], dtype=float)

        # Convert diameters from µm to voxels (1 voxel = 1e-4 m = 100 µm)
        sizes_vox_coarse = sizes_um_coarse / 100.0
        sizes_vox_fine = sizes_um_fine / 100.0

        # This preserves the true shape of the mass distribution before the d³ transformation.
        interp_vol_fracs_coarse = np.interp(x_axis, sizes_vox_coarse, vol_fracs_coarse, left=0.0, right=0.0)
        interp_vol_fracs_fine = np.interp(x_axis, sizes_vox_fine, vol_fracs_fine, left=0.0, right=0.0)

        # V_f(d) ∝ N(d)·d³  ⟹  N(d) ∝ V_f(d)/d³
        number_freq_coarse = np.zeros_like(interp_vol_fracs_coarse)
        valid_mask = x_axis > 0
        number_freq_coarse[valid_mask] = interp_vol_fracs_coarse[valid_mask] / (x_axis[valid_mask] ** 3)

        number_freq_fine = np.zeros_like(interp_vol_fracs_fine)
        number_freq_fine[valid_mask] = interp_vol_fracs_fine[valid_mask] / (x_axis[valid_mask] ** 3)

        # Normalize to create a valid Probability Mass Function (PMF) for scipy.stats
        probs_coarse = number_freq_coarse / number_freq_coarse.sum()
        probs_fine = number_freq_fine / number_freq_fine.sum()
        return x_axis, probs_coarse, probs_fine

    def generate_coffee_bed(self):
        shape = self.shape
        x_axis, probs_coarse, probs_fine = self.get_particle_size_distribution()
        custom_dist_object_coarse = scipy.stats.rv_discrete(name='coffee_dist', values=(x_axis, probs_coarse)) # rv_discrete creates a discretised random variable function
        custom_dist_object_fine = scipy.stats.rv_discrete(name='coffee_dist', values=(x_axis, probs_fine)) # rv_discrete creates a discretised random variable function

        # porespy generates an image of pores and throats with distances between them based on the distribution of coffee particles desired and minimum radius of 0.05 voxels
        im_coarse = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.05, porosity=self.porosity, dist=custom_dist_object_coarse)
        im_fine = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.05, porosity=self.porosity, dist=custom_dist_object_fine)

        # Trimming to cone geometry
        x,y,z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        center_y, center_x = shape[0] // 2, shape[1] // 2
        radius_at_z = -(z + 20) * np.tan(np.radians(30))
        cone_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius_at_z**2
        self.cone_mask = cone_mask # Needed for porosity calculation at the end
        im_coarse = im_coarse & cone_mask
        im_fine = im_fine & cone_mask

        self.im_coarse = im_coarse
        self.im_fine = im_fine
        # Default mask for porosity accounting when cone trimming or wall effect is disabled.
        #self.cone_mask = np.ones_like(im, dtype=bool)

    def plot_size_distribution_line(self, n_bins=40):
        """Plot the particle size distribution used to generate the coffee bed."""

        def _smooth_curve(values, sigma_bins=0.5):
            radius = int(np.ceil(4 * sigma_bins))
            x = np.arange(-radius, radius + 1, dtype=float)
            kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
            kernel /= np.sum(kernel)
            return np.convolve(values, kernel, mode="same")

        x_axis, probs_coarse, probs_fine = self.get_particle_size_distribution()
        size_um = x_axis * 100.0

        # Weight by sphere volume (∝ r³) to convert number distribution → volume fraction.
        # This is purely analytical; no 3-D image data is needed.
        volume_weights_coarse = probs_coarse * x_axis ** 3
        volume_weights_fine = probs_fine * x_axis ** 3
        original_peak_coarse = max(self.coarse_grind.values())
        volume_frac_scaled_coarse = (volume_weights_coarse / np.max(volume_weights_coarse)) * original_peak_coarse
        original_peak_fine = max(self.fine_grind.values())
        volume_frac_scaled_fine = (volume_weights_fine / np.max(volume_weights_fine)) * original_peak_fine

        smoothed_coarse = _smooth_curve(volume_frac_scaled_coarse)
        smoothed_fine = _smooth_curve(volume_frac_scaled_fine)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(size_um, smoothed_coarse, linewidth=2.5, color="red", label="Coarse")
        ax.plot(size_um, smoothed_fine, linewidth=2.5, color="blue", label="Fine")
        ax.set_xscale("log")
        ax.set_xlabel("Particle diameter (microns)")
        ax.set_ylabel("Volume fraction (%)")
        ax.set_title("Coffee particle size distribution")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def wall_effect(self, wall_porosity_boost=0.2, decay_width=10):
        im_coarse = self.im_coarse  # Use the coarse image for wall effect calculation
        im_fine = self.im_fine
        cone_mask_coarse = getattr(self, "cone_mask", np.ones_like(im_coarse, dtype=bool))
        cone_mask_fine = getattr(self, "cone_mask", np.ones_like(im_fine, dtype=bool))

        # Solid grounds in V60 cone
        grounds_coarse = (im_coarse == 0) & cone_mask_coarse
        grounds_fine = (im_fine == 0) & cone_mask_fine

        # Distance measurer from cone_mask boundary inwards (spim is an advanced multidimensional image processor)
        dt_coarse = spim.distance_transform_edt(cone_mask_coarse)
        dt_fine = spim.distance_transform_edt(cone_mask_fine)

        # Find spheres using spim logic rather than searching by voxels and identify their centroids
        labels_coarse, n_spheres_coarse = spim.label(grounds_coarse)
        centroids_coarse = spim.center_of_mass(grounds_coarse, labels_coarse, range(1, n_spheres_coarse+1))
        labels_fine, n_spheres_fine = spim.label(grounds_fine)
        centroids_fine = spim.center_of_mass(grounds_fine, labels_fine, range(1, n_spheres_fine+1))

        new_im_coarse = im_coarse.copy()
        new_im_fine = im_fine.copy()

        for i, center in enumerate(centroids_coarse):
            # z,y,x is order specified by spim
            z, y, x = int(center[0]), int(center[1]), int(center[2])
            
            # Find the distance to cone_mask boundary
            dist = dt_coarse[z,y,x]

            # Dynamic probability of removal based on exponentially decaying distance from cone_mask
            p_remove = wall_porosity_boost * np.exp(-dist / decay_width)

            # Remove if randomly generated probability falls within probability of removal
            if np.random.random() < p_remove:
                new_im_coarse[labels_coarse == (i+1)] = 1

        for i, center in enumerate(centroids_fine):
            # z,y,x is order specified by spim
            z, y, x = int(center[0]), int(center[1]), int(center[2])
            
            # Find the distance to cone_mask boundary
            dist = dt_fine[z,y,x]

            # Dynamic probability of removal based on exponentially decaying distance from cone_mask
            p_remove = wall_porosity_boost * np.exp(-dist / decay_width)

            # Remove if randomly generated probability falls within probability of removal
            if np.random.random() < p_remove:
                new_im_fine[labels_fine == (i+1)] = 1

        self.im_coarse = new_im_coarse
        self.im_fine = new_im_fine

sim = Simulation(
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
sim.plot_size_distribution_line()