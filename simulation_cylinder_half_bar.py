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
    
    grind_size_dist = {
        0.3: 0.2,
        0.4: 0.2,
        0.5: 0.2,
        0.6: 0.2,
        0.7: 0.2,
        0.8: 0.2,
        0.9: 0.2,
        1.0: 0.2,
        2: 0.3,
        3: 0.4,
        5: 0.5,
        6: 0.5,
        7: 0.5,
        8: 0.6,
        9: 0.6,
        10: 0.7,
        15: 1.0,
        20: 1.4,
        25: 1.5,
        30: 1.4,
        40: 1.3,
        50: 1.1,
        60: 1.0,
        70: 1.0,
        80: 0.9,
        90: 0.8,
        100: 0.9,
        150: 0.7,
        200: 1.6,
        250: 2.3,
        300: 4.5,
        400: 7.0,
        500: 7.7,
        600: 7.4,
        700: 6.9,
        800: 5.5,
        900: 4.3,
        1000: 3.6,
        1250: 2.0,
        1500: 0.1 
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
        self.total_extracted = 0.0 # Cumulative measure for how much solute leaves outlet_pores
        self.total_extracted_by_solute = {}
        self.extracted_mass_history_by_solute = {}
        self.initial_extractable_mass_by_solute = {}
        self.yield_by_solute = {}
        if solute_classes is None:
            self.solute_classes = {
                # TODO: Tune amount of coffee present initially and other parameters
                'acids': {
                    'k_fast': 30,
                    'k_slow': 0.108429,
                    'f_fast': 0.837191, # Fraction of initial mass that is in the fast-extracting class vs slow-extracting class; this is a simple way to capture the common observation of a fast initial extraction followed by slower extraction later on
                    'concentration': 9.5e3,
                    'c_sat': 15,
                }, # Target initial mass is 2.041e-3
            }
        else:
            self.solute_classes = solute_classes

    def _water_viscosity_from_temp_c(self, temp_c):
        # Linear interpolation over tabulated viscosity-vs-temperature data
        temps = np.array(sorted(self.viscosity_ref_table.keys()), dtype=float)
        mus = np.array([self.viscosity_ref_table[t] for t in temps], dtype=float)
        temp_arr = np.asarray(temp_c, dtype=float)
        return np.interp(temp_arr, temps, mus, left=mus[0], right=mus[-1])

    def get_particle_size_distribution(self):
        x_axis = np.arange(0.05, 100, 0.5)  # voxels

        sizes_um = np.array(sorted(self.grind_size_dist.keys()), dtype=float)
        vol_fracs = np.array([self.grind_size_dist[s] for s in sizes_um], dtype=float)

        # Convert diameters from µm to voxels (1 voxel = 1e-4 m = 100 µm)
        sizes_vox = sizes_um / 100.0

        # Volume fraction → number frequency: V_f(d) ∝ N(d)·d³  ⟹  N(d) ∝ V_f(d)/d³
        number_freq = vol_fracs / (4 * np.pi * (sizes_vox ** 3)/3)
        number_freq /= number_freq.sum()

        # Interpolate onto x_axis; zero outside the sampled range
        probs = np.interp(x_axis, sizes_vox, number_freq, left=0.0, right=0.0)
        probs = probs / probs.sum()
        return x_axis, probs

    def generate_coffee_bed(self):
        shape = self.shape
        x_axis, probs = self.get_particle_size_distribution()
        custom_dist_object = scipy.stats.rv_discrete(name='coffee_dist', values=(x_axis, probs)) # rv_discrete creates a discretised random variable function

        # porespy generates an image of pores and throats with distances between them based on the distribution of coffee particles desired and minimum radius of 0.05 voxels
        im = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.05, porosity=self.porosity, dist=custom_dist_object) 

        """# Trimming to cone geometry
        x,y,z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        center_y, center_x = shape[0] // 2, shape[1] // 2
        radius_at_z = -(z + 20) * np.tan(np.radians(30))
        cone_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius_at_z**2
        self.cone_mask = cone_mask # Needed for porosity calculation at the end
        im = im & cone_mask"""
        
        self.im = im
        # Default mask for porosity accounting when cone trimming or wall effect is disabled.
        self.cone_mask = np.ones_like(im, dtype=bool)
    
    def wall_effect(self, wall_porosity_boost=0.2, decay_width=10):
        im = self.im
        cone_mask = getattr(self, "cone_mask", np.ones_like(im, dtype=bool))

        # Solid grounds in V60 cone
        grounds = (im == 0) & cone_mask

        # Distance measurer from cone_mask boundary inwards (spim is an advanced multidimensional image processor)
        dt = spim.distance_transform_edt(cone_mask)

        # Find spheres using spim logic rather than searching by voxels and identify their centroids
        labels, n_spheres = spim.label(grounds)
        centroids = spim.center_of_mass(grounds, labels, range(1, n_spheres+1))

        new_im = im.copy()

        for i, center in enumerate(centroids):
            # z,y,x is order specified by spim
            z, y, x = int(center[0]), int(center[1]), int(center[2])
            
            # Find the distance to cone_mask boundary
            dist = dt[z,y,x]

            # Dynamic probability of removal based on exponentially decaying distance from cone_mask
            p_remove = wall_porosity_boost * np.exp(-dist / decay_width)

            # Remove if randomly generated probability falls within probability of removal
            if np.random.random() < p_remove:
                new_im[labels == (i+1)] = 1

        self.im = new_im
    
    def plot_coffee_bed(self):
        im = self.im
        shape = self.shape

        fig, axes = plt.subplots(1, 3, figsize=(18,16))
        axes[0].imshow(im[shape[0]//2,:,:], cmap='magma') # shape[i] // 2 means images are a slice from each mid coordinate
        axes[0].set_title('XY Plane (Side)')
        axes[0].set_xlabel('Y')
        axes[0].set_ylabel('Z')

        axes[1].imshow(im[:,shape[1]//2,:], cmap='magma')
        axes[1].set_title('XY Plane (Side)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')

        axes[2].imshow(im[:,:,shape[2]-1], cmap='magma')
        axes[2].set_title('XY Plane (Top-down)')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')

        plt.tight_layout()
        plt.show()
    
    def extract_network(self):
        snow_dict = ps.networks.snow2(self.im, voxel_size=1e-4, sigma=0.3, r_max=5) # resolution of 10 microns per voxel
        pn = op.io.network_from_porespy(snow_dict.network)
        pn['pore.diameter'] = pn['pore.inscribed_diameter']
        pn['throat.diameter'] = pn['throat.inscribed_diameter']

        # Ensures all pores and throats can be reached; otherwise matrix becomes unsolvable
        self.pn = self._ensure_connectivity(pn) 
        
        return pn
    
    def _ensure_connectivity(self, pn):
        adj_matrix = pn.create_adjacency_matrix(weights=None, fmt='csr')
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

        if n_components > 1:
            largest_cluster = np.argmax(np.bincount(labels))
            pores_to_trim = np.where(labels != largest_cluster)[0]
            op.topotools.trim(network=pn, pores=pores_to_trim)
            
        return pn
            
    def add_geometry_models(self):
        pn = self.pn

        # Sets a floor for pore and throat diameters; otherwise matrix can become too stiff
        pn['pore.diameter'][pn['pore.diameter'] < 1e-6] = 1e-6
        pn['throat.diameter'][pn['throat.diameter'] < 1e-6] = 1e-6
        
        # Establishes geometrical models for pores and throats
        pn.add_model(propname='pore.volume',
                    model=op.models.geometry.pore_volume.sphere,
                    pore_diameter='pore.diameter')
        
        pn.add_model(propname='pore.cross_sectional_area',
                    model=op.models.geometry.pore_cross_sectional_area.sphere,
                    pore_diameter='pore.diameter')
        
        pn['pore.shape_factor'] = 0.07 # Assume spherical shape factor
        
        pn.add_model(propname='throat.length',
                     model=op.models.geometry.throat_length.pyramids_and_cuboids,
                     pore_diameter='pore.diameter',
                     throat_diameter='throat.diameter')
        
        pn.add_model(propname='throat.cross_sectional_area',
                    model=op.models.geometry.throat_cross_sectional_area.cuboid,
                    throat_diameter='throat.diameter')
        
        pn.add_model(propname='throat.perimeter',
                     model=op.models.geometry.throat_perimeter.cuboid,
                     throat_diameter='throat.diameter')
        
        pn['throat.shape_factor'] = pn['throat.cross_sectional_area'] / (pn['throat.perimeter'])**2
        pn['throat.shape_factor'] = np.clip(pn['throat.shape_factor'], 0, 0.0795) # Safety cap
        
        pn.add_model(propname='throat.volume',
                     model=op.models.geometry.throat_volume.cuboid,
                     throat_diameter='throat.diameter',
                     throat_length='throat.length')
        
        # Precalculated effects for bulk motion and diffusive flow based on geometry 
        pn.add_model(propname='throat.hydraulic_size_factors',
                    model=op.models.geometry.hydraulic_size_factors.pyramids_and_cuboids,
                    pore_diameter='pore.diameter',
                    throat_diameter='throat.diameter')
        
        pn.add_model(propname='throat.diffusive_size_factors',
                    model=op.models.geometry.diffusive_size_factors.pyramids_and_cuboids,
                    pore_diameter='pore.diameter',
                    throat_diameter='throat.diameter')
        
        pn.add_model(propname='throat.conduit_lengths',
                     model=op.models.geometry.conduit_lengths.pyramids_and_cuboids,
                     pore_diameter='pore.diameter',
                     throat_diameter='throat.diameter')
    def phase(self):
        pn = self.pn
        phase = op.phase.Water(network=pn)

        # Keep phase temperatures in degC consistently across the simulation.
        phase['pore.temperature'] = self.temperature
        phase['throat.temperature'] = self.temperature

        mu = self._water_viscosity_from_temp_c(self.temperature)
        phase['pore.viscosity'] = mu
        phase['throat.viscosity'] = mu
        
        # According to Stokes-Einstein: D scales as absolute temperature / viscosity.
        # Temperatures are stored in degC, so convert to Kelvin only for this scaling.
        D_ref = 5e-10  # m²/s at 20°C for typical coffee solutes
        T_ref_k = 293.15
        mu_ref = 1.002e-3
        T_k = self.temperature + 273.15
        D = D_ref * (T_k / T_ref_k) * (mu_ref / mu)
        
        phase['pore.diffusivity'] = D
        phase['throat.diffusivity'] = D

        # Thermal conductivity correlation uses absolute temperature (Kelvin).
        # Using degC can make the correlation negative in the 20-95C range,
        # which destabilizes the thermal T solver.
        T_k = self.temperature + 273.15
        thermal_conductivity = -9.30e-6 * (T_k**2) + 7.19e-3*T_k - 0.711
        phase['pore.thermal_conductivity'] = thermal_conductivity
        phase['throat.thermal_conductivity'] = thermal_conductivity
        
        self.phase = phase
        return phase
    
    def add_physics_models(self):
        phase = self.phase
        # Compiled properties of viscosity and ability to flow due to bulk fluid motion and diffusion as per the schemes specified
        phase.add_model(propname='throat.hydraulic_conductance',
                        model=op.models.physics.hydraulic_conductance.valvatne_blunt,
                        pore_viscosity='pore.viscosity',
                        throat_viscosity='throat.viscosity',
                        pore_shape_factor='pore.shape_factor',
                        throat_shape_factor='throat.shape_factor',
                        pore_area='pore.cross_sectional_area',
                        throat_area='throat.cross_sectional_area',
                        conduit_lengths='throat.conduit_lengths')

        phase.add_model(propname='throat.diffusive_conductance',
                       model=op.models.physics.diffusive_conductance.taylor_aris_diffusion,
                       pore_area='pore.cross_sectional_area',
                       throat_area='throat.cross_sectional_area',
                       pore_diffusivity='pore.diffusivity',
                       throat_diffusivity='throat.diffusivity',
                       pore_pressure='pore.pressure',
                       throat_hydraulic_conductance='throat.hydraulic_conductance',
                       size_factors='throat.diffusive_size_factors')
        
        phase.add_model(propname='throat.ad_dif_conductance',
                       model=op.models.physics.ad_dif_conductance.ad_dif,
                       throat_hydraulic_conductance='throat.hydraulic_conductance',
                       throat_diffusive_conductance='throat.diffusive_conductance',
                       pore_pressure='pore.pressure',
                       s_scheme='powerlaw')

        # Used only if thermal time dependence is considered
        phase.add_model(propname='throat.thermal_conductance',
                        model=op.models.physics.thermal_conductance.generic_thermal,
                        pore_conductivity='pore.thermal_conductivity',
                        throat_conductivity='throat.thermal_conductivity',
                        size_factors='throat.diffusive_size_factors')
    def brew(self, brew_time, pour_rate, time_steps=1, shrink_factor=0.99, fines_rng_seed=None, store_snapshots=True):
        self.brew_time = brew_time
        self.pour_rate = pour_rate
        self.pressures = []  # One snapshot per time step (after fines + Stokes solve), for pressure animation
        pn = self.pn
        phase = self.phase

        # Coffee bean parameters
        rho_s = 765 # Density [kg/m3]
        cp_s = 2000 # Heat capacity [J/kgK]

        # Water parameters approximation (varies with temperature)
        rho_w = 965
        cp_w = 4190

        max_fines_per_pore = 100
        rng_fines = np.random.default_rng(fines_rng_seed)
        n_elite = max(1, int(round(0.1 * pn.Np)))
        elite_idx = rng_fines.choice(pn.Np, size=n_elite, replace=False)
        phase['pore.fines_elite'] = np.zeros(pn.Np, dtype=bool)
        phase['pore.fines_elite'][elite_idx] = True
        phase['pore.fines_elite_swollen'] = np.zeros(pn.Np, dtype=bool)
        phase['pore.fines_count'] = np.where(phase['pore.fines_elite'], 100, 10).astype(int)
        phase['throat.clogged_count'] = np.zeros(pn.Nt, dtype=int)
        max_throat_capacity = 50
        original_conductance = phase['throat.hydraulic_conductance'].copy()
        # Fines migration: per-throat baseline min(fines) over the two endpoint pores (see brew loop).
        _fc0 = phase['pore.fines_count']
        _conns0 = pn['throat.conns']
        initial_throat_pair_min = np.minimum(
            _fc0[_conns0[:, 0]], _fc0[_conns0[:, 1]]
        ).astype(np.float64)
        initial_throat_pair_min = np.maximum(initial_throat_pair_min, 1.0)

        # Manual time step
        dt = brew_time / time_steps
        self.dt = dt

        # Pre-compute a "bottom" pore subset for thermal debugging.
        # Use a quantile so it's robust to changes in network size.
        """z_coords_for_debug = pn['pore.coords'][:, 2]
        bottom_cutoff = float(np.quantile(z_coords_for_debug, 0.2))
        bottom_pores_debug_mask = z_coords_for_debug <= bottom_cutoff"""

        # Recallable function to solve flow characteristics (pressure, flow rate)
        def solve_flow(self, pour_rate):
            coords = pn.coords
            
            # Find boundary pores
            tol = 1e-6
            inlet_pores = pn.pores()[coords[:, 2] >= coords[:, 2].max() - tol]
            outlet_pores = pn.pores()[coords[:, 2] <= coords[:, 2].min() + tol]

            inlet_pressure = 50000  # Units of Pa
            
            # Initialise Stokes flow
            flow = op.algorithms.StokesFlow(network=pn, phase=phase)

            flow.settings['solver'] = 'spsolve'
            flow.settings['spsolve'] = pypardiso_spsolve

            # Set BC 
            flow.set_value_BC(pores=inlet_pores, values=inlet_pressure)
            flow.set_value_BC(pores=outlet_pores, values=200) # Backpressure due to filter paper resistance
            flow.run()
            if store_snapshots:
                self.pressures.append(flow['pore.pressure'].copy())

            # Copies results from flow 'local' to phase 'global'
            phase['pore.pressure'] = flow['pore.pressure']
            phase['throat.hydraulic_flow'] = flow.rate(throats=pn.Ts, mode='throat')
            self.flow = flow

            # Recursively finds conductance based on actual flow
            phase.regenerate_models(propnames=['throat.ad_dif_conductance'])

            return inlet_pores, outlet_pores

        # Implement transient advection solvers for diffusion and thermal effects
        ad = op.algorithms.AdvectionDiffusion(network=pn, phase=phase)
        ad_thermo = op.algorithms.AdvectionDiffusion(network=pn, phase=phase)
        # For temperature we want advection-diffusion in *energy* units.
        # We'll build `throat.ad_dif_heat_conductance` inside `brew()` each time
        # step using hydraulic flow scaled by (rho_w * cp_w).
        ad_thermo.settings['conductance'] = 'throat.ad_dif_heat_conductance'

        for solute_name, params in self.solute_classes.items():
            # Initial setup for how much solute is available for extraction and how much has been extracted
            ad['pore.concentration'] = 0.0 # Note ad 'local'
            ad_thermo['pore.temperature'] = 20.0 # Assume no preheating
            initial_mass = float(params['concentration']) * pn['pore.volume']
            phase[f'pore.{solute_name}_available'] = initial_mass.copy()
            initial_mass_fast = float(params['f_fast']) * initial_mass
            initial_mass_slow = (1.0 - float(params['f_fast'])) * initial_mass
            phase[f'pore.{solute_name}_available_fast'] = initial_mass_fast.copy()
            phase[f'pore.{solute_name}_available_slow'] = initial_mass_slow.copy()
            self.initial_mass = initial_mass
            initial_extractable_mass = float(np.sum(initial_mass))
            self.initial_extractable_mass_by_solute[solute_name] = initial_extractable_mass
            self.total_extracted = 0.0
            self.total_extracted_by_solute[solute_name] = 0.0
            self.extracted_mass_history_by_solute[solute_name] = []
            self.yield_by_solute[solute_name] = 0.0

            # Use as switches for swelling and temperature variation
            swelling_flag, temperature_flag = False, True

            # Initial setup for swelling and temperature variation 
            initial_pore_diameter = pn['pore.diameter'].copy()
            initial_throat_diameter = pn['throat.diameter'].copy()
            T_prev_step = 1e99

            # Initial call to solve flow (this will be the only call if no temperature or swelling variation)
            inlet_pores, outlet_pores = solve_flow(self, pour_rate)

            # Manual time stepping
            for step in range(time_steps):
                t = (step + 1) * dt

                # Swelling feature (disabled)
                if swelling_flag:
                    # Calculate swelling -> shrinkage of pores and throats
                    pn['throat.diameter'] = np.maximum(pn['throat.diameter']*shrink_factor, initial_throat_diameter * 0.985)
                    pn['pore.diameter'] = np.maximum(pn['pore.diameter']*shrink_factor, initial_pore_diameter * 0.985)

                    # Update geometry models -> phase models
                    pn.regenerate_models()
                    phase.regenerate_models()

                    # Resolve flow based on updated geometry and phase models
                    inlet_pores, outlet_pores = solve_flow(self, pour_rate)

                    # If all pores and throats are at minimum allowed diameter, prevent further swelling
                    if pn['throat.diameter'].all() <= 1e-6 and pn['pore.diameter'].all() <= 1e-5:
                        swelling_flag = False

                # Recalculation of viscosity and diffusivity
                if temperature_flag:
                    T_pore = phase['pore.temperature']
                    # Keep pore/throat temperatures in degC; evaluate throat temperature from adjacent pores.
                    T_throat = np.mean(T_pore[pn['throat.conns']], axis=1)
                    phase['throat.temperature'] = T_throat

                    phase['pore.viscosity'] = self._water_viscosity_from_temp_c(T_pore)
                    phase['throat.viscosity'] = self._water_viscosity_from_temp_c(T_throat)

                    # Stokes-Einstein scaling with absolute temperature in Kelvin
                    D_ref = 5e-10  # m²/s at 20°C for typical coffee solutes
                    T_ref_k = 293.15
                    mu_ref = 1.002e-3
                    phase['pore.diffusivity'] = D_ref * ((T_pore + 273.15) / T_ref_k) * (mu_ref / phase['pore.viscosity'])
                    phase['throat.diffusivity'] = D_ref * ((T_throat + 273.15) / T_ref_k) * (mu_ref / phase['throat.viscosity'])

                    # Thermal conductivity correlation uses absolute temperature (Kelvin).
                    T_pore_k = T_pore + 273.15
                    T_throat_k = T_throat + 273.15
                    phase['pore.thermal_conductivity'] = -9.30e-6 * (T_pore_k**2) + 7.19e-3*T_pore_k - 0.711
                    phase['throat.thermal_conductivity'] = -9.30e-6 * (T_throat_k**2) + 7.19e-3*T_throat_k - 0.711

                    # Update phase models (each one is specifically called to prevent updating hidden models)
                    phase.regenerate_models(propnames="throat.hydraulic_conductance")
                    phase.regenerate_models(propnames="throat.diffusive_conductance")
                    phase.regenerate_models(propnames="throat.ad_dif_conductance")
                    phase.regenerate_models(propnames="throat.thermal_conductance")

                    # Resolve flow based on updated phase models 
                    inlet_pores, outlet_pores = solve_flow(self, pour_rate)

                    # If all pore temperature is same as previous step, prevent further updating
                    if np.array_equal(T_pore,T_prev_step):
                        temperature_flag = False
                    else:
                        T_prev_step = phase['pore.temperature'].copy()

                # Calculate actual porosity at this stage since heat capacity also needs it
                self.actual_porosity = self.im[self.cone_mask].sum() / self.cone_mask.sum()

                # Thermal accumulation: (fluid + solid) heat capacity over dt — same unknown as T (°C).
                vol_term_thermal = (rho_w * cp_w * pn['pore.volume'] + rho_s * cp_s * (pn['pore.volume'] * ((1-self.actual_porosity)/self.actual_porosity))) / dt

                # Heat advection–diffusion: throat "hydraulic" weight for the thermal AD problem is the
                # volumetric conductance (m³/s) times rho*cp so inter-pore fluxes match enthalpy advection
                # rho*cp*Q*DeltaT alongside the transient term rho*cp*V*dT/dt (plus solid storage).

                ad_thermo.settings['conductance'] = 'throat.thermal_conductance'

                # 1. Start by clearing any old BCs and building the base spatial matrices
                ad_thermo.clear_BCs()
                ad_thermo.set_value_BC(pores=inlet_pores, values=95.0) # Inlet is fixed at 95C
                ad_thermo.set_outflow_BC(pores=outlet_pores)           # Let heat flow out naturally
                
                ad_thermo._build_A()
                ad_thermo._build_b()
                ad_thermo._apply_BCs()

                # 2. Add your custom Transient Accumulation (Dual-Mass)
                vol_term_thermal = (rho_w * cp_w * pn['pore.volume'] + rho_s * cp_s * (pn['pore.volume'] * ((1-self.actual_porosity)/self.actual_porosity))) / dt

                A_mat_T = ad_thermo.A + diags([vol_term_thermal], [0], format='csr')
                b_vec_T = ad_thermo.b + (vol_term_thermal * phase['pore.temperature'])

                # 3. Solve! (Notice we don't need the manual A_lil_T boundary hacks anymore!)
                T_new = pypardiso_spsolve(A_mat_T, b_vec_T)
                if np.isnan(T_new).any():
                    T_new = scipy_spsolve(A_mat_T, b_vec_T)

                # Debug: count how many pore temperatures are outside the clamp band.
                T_min_c, T_max_c = 20.0, 95.0
                """if step in {0, time_steps // 2, time_steps - 1}:
                    n_below = int(np.count_nonzero(T_new < T_min_c))
                    n_above = int(np.count_nonzero(T_new > T_max_c))
                    n_clipped = n_below + n_above
                    if n_clipped > 0:
                        print(
                            f"[T solver clip] step={step+1}/{time_steps} "
                            f"mean_unclipped={np.mean(T_new):.2f}C "
                            f"min_unclipped={np.min(T_new):.2f}C max_unclipped={np.max(T_new):.2f}C "
                            f"clip_low={n_below} clip_high={n_above} of {pn.Np}"
                        )"""

                # Save mean temperature at each time step
                self.temperature_variation['unclipped'].append(np.mean(T_new))
                if store_snapshots:
                    self.temperature_fields['unclipped'].append(T_new.copy())

                T_clipped = np.clip(T_new, T_min_c, T_max_c) # Keep between 20C and 95C
                self.temperature_variation['final'].append(np.mean(T_clipped))
                if store_snapshots:
                    self.temperature_fields['clipped'].append(T_clipped.copy())

                # Debug: report temperatures in bottom region (unclipped vs clipped)
                """if step in {0, time_steps // 2, time_steps - 1}:
                    bottom_uncl = T_new[bottom_pores_debug_mask]
                    bottom_cl = T_clipped[bottom_pores_debug_mask]
                    print(
                        f"[thermo bottom debug] step={step+1}/{time_steps} "
                        f"mean_all_uncl={np.mean(T_new):.2f}C mean_all_cl={np.mean(T_clipped):.2f}C "
                        f"mean_bottom_uncl={np.mean(bottom_uncl):.2f}C min_bottom_uncl={np.min(bottom_uncl):.2f}C "
                        f"mean_bottom_cl={np.mean(bottom_cl):.2f}C min_bottom_cl={np.min(bottom_cl):.2f}C"
                    )"""

                # Update phase
                ad_thermo['pore.temperature'] = T_clipped
                phase['pore.temperature'] = T_clipped

                # Diagnostic for residence time during brewing phase (useful for swelling)
                """P = phase['pore.pressure']
                g = phase['throat.hydraulic_conductance']
                conns = pn['throat.conns']
                manual_flow = g * np.abs(P[conns[:, 0]] - P[conns[:, 1]])

                # A more robust way to find total flow Q:
                # Find all throats connecting the top half of the bed to the bottom half
                z_coords = pn['pore.coords'][:, 2]
                mid_z = np.median(z_coords)

                top_pores = np.where(z_coords > mid_z)[0]
                bot_pores = np.where(z_coords <= mid_z)[0]

                # Throats that cross the middle plane
                cross_throats = pn.find_connected_pores(top_pores, bot_pores)
                Q_total = np.sum(manual_flow[cross_throats])

                V_total = np.sum(pn['pore.volume']) + np.sum(pn['throat.volume'])

                print(f"Corrected Residence Time: {V_total / Q_total} seconds at time step {step}")"""

                # Solute AD: same pattern as thermal — BCs via the algorithm, then custom RHS/matrix terms.
                ad.clear_BCs()
                ad.set_value_BC(pores=inlet_pores, values=0.0)  # fresh feed, C = 0
                ad.set_outflow_BC(pores=outlet_pores)
                
                ad._build_A()
                ad._build_b()
                ad._apply_BCs()

                # Extraction linearization Y = A1*C + A2 (same kinetics as post-step mass update).
                remaining_fast = np.divide(
                    phase[f'pore.{solute_name}_available_fast'],
                    initial_mass_fast,
                    out=np.zeros_like(initial_mass_fast),
                    where=initial_mass_fast > 0,
                )
                remaining_slow = np.divide(
                    phase[f'pore.{solute_name}_available_slow'],
                    initial_mass_slow,
                    out=np.zeros_like(initial_mass_slow),
                    where=initial_mass_slow > 0,
                )
                k_eff = (params['k_fast'] * remaining_fast) + (params['k_slow'] * remaining_slow)
                A1 = -k_eff * pn['pore.volume']
                A2 = k_eff * params['c_sat'] * pn['pore.volume']
                # Do not add extraction on Dirichlet inlet pores (equivalent to former row-zeroing).
                A1_eff = A1.copy()
                A2_eff = A2.copy()
                A1_eff[inlet_pores] = 0.0
                A2_eff[inlet_pores] = 0.0
                M_source = spdiags(data=-A1_eff, diags=0, m=pn.Np, n=pn.Np)
                A_mat = ad.A + M_source
                b_vec = ad.b + A2_eff
                
                """# Error handling for 0 on the diagonals 
                diagonals = A_mat.diagonal() 
                zero_diags = np.where(np.abs(diagonals) < 1e-20)[0]
                if len(zero_diags) > 0:
                    print("Remaining error of near 0s on the diagonal: ", len(zero_diags))
                    fix_array = np.zeros(pn.Np)
                    fix_array[zero_diags] = 1.0
                    A_mat = A_mat + diags([fix_array] [0], format='csr')"""

                # Solve for C_new, preferably with pypardiso solver but with scipy solver if any errors
                C_new = pypardiso_spsolve(A_mat, b_vec)
                if np.isnan(C_new).any():
                    C_new = scipy_spsolve(A_mat, b_vec)

                # Update for next time step
                ad['pore.concentration'] = C_new.copy() # Concentration in each pore

                # Update remaining solute and ensure extraction does not exceed available
                driving_force = np.maximum(0, params['c_sat'] - C_new)
                mass_from_fast = params['k_fast'] * driving_force * pn['pore.volume'] * dt
                mass_from_slow = params['k_slow'] * driving_force * pn['pore.volume'] * dt
                extracted_fast = np.minimum(mass_from_fast, phase[f'pore.{solute_name}_available_fast'])
                extracted_slow = np.minimum(mass_from_slow, phase[f'pore.{solute_name}_available_slow'])
                phase[f'pore.{solute_name}_available_fast'] -= extracted_fast
                phase[f'pore.{solute_name}_available_slow'] -= extracted_slow
                phase[f'pore.{solute_name}_available'] = (
                    phase[f'pore.{solute_name}_available_fast']
                    + phase[f'pore.{solute_name}_available_slow']
                )
                extracted_step = extracted_fast + extracted_slow

                # Store for data visualisation
                if solute_name == 'acids':
                    self.time_steps.append(t)
                if store_snapshots:
                    self.concentrations[solute_name].append(C_new.copy())
                self.total_extracted += float(np.sum(np.maximum(extracted_step, 0.0)))
                self.total_extracted_by_solute[solute_name] = self.total_extracted
                self.extracted_mass_history_by_solute[solute_name].append(self.total_extracted)
                bean_mass = initial_extractable_mass / 0.3
                self.yield_by_solute[solute_name] = (self.total_extracted / bean_mass) if bean_mass > 0 else np.nan

                # Velocity and flow direction (use throat hydraulic flow, not conductance)
                h_flow = phase['throat.hydraulic_flow']
                u_throats = np.abs(h_flow) / pn['throat.cross_sectional_area']

                upstream_pores = np.where(h_flow > 0, pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])
                downstream_pores = np.where(h_flow > 0, pn['throat.conns'][:, 1], pn['throat.conns'][:, 0])

                # 2. Probabilistic Entrainment
                # Per-throat: P = min(1, u * p_scale). p_scale = 0.2 * (pair_min / initial_pair_min)
                # where pair_min is min(fines) over the two pores of that throat; drops as those pores deplete.
                _c = phase['pore.fines_count']
                _tc = pn['throat.conns']
                pair_min = np.minimum(_c[_tc[:, 0]], _c[_tc[:, 1]]).astype(np.float64)
                p_probability_scale = 0.4 * (pair_min / initial_throat_pair_min)
                p_move = np.minimum(1.0, u_throats * p_probability_scale)

                # 3. Binomial Draw: How many fines WANT to move?
                available_fines = phase['pore.fines_count'][upstream_pores]
                fines_to_pull = np.random.binomial(n=available_fines, p=p_move)

                # --- TRAFFIC CONTROL 1: Prevent "Overdrawing" ---
                # Multiple throats might pull from the same pore, requesting more fines than exist.
                total_requested = np.bincount(upstream_pores, weights=fines_to_pull, minlength=pn.Np)
                overdrawn_pores = total_requested > phase['pore.fines_count']

                # Scale down the requests proportionally, flooring to keep integers
                scale_factor = np.ones(pn.Np)
                scale_factor[overdrawn_pores] = phase['pore.fines_count'][overdrawn_pores] / total_requested[overdrawn_pores]
                fines_to_pull = np.floor(fines_to_pull * scale_factor[upstream_pores]).astype(int)

                # --- TRAFFIC CONTROL 2: Enforce the 100-Particle Limit ---
                is_small = pn['throat.diameter'] < 100e-6

                # Fines that survive the throat diameter check
                arriving_at_pores = fines_to_pull * (~is_small)
                total_arriving = np.bincount(downstream_pores, weights=arriving_at_pores, minlength=pn.Np)

                available_space = np.maximum(max_fines_per_pore - phase['pore.fines_count'], 0)
                overflow_pores = total_arriving > available_space

                # If more arrive than space permits, reject the excess
                accept_factor = np.ones(pn.Np)
                accept_factor[overflow_pores] = available_space[overflow_pores] / total_arriving[overflow_pores]

                # Fines that successfully enter the downstream pore
                successful_transfer = np.floor(arriving_at_pores * accept_factor[downstream_pores]).astype(int)

                # --- TRAFFIC CONTROL 3: Handle the Rejected Fines ---
                # Fines that didn't fit into the downstream pore get jammed in the throat
                rejected_by_pore = arriving_at_pores - successful_transfer

                # Total fines stuck in the throat = (caught by small diameter) + (rejected by full pore)
                fines_clogging = (fines_to_pull * is_small) + rejected_by_pore

                fines_count_before = phase['pore.fines_count'].copy()

                # 4. Update the Discrete Counts
                np.add.at(phase['pore.fines_count'], upstream_pores, -fines_to_pull)
                np.add.at(phase['pore.fines_count'], downstream_pores, successful_transfer)
                phase['throat.clogged_count'] += fines_clogging

                # Elite pores: one-time swell when inventory first hits zero (no fines refill).
                swell_mask = (
                    phase['pore.fines_elite']
                    & (~phase['pore.fines_elite_swollen'])
                    & (fines_count_before > 0)
                    & (phase['pore.fines_count'] == 0)
                )
                swell_ps = np.where(swell_mask)[0]
                if swell_ps.size > 0:
                    pn['pore.diameter'][swell_ps] *= 1.5
                    pn['pore.diameter'][pn['pore.diameter'] < 1e-6] = 1e-6
                    conns_td = pn['throat.conns']
                    pn['throat.diameter'] = np.minimum(
                        pn['pore.diameter'][conns_td[:, 0]],
                        pn['pore.diameter'][conns_td[:, 1]],
                    )
                    pn['throat.diameter'][pn['throat.diameter'] < 1e-6] = 1e-6
                    phase['pore.fines_elite_swollen'][swell_ps] = True
                    pn.regenerate_models()
                    phase.regenerate_models()
                    original_conductance = phase['throat.hydraulic_conductance'].copy()
                # 5. Update Conductance
                # Use original conductance so the power law scales correctly every step.
                # Fully-clogged throats keep a tiny residual (1e-8 of original) so the Stokes
                # matrix never has an all-zero row, which would make it singular.
                clogging_ratio = np.clip(phase['throat.clogged_count'] / max_throat_capacity, 0, 1)
                phase['throat.hydraulic_conductance'] = original_conductance * np.maximum((1 - clogging_ratio)**3, 1e-8)

                # Stokes solve after fines/clogging so pressure field matches current conductances (one snapshot per step).
                inlet_pores, outlet_pores = solve_flow(self, pour_rate)

                # Residence-time diagnostic: use solved outlet flow directly.
                Q_total = float(
                    np.abs(np.atleast_1d(self.flow.rate(pores=outlet_pores, mode='group')).sum())
                )
                V_total = np.sum(pn['pore.volume']) + np.sum(pn['throat.volume'])
                rt = np.inf if Q_total <= 0 else (V_total / Q_total)
                #print(f"Residence Time: {rt} seconds at time step {step+1}/{time_steps}, ")

    def generate_brewing_animation(self, solute_name='acids'):
        # Get coordinates
        coords = self.pn['pore.coords']
        y = coords[:, 1]
        z = coords[:, 2]
        
        # Define the grid where we want to "paint" the heatmap
        xi = np.linspace(y.min(), y.max(), 100)
        zi = np.linspace(z.min(), z.max(), 100)
        xi, zi = np.meshgrid(xi, zi)

        fig, ax = plt.subplots(figsize=(6, 8))
        
        # Initialize the plot with the first time step
        c_data = self.concentrations[solute_name][0]
        grid_c = griddata((y, z), c_data, (xi, zi), method='linear')
        
        im = ax.imshow(grid_c, extent=(y.min(), y.max(), z.min(), z.max()), 
                        origin='lower', aspect='auto', cmap='magma')
        plt.colorbar(im, label='Concentration [kg/m³]')
        ax.set_title(f'Extraction Front: {solute_name}')
        ax.set_xlabel('Width [m]')
        ax.set_ylabel('Bed Depth [m]')

        # Update function for the animation
        def update(frame):
            c_data = self.concentrations[solute_name][frame]
            # Interpolate scattered pore data to the regular grid
            grid_c = griddata((y, z), c_data, (xi, zi), method='linear')
            im.set_array(grid_c)
            ax.set_title(f'Time Step: {frame} - {solute_name}')
            return [im]

        ani = FuncAnimation(fig, update, frames=len(self.concentrations[solute_name]), 
                            interval=50, blit=True)
        
        # To save as MP4 (requires ffmpeg)
        ani.save('coffee_extraction.gif', writer='pillow')

    def generate_pressure_animation(self, save_path='coffee_pressure.gif', interval=50):
        """
        Animate pore pressure on the same (y, z) slice grid as the solute GIF.
        Snapshots are stored after each brew step (post–fines migration Stokes solve).
        """
        n_frames = len(self.pressures)
        if n_frames == 0:
            raise RuntimeError("No pressure snapshots; run brew() first.")

        coords = self.pn['pore.coords']
        y = coords[:, 1]
        z = coords[:, 2]

        xi = np.linspace(y.min(), y.max(), 100)
        zi = np.linspace(z.min(), z.max(), 100)
        xi, zi = np.meshgrid(xi, zi)

        p_min = min(float(np.min(p)) for p in self.pressures)
        p_max = max(float(np.max(p)) for p in self.pressures)
        if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
            p_max = p_min + 1.0

        P0 = self.pressures[0]
        grid0 = griddata((y, z), P0, (xi, zi), method='linear')

        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(
            grid0,
            extent=(y.min(), y.max(), z.min(), z.max()),
            origin='lower',
            aspect='auto',
            cmap='viridis',
            vmin=p_min,
            vmax=p_max,
        )
        plt.colorbar(im, label='Pressure [Pa]')
        ax.set_title('Pore pressure (after fines / Stokes)')
        ax.set_xlabel('Width [m]')
        ax.set_ylabel('Bed depth [m]')

        use_time = len(self.time_steps) == n_frames

        def update(frame):
            P = self.pressures[frame]
            grid_p = griddata((y, z), P, (xi, zi), method='linear')
            im.set_array(grid_p)
            if use_time:
                ax.set_title(f't = {self.time_steps[frame]:.1f} s  |  step {frame}')
            else:
                ax.set_title(f'Pressure  |  step {frame}')
            return [im]

        ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
        ani.save(save_path, writer='pillow')

    def generate_temperature_animation(self, save_path='temperature_debug.gif', interval=50):
        """
        Side-by-side animation of pore temperature heatmaps:
        - unclipped: raw T_new from the thermal linear solve
        - clipped: after applying the [20C, 95C] clamp
        Color scale is fixed (80–95 °C) so GIFs are comparable across runs.
        """

        # Fixed color limits for cross-run comparison (deg C).
        t_vmin_gif, t_vmax_gif = 90.0, 95.0

        coords = self.pn['pore.coords']
        y = coords[:, 1]
        z = coords[:, 2]

        # Define the grid where we want to interpolate (same approach as concentration animation).
        xi = np.linspace(y.min(), y.max(), 100)
        zi = np.linspace(z.min(), z.max(), 100)
        xi, zi = np.meshgrid(xi, zi)

        n_frames = len(self.temperature_fields['clipped'])
        if n_frames == 0:
            raise RuntimeError("Temperature animation has no frames to render.")

        # Initial frames
        T_cl0 = self.temperature_fields['clipped'][0]
        T_ucl0 = self.temperature_fields['unclipped'][0]

        grid_cl0 = griddata((y, z), T_cl0, (xi, zi), method='linear')
        grid_ucl0 = griddata((y, z), T_ucl0, (xi, zi), method='linear')

        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        ax_cl = axes[0]
        ax_ucl = axes[1]

        im_cl = ax_cl.imshow(
            grid_cl0, extent=(y.min(), y.max(), z.min(), z.max()),
            origin='lower', aspect='auto', cmap='magma', vmin=t_vmin_gif, vmax=t_vmax_gif
        )
        im_ucl = ax_ucl.imshow(
            grid_ucl0, extent=(y.min(), y.max(), z.min(), z.max()),
            origin='lower', aspect='auto', cmap='magma', vmin=t_vmin_gif, vmax=t_vmax_gif
        )

        plt.colorbar(im_cl, ax=ax_cl, label='Temperature [C] (clipped)')
        plt.colorbar(im_ucl, ax=ax_ucl, label='Temperature [C] (unclipped)')

        ax_cl.set_title(f'Clipped mean: {np.mean(T_cl0):.2f}C')
        ax_cl.set_xlabel('Width [m]')
        ax_cl.set_ylabel('Bed Depth [m]')
        ax_ucl.set_title(f'Unclipped mean: {np.mean(T_ucl0):.2f}C')
        ax_ucl.set_xlabel('Width [m]')
        ax_ucl.set_ylabel('Bed Depth [m]')

        def update(frame):
            T_cl = self.temperature_fields['clipped'][frame]
            T_ucl = self.temperature_fields['unclipped'][frame]

            grid_cl = griddata((y, z), T_cl, (xi, zi), method='linear')
            grid_ucl = griddata((y, z), T_ucl, (xi, zi), method='linear')

            im_cl.set_array(grid_cl)
            im_ucl.set_array(grid_ucl)
            ax_cl.set_title(f'Time Step: {frame}, Clipped mean: {np.mean(T_cl):.2f}C')
            ax_ucl.set_title(f'Unclipped mean: {np.mean(T_ucl):.2f}C')
            return [im_cl, im_ucl]

        ani = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=True
        )

        ani.save(save_path, writer='pillow')

    def plot_results(self):
        pn = self.pn
        coords = pn.coords
        
        #n_steps = len(self.time_steps)
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # Plot 1: Concentration distribution at final step
        for solute in self.solute_classes.keys():
            axes[0, 0].hist(self.concentrations[solute][-1], bins=40, edgecolor='black', alpha=0.7, label=solute)
        axes[0, 0].set_xlabel('Pore concentration (normalized)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Final Concentration Distribution (t={self.time_steps[-1]:.1f}s)')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Pressure profile (final)
        P_final = self.pressures[-1]
        axes[1, 0].scatter(coords[:, 2], P_final, alpha=0.5, s=10, color='red')
        axes[1, 0].set_xlabel('Z-coordinate (voxels)')
        axes[1, 0].set_ylabel('Pressure (Pa)')
        axes[1, 0].set_title('Pressure Profile Along Flow Direction (Final)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 3: Concentration profile (final)
        for solute in self.solute_classes.keys():
            axes[1, 1].scatter(coords[:, 2], self.concentrations[solute][-1], alpha=0.5, s=10, label=solute)
        axes[1, 1].set_xlabel('Z-coordinate (voxels)')
        axes[1, 1].set_ylabel('Concentration (normalized)')
        axes[1, 1].set_title('Extraction Profile Along Flow (Final)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 4: Network statistics
        pore_diam = pn['pore.diameter']
        throat_diam = pn['throat.diameter']
        axes[2, 0].hist(pore_diam, bins=30, alpha=0.6, label='Pores', edgecolor='black')
        axes[2, 0].hist(throat_diam, bins=30, alpha=0.6, label='Throats', edgecolor='black')
        axes[2, 0].set_xlabel('Diameter (voxels)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Pore/Throat Size Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 5: Differential beverage concentration vs cumulative liquid amount.
        time_arr = np.asarray(self.time_steps, dtype=float)
        water_passed_so_far = self.pour_rate * 1e-6 * 1e6 * time_arr
        for solute in self.solute_classes.keys():
            extracted_hist = np.asarray(self.extracted_mass_history_by_solute.get(solute, []), dtype=float)
            n = min(len(time_arr), len(extracted_hist))
            if n == 0:
                continue
            coffee_mass = self.initial_extractable_mass_by_solute[solute] * 1000 / 0.3
            retained_water_mass = 2.6 * coffee_mass
            beverage_mass_cum = np.maximum(0.0, water_passed_so_far[:n] - retained_water_mass)

            # Differential concentration of liquid leaving the bed:
            # c_brew = d(extracted_mass)/d(beverage_mass_collected), converted to mg/g.
            d_extracted = np.diff(extracted_hist[:n], prepend=0.0)
            d_beverage = np.diff(beverage_mass_cum, prepend=0.0)
            beverage_conc = np.full(n, np.nan, dtype=float)
            valid = d_beverage > 0
            beverage_conc[valid] = 1e6 * (d_extracted[valid] / d_beverage[valid])
            axes[0,1].plot(beverage_mass_cum, beverage_conc, 'o-', alpha=0.7, label=solute)
        axes[0,1].set_xlabel('Cumulative liquid amount')
        axes[0,1].set_ylabel('Differential beverage concentration (mg/g)')
        axes[0,1].set_title('Differential beverage concentration vs cumulative liquid amount')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Plot 6: Concentration against time for different parts of CV
        z_coords = pn['pore.coords'][:,2]
        top_p = np.argmax(z_coords)
        mid_p = np.argmin(np.abs(z_coords - np.median(z_coords)))
        bot_p = np.argmin(z_coords)
        top_curve = [c[top_p] for c in self.concentrations['acids']]
        mid_curve = [c[mid_p] for c in self.concentrations['acids']]
        bot_curve = [c[bot_p] for c in self.concentrations['acids']]
        axes[2,1].plot(self.time_steps, top_curve, label='Inlet (Top)')
        axes[2,1].plot(self.time_steps, mid_curve, label='Middle')
        axes[2,1].plot(self.time_steps, bot_curve, label='Outlet (Bottom)')
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Concentrations')
        axes[2, 1].set_title('Concentrations of top, middle and bottom of CV against time')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    def print_statistics(self):
        pn = self.pn
        phase = self.phase

        # Flow rate for residence-time estimate: use outlet flow from latest solved StokesFlow.
        coords = pn['pore.coords']
        tol = 1e-6
        outlet_pores = pn.pores()[coords[:, 2] <= coords[:, 2].min() + tol]
        Q_total = float(
            np.abs(np.atleast_1d(self.flow.rate(pores=outlet_pores, mode='group')).sum())
        )

        V_total = np.sum(pn['pore.volume']) + np.sum(pn['throat.volume'])

        print(f"\n{'='*60}")
        print(f"COFFEE BED SIMULATION STATISTICS")
        print(f"{'='*60}")
        print(f"Initial temperature: {self.temperature}°C")
        print(f"Target porosity: {self.porosity:.1%}")
        print(f"Actual porosity: {self.actual_porosity:.1%}")
        print(f"Domain shape: {self.shape}")
        print(f"\nNetwork Properties:")
        print(f"  Number of pores: {pn.Np}")
        print(f"  Number of throats: {pn.Nt}")
        print(f"  Avg pore diameter: {pn['pore.diameter'].mean():.7f}m")
        print(f"  Avg throat diameter: {pn['throat.diameter'].mean():.7f}m")
        print(f"  Pore diameter range: {pn['pore.diameter'].min():.7f} - {pn['pore.diameter'].max():.7f}")
        print(f"  Throat diameter range: {pn['throat.diameter'].min():.7f} - {pn['throat.diameter'].max():.7f}")
        print(f"\nConductance properties:")
        print(f" Hydraulic conductance: {phase['throat.hydraulic_conductance'].min():.3e} - {phase['throat.hydraulic_conductance'].max():.3e}")
        print(f" Hydraulic conductance ratio: {phase['throat.hydraulic_conductance'].max()/phase['throat.hydraulic_conductance'].min():.0e}")
        print(f" Zero/negative hydraulic conductances: {(phase['throat.hydraulic_conductance']<=0).sum()}")
        print(f" Diffusive conductance: {phase['throat.diffusive_conductance'].min():.3e} - {phase['throat.diffusive_conductance'].max():.3e}")
        print(f" Diffusive conductance ratio: {phase['throat.diffusive_conductance'].max()/phase['throat.diffusive_conductance'].min():.0e}")
        print(f" Zero/negative diffusive conductances: {(phase['throat.diffusive_conductance']<=0).sum()}")
        print(f"\nPhase Properties at {self.temperature}°C:")
        print(f"  Viscosity: {self.phase['pore.viscosity'][0]:.3e} Pa·s")
        print(f"  Diffusivity: {self.phase['pore.diffusivity'][0]:.3e} m²/s")
        print(f"\nBrewing Progress:")
        print(f"  Total brew time: {self.time_steps[-1]:.1f}s")
        rt = np.inf if Q_total <= 0 else (V_total / Q_total)
        print(f"Corrected Residence Time: {rt} seconds")
        if self.concentrations:
            for solute in self.solute_classes.keys():
                print(f" {solute} statistics:")
                initial_extractable_mass = self.initial_extractable_mass_by_solute.get(solute, float(np.sum(self.initial_mass)))
                extracted_mass = self.total_extracted_by_solute.get(solute, self.total_extracted)
                bean_mass = initial_extractable_mass / 0.3
                yield_val = (extracted_mass / bean_mass) if bean_mass > 0 else np.nan
                print("Total mass to be extracted: ", initial_extractable_mass)
                print(f"Extracted mass: {extracted_mass}")
                print(f"Yield: {yield_val:%}")
                print(f"TDS: {(extracted_mass / ((self.brew_time*self.pour_rate*1e-6*1000) - (2.6 * bean_mass))):%}")
                print()
        print(f"{'='*60}\n")