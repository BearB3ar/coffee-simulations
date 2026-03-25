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

    def __init__(self, domain_shape=[300,300,300], porosity = 0.46, temperature = 95, particle_size_dist = 'twin_lognormal', solute_classes=None):
        self.shape = domain_shape # Cuboidal control volume, trimming to cone shape done later (units of voxels)
        self.porosity = porosity # Target porosity, actual porosity depends on porespy image generation + wall effect 
        self.temperature = temperature # Initial temperature
        self.particle_size_dist = particle_size_dist
        self.time_steps = [] # For results graphs
        self.concentrations = { # Stores concentration of the entire bed at every time step
            'acids': [],
        }
        self.temperature_variation = { # Stores mean temperature of the bed at every time step, before and after clipping
            "final": []
        }
        # Stores full pore temperature fields for debugging the T solver.
        # Each entry is an array of shape (Np,) in the same pore ordering as `self.pn`.
        self.temperature_fields = {
            "clipped": []
        }
        self.pressures = [] # Stores pressures of the entire bed at every time step
        self.total_extracted = 0.0 # Cumulative measure for how much solute leaves outlet_pores
        if solute_classes is None:
            self.solute_classes = {
                # TODO: Tune amount of coffee present initially and other parameters
                'acids': {'k' : 10e-3, 'concentration' : 12e8, 'c_sat' : 150.0}, # Target initial mass is 6.47e-2
            }
        else:
            self.solute_classes = solute_classes

    def _water_viscosity_from_temp_c(self, temp_c):
        # Linear interpolation over tabulated viscosity-vs-temperature data
        temps = np.array(sorted(self.viscosity_ref_table.keys()), dtype=float)
        mus = np.array([self.viscosity_ref_table[t] for t in temps], dtype=float)
        temp_arr = np.asarray(temp_c, dtype=float)
        return np.interp(temp_arr, temps, mus, left=mus[0], right=mus[-1])

    def generate_coffee_bed(self):
        shape = self.shape
        # Approximates the grinds distribution as 2 separate lognormal curves
        if self.particle_size_dist == "twin_lognormal":
            x_axis = np.arange(0.5,100,0.5)
            target_peak = scipy.stats.lognorm(s=0.42, scale=(650e-6/2)/1e-4) # s sets the standard deviation, scale sets the median (units of voxels)
            weight_target = 0.92 
            fines_peak = scipy.stats.lognorm(s=0.8, scale=(100e-6/2)/1e-4)
            weight_fines = 0.08

            probs = (target_peak.pdf(x_axis) * weight_target) + (fines_peak.pdf(x_axis) * weight_fines) # Both curves are weighted and rebased
            probs = probs / probs.sum()
            custom_dist_object = scipy.stats.rv_discrete(name='coffee_dist', values=(x_axis, probs)) # rv_discrete creates a discretised random variable function

            # porespy generates an image of pores and throats with distances between them based on the distribution of coffee particles desired and minimum radius of 0.05 voxels 
            im = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.05, porosity=self.porosity, dist=custom_dist_object) 

        # Alternative distribution method but less preferred due to clipping between images causing unrealistic geometry
        elif self.particle_size_dist == "bimodal":
            target_dist = scipy.stats.norm(loc=(650e-6/2)/1e-4, scale=0.5)
            fines_dist = scipy.stats.norm(loc=(100e-6/2)/1e-4, scale=1)
            im_main = ps.generators.polydisperse_spheres(shape=self.shape, r_min=2, porosity=self.porosity*0.9, dist=target_dist)
            im_fines = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.5, porosity=self.porosity*0.1, dist=fines_dist)

            im = np.logical_or(im_main, im_fines).astype(int)

        # Trimming to cone geometry
        x,y,z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        center_y, center_x = shape[0] // 2, shape[1] // 2
        radius_at_z = -(z + 20) * np.tan(np.radians(30))
        cone_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius_at_z**2
        self.cone_mask = cone_mask # Needed for porosity calculation at the end
        im = im & cone_mask
        
        self.im = im
    
    def wall_effect(self, wall_porosity_boost=0.2, decay_width=10):
        im = self.im
        cone_mask = self.cone_mask

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
        snow_dict = ps.networks.snow2(self.im, voxel_size=1e-4, sigma=0.3, r_max=5) # resolution of 100 microns per voxel
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
        
        pn.add_model(propname='pore.area',
                    model=op.models.geometry.pore_cross_sectional_area.sphere,
                    pore_diameter='pore.diameter')
        
        pn.add_model(propname='throat.length',
                     model=op.models.geometry.throat_length.pyramids_and_cuboids,
                     pore_diameter='pore.diameter',
                     throat_diameter='throat.diameter')
        
        pn.add_model(propname='throat.cross_sectional_area',
                    model=op.models.geometry.throat_cross_sectional_area.cylinder,
                    throat_diameter='throat.diameter')
        
        pn.add_model(propname='throat.volume',
                     model=op.models.geometry.throat_volume.cylinder,
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
                        model=op.models.physics.hydraulic_conductance.hagen_poiseuille,
                        pore_viscosity='pore.viscosity',
                        throat_viscosity='throat.viscosity',
                        size_factors='throat.hydraulic_size_factors')

        phase.add_model(propname='throat.diffusive_conductance',
                       model=op.models.physics.diffusive_conductance.generic_diffusive,
                       pore_diffusivity='pore.diffusivity',
                       throat_diffusivity='throat.diffusivity',
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
        
    def brew(self, brew_time, pour_rate, time_steps=1, shrink_factor=0.99):
        self.brew_time = brew_time
        self.pour_rate = pour_rate
        pn = self.pn
        phase = self.phase

        # Coffee bean parameters
        rho_s = 765 # Density [kg/m3]
        cp_s = 2000 # Heat capacity [J/kgK]

        # Water parameters approximation (varies with temperature)
        rho_w = 965
        cp_w = 4190

        # TODO: Tune fine migration parameters
        # Maximum amount of fines in each throat
        max_clog_capacity = pn['throat.volume'] * rho_s * (1-0.8)

        V_total_cell = (pn['pore.diameter']**3) # Simplification of the voxel box
        V_solid = np.maximum(V_total_cell - pn['pore.volume'],0)
        phase['pore.fines_mass'] = (V_solid / V_solid.sum()) * 0.005

        # Logs the fines that have been displaced and their location
        phase['throat.clogged_mass'] = np.zeros(pn.Nt)

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

            inlet_pressure = 1000 * (pour_rate / 200) + 500  # Units of Pa
            
            # Initialise Stokes flow
            flow = op.algorithms.StokesFlow(network=pn, phase=phase)

            flow.settings['solver'] = 'spsolve'
            flow.settings['spsolve'] = pypardiso_spsolve

            # Set BC 
            flow.set_value_BC(pores=inlet_pores, values=inlet_pressure)
            flow.set_value_BC(pores=outlet_pores, values=0.0)
            flow.run()
            self.pressures.append(flow['pore.pressure'].copy())

            # Copies results from flow 'local' to phase 'global'
            phase['pore.pressure'] = flow['pore.pressure']
            phase['throat.hydraulic_flow'] = flow.rate(throats=pn.Ts, mode='throat')

            # Recursively finds conductance based on actual flow
            phase.regenerate_models(propnames=['throat.ad_dif_conductance'])

            return inlet_pores, outlet_pores

        # Implement transient advection solvers for diffusion and thermal effects
        ad = op.algorithms.AdvectionDiffusion(network=pn, phase=phase)
        ad_thermo = op.algorithms.AdvectionDiffusion(network=pn, phase=phase)
        ad_thermo.settings['conductance'] = 'throat.thermal_conductance'

        for solute_name, params in self.solute_classes.items():
            # Initial setup for how much solute is available for extraction and how much has been extracted
            ad['pore.concentration'] = 0.0 # Note ad 'local'
            ad_thermo['pore.temperature'] = 20.0 # Assume no preheating
            initial_mass, phase[f'pore.{solute_name}_available'] = float(params['concentration']) * pn['pore.volume'], float(params['concentration']) * pn['pore.volume']
            self.initial_mass = initial_mass

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

                # Solve temperature variation
                # Start by building A and b matrices in Ax=b
                ad_thermo._build_A()
                ad_thermo._build_b()

                # Calculate flow outflow, to be used in solute and thermal outflow
                Q_out = np.zeros(pn.Np)
                for pore in outlet_pores:
                    # Find all throats connected to this specific outlet pore
                    connected_throats = pn.find_neighbor_throats(pores=pore)
                    
                    # Since water is incompressible and the network ends here, 
                    # the absolute sum of flow in these throats is exactly the flow leaving into the cup.
                    Q_out[pore] = np.sum(np.abs(phase['throat.hydraulic_flow'][connected_throats]))

                # Calculate actual porosity at this stage since heat capacity also needs it
                self.actual_porosity = self.im[self.cone_mask].sum() / self.cone_mask.sum()

                # Calculate thermal accumulation (Density * Heat Capacity * Volume / dt)
                # Heat capacity is of the water-filled pore and the solid grain
                vol_term_thermal = (rho_w * cp_w * pn['pore.volume'] + rho_s * cp_s * (pn['pore.volume'] * ((1-self.actual_porosity)/self.actual_porosity))) / dt
                
                # Calculate thermal drain due to outflow
                thermal_drain = np.zeros(pn.Np)
                # Outflow sink for the advected "temperature" variable should be scaled
                # like the concentration outflow (i.e. proportional to volumetric flow).
                # Multiplying by rho*cp makes this term inconsistent with ad_thermo's units.
                thermal_drain[outlet_pores] = Q_out[outlet_pores]

                """# Debug: thermal_drain magnitude vs solver diagonal
                if step in {0, time_steps // 2, time_steps - 1}:
                    diag_ad = ad_thermo.A.diagonal()
                    if len(outlet_pores) > 0:
                        td_out = thermal_drain[outlet_pores]
                        diag_out = diag_ad[outlet_pores]
                        guard = np.where(diag_out != 0, diag_out, np.nan)
                        ratio = td_out / guard
                        print(
                            f"[thermo drain debug] step={step+1}/{time_steps} "
                            f"outlet_pores={len(outlet_pores)} "
                            f"thermal_drain_out_min={float(np.nanmin(td_out)):.3e} "
                            f"thermal_drain_out_max={float(np.nanmax(td_out)):.3e} "
                            f"diag_out_median={float(np.nanmedian(diag_out)):.3e} "
                            f"ratio_td_to_diag_median={float(np.nanmedian(ratio)):.3g} "
                            f"ratio_min={float(np.nanmin(ratio)):.3g} ratio_max={float(np.nanmax(ratio)):.3g}"
                        )"""

                # IMPORTANT: `AdvectionDiffusion` already includes the transient/accumulation
                # term internally for the field `ad_thermo['pore.temperature']`.
                # Adding `vol_term_thermal` again double-counts accumulation and can over-cool parts
                # of the domain. Keep only the explicit outlet drain as an extra sink term.
                A_mat_T = ad_thermo.A + diags([thermal_drain], [0], format='csr')
                b_vec_T = ad_thermo.b

                # Apply BC
                A_lil_T = A_mat_T.tolil() # Convert to LIL format for faster row operations
                A_lil_T[inlet_pores, :] = 0.0
                A_lil_T[inlet_pores, inlet_pores] = 1.0
                A_mat_T = A_lil_T.tocsr() # Convert back to csr format for sparse solver
                b_vec_T[inlet_pores] = 95 # 95°C Inlet

                # Solve for temperature (pypardiso is a faster solver but may trip if matrix is not well conditioned)
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
                #self.temperature_variation['unclipped'].append(np.mean(T_new))
                #self.temperature_fields['unclipped'].append(T_new.copy())

                T_clipped = np.clip(T_new, T_min_c, T_max_c) # Keep between 20C and 95C
                self.temperature_variation['final'].append(np.mean(T_clipped))
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

                # Build original version of A and b for solute flow
                ad._build_A()
                ad._build_b()

                # Calculate extraction equation
                # Calculate A1 and A2 (Y = A1X + A2)
                placeholder = np.ones(pn.Np)
                remaining_ratio = phase[f'pore.{solute_name}_available'] / initial_mass
                A1 = -params['k'] * remaining_ratio * placeholder * pn['pore.volume']
                A2 = params['k'] * params['c_sat'] * remaining_ratio * placeholder * pn['pore.volume']

                # IMPORTANT: `AdvectionDiffusion` already includes the transient/accumulation
                # term for `ad['pore.concentration']`, so we should not add `vol_term` again.
                # Only add the reaction/extraction contribution (from A1/A2) here.
                # `A1` is negative in your linearization, so `-A1` is the diagonal sink/source weight.
                M_source = spdiags(data=-A1, diags=0, m=pn.Np, n=pn.Np)
                M_outflow = diags([Q_out], [0], format='csr')
                A_mat = ad.A + M_source - M_outflow
                # `ad.b` already contains the transient term contributions using the current
                # `ad['pore.concentration']` (previous time step).
                b_vec = ad.b + A2

                # Enforce boundary conditions for A matrix
                A_lil = A_mat.tolil() # Converts matrix from CSR to LIL format for faster row operations
                A_lil[inlet_pores, :] = 0.0 # Zeros row for boundary pores
                A_lil[inlet_pores, inlet_pores] = 1.0 # Puts 1s on the diagonal
                A_mat = A_lil.tocsr() # Converts back to CSR format

                # Enforce boundary conditions for b vector
                b_vec[inlet_pores] = 0.0
                
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
                mass_to_extract = params['k'] * driving_force * pn['pore.volume'] * dt
                phase[f'pore.{solute_name}_available'] -= np.minimum(mass_to_extract, phase[f'pore.{solute_name}_available'])

                # Store for data visualisation
                if solute_name == 'acids':
                    self.time_steps.append(t)
                self.concentrations[solute_name].append(C_new.copy())
                self.total_extracted += np.mean(np.maximum(mass_to_extract, 0))      

                # Velocity through each throat
                u_throats = np.abs(phase['throat.hydraulic_flow']) / pn['throat.cross_sectional_area']

                # TODO: Tune fines migration -> clogging
                # 1. Identify which pore is "Upstream" for every throat
                # pn['throat.conns'] is (Nt, 2). 
                # flow > 0 means water goes from col 0 to col 1.
                flow = phase['throat.hydraulic_flow']
                upstream_pores = np.where(flow > 0, pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])

                entrainment_rate = 0.001 # Rate at which fines are dislodged once u > critical_velocity
                critical_velocity = 0.1 # Speed above which fines are dislodged

                # 2. Calculate entrainment for every throat (as a mass per second)
                # mobile_rate is (Nt,)
                mobile_rate = np.where(
                    u_throats > critical_velocity, 
                    entrainment_rate, 
                    0
                )

                # 3. Calculate actual mass to move this step (kg)
                # We pull from the 'fines_mass' of the upstream_pores
                mass_to_pull = phase['pore.fines_mass'][upstream_pores] * mobile_rate * dt

                # 4. Update the Pores (Mass Conservation)
                # Since multiple throats might pull from the same pore, use np.add.at for safety
                np.add.at(phase['pore.fines_mass'], upstream_pores, -mass_to_pull)
                phase['pore.fines_mass'] = np.maximum(phase['pore.fines_mass'], 0)

                # Identify downstream pores
                downstream_pores = np.where(flow > 0, pn['throat.conns'][:, 1], pn['throat.conns'][:, 0])

                # Finds small throats
                is_small = pn['throat.diameter'] < 100e-6

                # Add to clogs
                phase['throat.clogged_mass'] += mass_to_pull * is_small

                # Add the rest to the next pore's "available" fines
                np.add.at(phase['pore.fines_mass'], downstream_pores, mass_to_pull * (~is_small))

                # Conductance decreases with clogging ratio in accordance with power law
                clogging_ratio = phase['throat.clogged_mass'] / max_clog_capacity
                phase['throat.hydraulic_conductance'] *= (1-clogging_ratio)**3

                phase.regenerate_models(propnames='throat.hydraulic_conductance')

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

    def generate_temperature_animation(self, save_path='temperature_debug.gif', interval=80):
        """
        Side-by-side animation of pore temperature heatmaps:
        - unclipped: raw T_new from the thermal linear solve
        - clipped: after applying the [20C, 95C] clamp
        """
        if not self.temperature_fields.get('clipped'):
            raise RuntimeError(
                "No temperature fields stored. Run `brew()` first (or ensure the updated code path executes)."
            )

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

        # Set color limits for easier visual comparison.
        cl_vals = np.array([np.min(arr) for arr in self.temperature_fields['clipped'][:n_frames]])
        cl_vals2 = np.array([np.max(arr) for arr in self.temperature_fields['clipped'][:n_frames]])

        cl_vmin = float(np.min(cl_vals))
        cl_vmax = float(np.max(cl_vals2))

        # Initial frames
        T_cl0 = self.temperature_fields['clipped'][0]

        grid_cl0 = griddata((y, z), T_cl0, (xi, zi), method='linear')

        fig, axes = plt.subplots(1, 1, figsize=(12, 7))
        ax_cl = axes[0]

        im_cl = ax_cl.imshow(
            grid_cl0, extent=(y.min(), y.max(), z.min(), z.max()),
            origin='lower', aspect='auto', cmap='magma', vmin=cl_vmin, vmax=cl_vmax
        )

        plt.colorbar(im_cl, ax=ax_cl, label='Temperature [C] (clipped)')

        ax_cl.set_title(f'Clipped mean: {np.mean(T_cl0):.2f}C')
        ax_cl.set_xlabel('Width [m]')
        ax_cl.set_ylabel('Bed Depth [m]')

        def update(frame):
            T_cl = self.temperature_fields['clipped'][frame]

            grid_cl = griddata((y, z), T_cl, (xi, zi), method='linear')

            im_cl.set_array(grid_cl)
            ax_cl.set_title(f'Clipped mean: {np.mean(T_cl):.2f}C')
            return [im_cl]

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

        # Plot 5: Mean temperature variation over time
        for temp in self.temperature_variation.keys():
            axes[0,1].plot(self.time_steps, self.temperature_variation[temp], 'o-', alpha=0.7, label=temp)
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Mean temperature')
        axes[0,1].set_title('Mean temperature variation over time')
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

        flow = phase['throat.hydraulic_flow']

        # Find all throats connecting the top half of the bed to the bottom half
        z_coords = pn['pore.coords'][:, 2]
        mid_z = np.median(z_coords)

        top_pores = np.where(z_coords > mid_z)[0]
        bot_pores = np.where(z_coords <= mid_z)[0]

        # Throats that cross the middle plane
        cross_throats = pn.find_connected_pores(top_pores, bot_pores)
        # Flow rate
        Q_total = np.sum(flow[cross_throats])

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
        print(f"Corrected Residence Time: {V_total / Q_total} seconds")
        if self.concentrations:
            for solute in self.solute_classes.keys():
                print(f" {solute} statistics:")
                print("Total mass to be extracted: ", np.sum(self.initial_mass))
                print(f"Extracted mass: {self.total_extracted}")
                print(f"EY: {self.total_extracted/np.sum(self.initial_mass):%}")
                print(f"TDS: {(self.total_extracted / (self.brew_time*self.pour_rate)):%}")
                print()
        print(f"{'='*60}\n")