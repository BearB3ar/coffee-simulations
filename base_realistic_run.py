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

class Simulation:
    def __init__(self, domain_shape=[300,300,300], porosity = 0.46, temperature = 95, particle_size_dist = 'twin_lognormal', solute_classes=None):
        self.shape = domain_shape
        self.porosity = porosity
        self.temperature = temperature
        self.particle_size_dist = particle_size_dist
        self.time_steps = []
        self.concentrations = {
            'acids': [],
        }
        self.temperature_variation = {
            "unclipped": [],
            "final": []
        }
        self.pressures = []
        self.total_extracted = 0.0
        if solute_classes is None:
            self.solute_classes = {
                'acids': {'k' : 10e-3, 'concentration' : 400, 'c_sat' : 150.0},
            }
        else:
            self.solute_classes = solute_classes

    def generate_coffee_bed(self):
        shape = self.shape
        if self.particle_size_dist == "twin_lognormal":
            x_axis = np.arange(0.5,100,0.5)
            target_peak = scipy.stats.lognorm(s=0.42, scale=(650e-6/2)/1e-4)
            weight_target = 0.92
            fines_peak = scipy.stats.lognorm(s=0.8, scale=(100e-6/2)/1e-4)
            weight_fines = 0.08

            probs = (target_peak.pdf(x_axis) * weight_target) + (fines_peak.pdf(x_axis) * weight_fines)
            probs = probs / probs.sum()
            custom_dist_object = scipy.stats.rv_discrete(name='coffee_dist', values=(x_axis, probs))

            im = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.05, porosity=self.porosity, dist=custom_dist_object)

        elif self.particle_size_dist == "bimodal":
            target_dist = scipy.stats.norm(loc=(650e-6/2)/1e-4, scale=0.5)
            fines_dist = scipy.stats.norm(loc=(100e-6/2)/1e-4, scale=1)
            im_main = ps.generators.polydisperse_spheres(shape=self.shape, r_min=2, porosity=self.porosity*0.9, dist=target_dist)
            im_fines = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.5, porosity=self.porosity*0.1, dist=fines_dist)

            im = np.logical_or(im_main, im_fines).astype(int)

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

        # Distance measurer from cone_mask boundary inwards
        dt = spim.distance_transform_edt(cone_mask)

        # Find spheres using spim logic rather than searching by voxels and identify their centroids
        labels, n_spheres = spim.label(grounds)
        centroids = spim.center_of_mass(grounds, labels, range(1, n_spheres+1))

        new_im = im.copy()

        for i, center in enumerate(centroids):
            # z,y,x is order specified by centroids
            z, y, x = int(center[0]), int(center[1]), int(center[2])
            
            # Find the distance to cone_mask boundary
            dist = dt[z,y,x]

            # Dynamic probability of removal based on distance from cone_mask
            p_remove = wall_porosity_boost * np.exp(-dist / decay_width)

            # Remove if randomly generated probability falls within probability of removal
            if np.random.random() < p_remove:
                new_im[labels == (i+1)] = 1

        self.im = new_im
    
    def plot_coffee_bed(self):
        im = self.im
        shape = self.shape

        fig, axes = plt.subplots(1, 3, figsize=(18,16))
        axes[0].imshow(im[shape[0]//2,:,:], cmap='magma')
        axes[0].set_title('XY Plane (Side)')
        axes[0].set_xlabel('Y')
        axes[0].set_ylabel('Z')

        axes[1].imshow(im[:,shape[1]//2,:], cmap='magma')
        axes[1].set_title('XY Plane (Side)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')

        axes[2].imshow(im[:,:,shape[2]//2], cmap='magma')
        axes[2].set_title('XY Plane (Top-down)')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')

        plt.tight_layout()
        plt.show()
    
    def extract_network(self):
        snow_dict = ps.networks.snow2(self.im, voxel_size=1e-5, sigma=0.3, r_max=5) # 10 microns per voxel
        pn = op.io.network_from_porespy(snow_dict.network)
        pn['pore.diameter'] = pn['pore.inscribed_diameter']
        pn['throat.diameter'] = pn['throat.inscribed_diameter']

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

        pn['pore.diameter'][pn['pore.diameter'] < 1e-6] = 1e-6
        pn['throat.diameter'][pn['throat.diameter'] < 1e-6] = 1e-6
        
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

        phase['pore.temperature'] = self.temperature
        phase['throat.temperature'] = self.temperature
        
        mu_ref = 0.95e-3  # Pa·s at 20°C
        mu = mu_ref * np.exp(-0.03 * (self.temperature - 20))
        
        phase['pore.viscosity'] = mu
        phase['throat.viscosity'] = mu
        
        # Temperature-dependent diffusivity (m²/s)
        # Stokes-Einstein: D ~ T / mu
        # Solutes in water scale roughly as D ~ 5e-10 * (T/293) / (mu/0.001)
        D_ref = 5e-10  # m²/s at 20°C for typical coffee solutes
        D = D_ref * (self.temperature + 273.15) / 293.15 * (1.002e-3 / mu)
        
        phase['pore.diffusivity'] = D
        phase['throat.diffusivity'] = D

        thermal_conductivity = -8.354e-6 * (self.temperature**2) + 6.53e-3*self.temperature - 0.5981
        phase['pore.thermal_conductivity'] = thermal_conductivity
        phase['throat.thermal_conductivity'] = thermal_conductivity
        
        self.phase = phase
        return phase
    
    def add_physics_models(self):
        phase = self.phase
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

        # Water parameters
        rho_w = 965
        cp_w = 4190

        dt = brew_time / time_steps
        self.dt = dt
        def solve_flow(self, pour_rate):
            coords = pn.coords

            phase['pore.concentration'] = 0.0
            
            # Define boundary conditions based on V60 geometry
            tol = 1e-6
            inlet_pores = pn.pores()[coords[:, 2] >= coords[:, 2].max() - tol]
            outlet_pores = pn.pores()[coords[:, 2] <= coords[:, 2].min() + tol]

            inlet_pressure = 1000 * (pour_rate / 200) + 500  # Pa, scaled relative to 50 mL/s
            # Run Stokes flow
            flow = op.algorithms.StokesFlow(network=pn, phase=phase)

            flow.settings['solver'] = 'spsolve'
            flow.settings['spsolve'] = pypardiso_spsolve

            flow.set_value_BC(pores=inlet_pores, values=inlet_pressure)
            flow.set_value_BC(pores=outlet_pores, values=0.0)
            flow.run()
            self.pressures.append(flow['pore.pressure'].copy())

            phase['pore.pressure'] = flow['pore.pressure']
            phase.regenerate_models(propnames=['throat.ad_dif_conductance'])

            return inlet_pores, outlet_pores

        # Implement transient advection diffusion solver
        tad = op.algorithms.TransientAdvectionDiffusion(network=pn, phase=phase)
        tad_thermo = op.algorithms.TransientAdvectionDiffusion(network=pn, phase=phase)
        tad_thermo.settings['conductance'] = 'throat.thermal_conductance'

        for solute_name, params in self.solute_classes.items():
            swelling_flag, temperature_flag = False, False

            # Initial setup for how much solute is available for extraction and how much has been extracted
            tad['pore.concentration'] = 0.0
            C_initial = tad['pore.concentration'].copy()
            initial_mass, phase[f'pore.{solute_name}_available'] = float(params['concentration']) * pn['pore.volume'], float(params['concentration']) * pn['pore.volume']
            self.initial_mass = initial_mass
            phase[f'pore.{solute_name}_concentration'] = 0.0

            initial_pore_diameter = pn['pore.diameter'].copy()
            initial_throat_diameter = pn['throat.diameter'].copy()
            T_prev_step = 1e99

            inlet_pores, outlet_pores = solve_flow(self, pour_rate)

            for step in range(time_steps):
                t = (step + 1) * dt

                # Swelling feature (disabled)
                if swelling_flag:
                    # Calculate swelling -> shrinkage of pores and throats
                    pn['throat.diameter'] = np.maximum(pn['throat.diameter']*shrink_factor, initial_throat_diameter * 0.985)
                    pn['pore.diameter'] = np.maximum(pn['pore.diameter']*shrink_factor, initial_pore_diameter * 0.985)

                    # Update geometry models
                    pn.regenerate_models()
                    phase.regenerate_models()

                    inlet_pores, outlet_pores = solve_flow(self, pour_rate)
                    if pn['throat.diameter'].all() <= 1e-6 and pn['pore.diameter'].all() <= 1e-5:
                        swelling_flag = False

                # Recalculation of viscosity and diffusivity
                if temperature_flag:
                    T_pore = phase['pore.temperature']
                    T_throat = phase['throat.temperature']

                    mu_ref = 0.95e-3  # Pa·s at 20°C
                    
                    phase['pore.viscosity'] = mu_ref * np.exp(-0.03 * (T_pore - 20))
                    phase['throat.viscosity'] = mu_ref * np.exp(-0.03 * (T_throat - 20))
                    
                    # Temperature-dependent diffusivity (m²/s)
                    # Stokes-Einstein: D ~ T / mu
                    # Solutes in water scale roughly as D ~ 5e-10 * (T/293) / (mu/0.001)
                    D_ref = 5e-10  # m²/s at 20°C for typical coffee solutes
                    
                    phase['pore.diffusivity'] = D_ref * (T_pore + 273.15) / 293.15 * (1.002e-3 / phase['pore.viscosity'])
                    phase['throat.diffusivity'] = D_ref * (T_throat + 273.15) / 293.15 * (1.002e-3 / phase['throat.viscosity'])

                    phase['pore.thermal_conductivity'] = -8.354e-6 * (T_pore**2) + (6.53e-3)*T_pore - 0.5981
                    phase['throat.thermal_conductivity'] = -8.354e-6 * (T_throat**2) + (6.53e-3)*T_throat - 0.5981

                    phase.regenerate_models(propnames="throat.hydraulic_conductance")
                    phase.regenerate_models(propnames="throat.diffusive_conductance")
                    phase.regenerate_models(propnames="throat.ad_dif_conductance")
                    phase.regenerate_models(propnames="throat.thermal_conductance")

                    inlet_pores, outlet_pores = solve_flow(self, pour_rate)
                    if np.array_equal(T_pore,T_prev_step):
                        temperature_flag = False
                    else:
                        T_prev_step = phase['pore.temperature'].copy()

                # Solve temperature variation
                tad_thermo._build_A()
                tad_thermo._build_b()

                # Calculate flow outflow, to be used in solute outflow
                Q_out = np.zeros(pn.Np)
                for pore in outlet_pores:
                    # Find all throats connected to this specific outlet pore
                    connected_throats = pn.find_neighbor_throats(pores=pore)
                    
                    # Since water is incompressible and the network ends here, 
                    # the absolute sum of flow in these throats is exactly the flow leaving into the cup.
                    Q_out[pore] = np.sum(np.abs(phase['throat.ad_dif_conductance'][connected_throats]))

                # 1. Calculate Thermal Accumulation (Density * Heat Capacity * Volume / dt)
                # We use the combined heat capacity of the water-filled pore and the solid grain
                # Porosity (phi) is handled via pore.volume (fluid) and solid_volume
                vol_term_thermal = (rho_w * cp_w * pn['pore.volume'] + rho_s * cp_s * (pn['pore.volume'] * ((1-self.porosity)/self.porosity))) / dt
                thermal_drain = Q_out * rho_w * cp_w

                # Add accumulation to LHS and RHS
                A_mat_T = tad_thermo.A + diags([vol_term_thermal], [0], format='csr') + diags([thermal_drain], [0], format='csr')
                b_vec_T = tad_thermo.b + (vol_term_thermal * phase['pore.temperature'])

                # Apply Boundary Conditions
                A_lil_T = A_mat_T.tolil()
                A_lil_T[inlet_pores, :] = 0.0
                A_lil_T[inlet_pores, inlet_pores] = 1.0
                A_mat_T = A_lil_T.tocsr()

                b_vec_T[inlet_pores] = 95 # 95°C Inlet

                # 4. Solve for Temperature
                T_new = pypardiso_spsolve(A_mat_T, b_vec_T)
                if np.isnan(T_new).any():
                    T_new = scipy_spsolve(A_mat_T, b_vec_T)

                self.temperature_variation['unclipped'].append(np.mean(T_new))
                T_new = np.clip(T_new, 293.15, 368.15) # Keep between 20C and 95C
                self.temperature_variation['final'].append(np.mean(T_new))

                phase['pore.temperature'] = T_new

                # Diagnostic for residence time during brewing phase
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

                # Build original version of A and b
                tad._build_A()
                tad._build_b()

                # Calculate A1 and A2 (Y = A1X + A2)
                placeholder = np.ones(pn.Np)
                remaining_ratio = phase[f'pore.{solute_name}_available'] / initial_mass
                A1 = -params['k'] * remaining_ratio * placeholder * pn['pore.volume']
                A2 = params['k'] * params['c_sat'] * remaining_ratio * placeholder * pn['pore.volume']

                # Calculate volume term
                vol_term = pn['pore.volume'] / dt
                vol_term_LHS = vol_term.copy()

                # Add transient and extraction terms to all pores
                M_source = spdiags(data=vol_term_LHS-A1, diags=0, m=pn.Np, n=pn.Np)
                M_outflow = diags([Q_out], [0], format='csr')
                A_mat = tad.A + M_source - M_outflow
                b_vec = tad.b + A2 + (vol_term * C_initial)

                # Enforce boundary conditions for A matrix
                A_lil = A_mat.tolil() # Converts matrix from CSR to LIL format for faster row operations
                A_lil[inlet_pores, :] = 0.0 # Zeros row for boundary pores
                A_lil[inlet_pores, inlet_pores] = 1.0 # Puts 1s on the diagonal
                A_mat = A_lil.tocsr() # Converts back to CSR format

                # Enforce boundary conditions for b vector
                b_vec[inlet_pores] = 0.0
                
                # Error handling for 0 on the diagonals
                diagonals = A_mat.diagonal() 
                zero_diags = np.where(np.abs(diagonals) < 1e-20)[0]
                if len(zero_diags) > 0:
                    print("Remaining error of near 0s on the diagonal: ", len(zero_diags))
                    fix_array = np.zeros(pn.Np)
                    fix_array[zero_diags] = 1.0
                    A_mat = A_mat + diags([fix_array] [0], format='csr')

                # Solve for C_new, preferably with pypardiso solver but with scipy solver if any errors
                C_new = pypardiso_spsolve(A_mat, b_vec)
                if np.isnan(C_new).any():
                    C_new = scipy_spsolve(A_mat, b_vec)

                # Update for next time step
                tad['pore.concentration'] = C_new.copy()
                C_initial = C_new.copy()
                phase[f'pore.{solute_name}_concentration'] = C_new.copy()

                # Ensures extraction does not exceed available
                driving_force = np.maximum(0, params['c_sat'] - C_new)
                mass_to_extract = params['k'] * driving_force * pn['pore.volume'] * dt
                phase[f'pore.{solute_name}_available'] -= np.minimum(mass_to_extract, phase[f'pore.{solute_name}_available'])

                # Store for data visualisation
                if solute_name == 'acids':
                    self.time_steps.append(t)
                self.concentrations[solute_name].append(C_new.copy())
                self.total_extracted += np.mean(np.minimum(mass_to_extract, phase[f'pore.{solute_name}_available']))      

    def mass_balance(self):
        mass_in_fluid = np.sum(self.phase['pore.concentration'] * self.pn['pore.volume'])
        total_mass_extracted = 0.0
        for solute in self.solute_classes.keys():
            initial_mass = float(self.solute_classes[solute]['concentration']) * self.pn['pore.volume'].sum()
            remaining_mass = np.sum(self.phase[f'pore.{solute}_available'])
            total_mass_extracted += (initial_mass - remaining_mass)
        return mass_in_fluid, total_mass_extracted

    def generate_brewing_animation(self, solute_name='acids'):
        # 1. Get coordinates (assuming X and Z for a side-profile heatmap)
        coords = self.pn['pore.coords']
        y = coords[:, 1]
        z = coords[:, 2]
        
        # 2. Define the grid where we want to "paint" the heatmap
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

        # 3. Update function for the animation
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
        #plt.show()

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
        """for solute in self.solute_classes.keys():
            mass = self.concentrations[solute][-1] * pn['pore.volume']
            axes[2, 1].scatter(coords[:, 2], mass, alpha=0.5, s=10, label=solute)"""
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

        P = phase['pore.pressure']
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

        print(f"\n{'='*60}")
        print(f"COFFEE BED SIMULATION STATISTICS")
        print(f"{'='*60}")
        print(f"Initial temperature: {self.temperature}°C")
        print(f"Target porosity: {self.porosity:.1%}")
        print(f"Actual porosity: {self.im[self.cone_mask].sum() / self.cone_mask.sum():.1%}")
        print(f"Domain shape: {self.shape}")
        print(f"\nNetwork Properties:")
        print(f"  Number of pores: {pn.Np}")
        print(f"  Number of throats: {pn.Nt}")
        print(f"  Pore volume: {pn['pore.volume']}")
        print(f"  Avg pore diameter: {pn['pore.diameter'].mean():.7f} voxels")
        print(f"  Avg throat diameter: {pn['throat.diameter'].mean():.7f} voxels")
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
                print("Total mass to be extracted: ", np.mean(self.initial_mass))
                print(f"Extracted mass: {self.total_extracted}")
                print(f"EY: {self.total_extracted/np.mean(self.initial_mass):%}")
                print(f"TDS: {(self.total_extracted / (self.brew_time*self.pour_rate)):%}")
                print()
        print(f"{'='*60}\n")