from cmath import phase
import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import scipy.stats
from pypardiso import spsolve

class Simulation:
    def __init__(self, domain_shape=[100,100,75], porosity = 0.44, temperature = 95, particle_size_dist = 'twin_lognormal', solute_classes=None):
        self.shape = domain_shape
        self.porosity = porosity
        self.temperature = temperature
        self.particle_size_dist = particle_size_dist
        self.time_steps = []
        self.concentrations = {
            'acids': [],
            'sugars': [],
            'melanoidins': []
        }
        self.pressures = []
        self.outlet_concentrations = {
            'acids': [],
            'sugars': [],
            'melanoidins': []
        }
        self.total_extracted = 0.0
        if solute_classes is None:
            self.solute_classes = {
                'acids': {'k' : 0.05, 'concentration' : 400, 'c_sat' : 15.0},
                'sugars': {'k' : 0.02, 'concentration' : 200, 'c_sat' : 25.0},
                'melanoidins': {'k': 0.05, 'concentration' : 100, 'c_sat' : 10.0}
            }
        else:
            self.solute_classes = solute_classes

    def generate_coffee_bed(self):
        if self.particle_size_dist == "twin_lognormal":
            x_axis = np.arange(0.5,50,0.1)
            target_peak = scipy.stats.lognorm(s=0.4, scale=3.8)
            weight_target = 0.92
            fines_peak = scipy.stats.lognorm(s=0.8, scale=0.96)
            weight_fines = 0.08

            probs = (target_peak.pdf(x_axis) * weight_target) + (fines_peak.pdf(x_axis) * weight_fines)
            probs = probs / probs.sum()
            custom_dist_object = scipy.stats.rv_discrete(name='coffee_dist', values=(x_axis, probs))

            im = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.5, porosity=self.porosity, dist=custom_dist_object)

        elif self.particle_size_dist == "bimodal":
            target_dist = scipy.stats.norm(loc=(650e-6/2)/1e-4, scale=0.5)
            fines_dist = scipy.stats.norm(loc=(100e-6/2)/1e-4, scale=1)
            im_main = ps.generators.polydisperse_spheres(shape=self.shape, r_min=2, porosity=self.porosity*0.9, dist=target_dist)
            im_fines = ps.generators.polydisperse_spheres(shape=self.shape, r_min=0.5, porosity=self.porosity*0.1, dist=fines_dist)

            im = np.logical_or(im_main, im_fines).astype(int)
        
        self.im = im
        return im
    
    def extract_network(self):
        snow_dict = ps.networks.snow2(self.im, voxel_size=1e-4, sigma=0.3, r_max=5) # 100 microns per voxel
        pn = op.io.network_from_porespy(snow_dict.network)
        pn['pore.diameter'] = pn['pore.inscribed_diameter']
        pn['throat.diameter'] = pn['throat.inscribed_diameter']

        self.pn = self._ensure_connectivity(pn)

        adj_matrix = pn.create_adjacency_matrix(weights=None, fmt='csr')
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        
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
        pn.add_model(propname='throat.cross_sectional_area',
                    model=op.models.geometry.throat_cross_sectional_area.cylinder,
                    throat_diameter='throat.diameter')
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
        
        mu_ref = 1.002e-3  # Pa·s at 20°C
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
        
        # Add density variation with temperature
        rho_ref = 1000  # kg/m³ at 20°C
        alpha = 0.0002  # volumetric expansion coefficient
        rho = rho_ref / (1 + alpha * (self.temperature - 20))
        phase['pore.density'] = rho
        
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
        
        phase.add_model(propname='pore.R_source',
                        model=op.models.physics.source_terms.linear,
                        X='pore.X',
                        A1='pore.A1',
                        A2='pore.A2',
                        regen_mode='deferred')
        
    def brew(self, brew_time, pour_rate, time_steps=1):
        pn = self.pn
        phase = self.phase
        coords = pn.coords

        phase['pore.concentration'] = 0.0
        
        # Define boundary conditions based on V60 geometry
        tol = 1e-6
        inlet_pores = pn.pores()[coords[:, 2] >= coords[:, 2].max() - tol]
        outlet_pores = pn.pores()[coords[:, 2] <= coords[:, 2].min() + tol]
        
        dt = brew_time / time_steps
        self.dt = dt

        inlet_pressure = 1000 * (pour_rate / 50)  # Pa, scaled relative to 50 mL/s
            
        # Run Stokes flow
        flow = op.algorithms.StokesFlow(network=pn, phase=phase)

        flow.settings['solver'] = 'spsolve'
        flow.settings['spsolve'] = spsolve

        flow.set_value_BC(pores=inlet_pores, values=inlet_pressure)
        flow.set_value_BC(pores=outlet_pores, values=0.0)
        flow.run()
        self.pressures.append(flow['pore.pressure'].copy())

        self.pressures.append(flow['pore.pressure'].copy())

        phase['pore.pressure'] = flow['pore.pressure']
        phase.regenerate_models(propnames=['throat.ad_dif_conductance'])

        # Implement transient advection diffusion solver
        tad = op.algorithms.TransientAdvectionDiffusion(network=pn, phase=phase)

        print(tad.settings.keys())
        tad.settings['solver'] = 'spsolve'
        tad.settings['spsolve'] = spsolve
        tad.settings['verbose'] = True

        for solute_name, params in self.solute_classes.items():
            tad['pore.concentration'] = 0.0
            C_initial = tad['pore.concentration'].copy()
            initial_mass, phase[f'pore.{solute_name}_available'] = float(params['concentration']) * pn['pore.volume'], float(params['concentration']) * pn['pore.volume']
            phase[f'pore.{solute_name}_concentration'] = 0.0

            for step in range(time_steps):
                t = (step + 1) * dt

                # Calculate extraction source term R_source
                #A2, A1 = np.zeros(pn.Np), np.zeros(pn.Np)
                #A2 += params['k'] * params['c_sat']
                #A1 += -params['k']
                remaining_ratio = phase[f'pore.{solute_name}_available'] / initial_mass
                phase['pore.A2'] = params['k'] * params['c_sat'] * remaining_ratio
                phase['pore.A1'] = -params['k'] * remaining_ratio
                phase['pore.X'] = phase[f'pore.{solute_name}_concentration']
                """
                R_source = np.zeros(pn.Np)
                available = phase[f'pore.{solute_name}_available']
                extracted = params['k'] * available * dt
                extracted = np.clip(extracted, 0, available)
                phase[f'pore.{solute_name}_available'] -= extracted
                R_source += extracted / dt

                phase['pore.R'] = R_source
                """
                phase.regenerate_models(propnames=['pore.R_source'])
                #print(np.mean(phase['pore.R']))
                tad.set_source(propname='pore.R_source', pores=pn.pores())
                print(f"Solving for {solute_name}, time step {step+1}")
                tad.run(x0=C_initial,tspan=[t-dt, t])

                phase['pore.concentration'] = tad['pore.concentration'].copy()
                C_initial = tad['pore.concentration'].copy()
                phase[f'pore.{solute_name}_available'] -= params['k'] * (params['c_sat'] - tad['pore.concentration']) * pn['pore.volume'] * dt

                # Store results
                if solute_name == 'acids':
                    self.time_steps.append(t)
                self.outlet_concentrations[solute_name].append(phase['pore.concentration'][outlet_pores].mean())
                self.concentrations[solute_name].append(phase['pore.concentration'].copy())

    def mass_balance(self):
        mass_in_fluid = np.sum(self.phase['pore.concentration'] * self.pn['pore.volume'])
        total_mass_extracted = 0.0
        for solute in self.solute_classes.keys():
            initial_mass = float(self.solute_classes[solute]['concentration']) * self.pn['pore.volume'].sum()
            remaining_mass = np.sum(self.phase[f'pore.{solute}_available'])
            total_mass_extracted += (initial_mass - remaining_mass)
        return mass_in_fluid, total_mass_extracted

    def plot_results(self):
        pn = self.pn
        coords = pn.coords
        
        n_steps = len(self.time_steps)
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

        # Plot 5: Outlet mass frequency
        for solute in self.solute_classes.keys():
            mass = self.concentrations[solute][-1] * pn['pore.volume']
            axes[0,1].hist(mass, bins=40, edgecolor='black', alpha=0.7, label=solute)
        axes[0,1].set_xlabel('Outlet mass (kg)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Outlet Mass Frequency')
        axes[0,1].grid(True, alpha=0.3)

        # Plot 6: Outlet mass with respect to position
        for solute in self.solute_classes.keys():
            mass = self.concentrations[solute][-1] * pn['pore.volume']
            axes[0, 2].scatter(coords[:, 2], mass, alpha=0.5, s=10, label=solute)
        axes[0, 2].set_xlabel('Z-coordinate (voxels)')
        axes[0, 2].set_ylabel('Mass (kg)')
        axes[0, 2].set_title('Mass Profile Along Flow (Final)')
        axes[0, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    def print_statistics(self):
        pn = self.pn
        print(f"\n{'='*60}")
        print(f"COFFEE BED SIMULATION STATISTICS")
        print(f"{'='*60}")
        print(f"Temperature: {self.temperature}°C")
        print(f"Target porosity: {self.porosity:.1%}")
        print(f"Actual porosity: {self.im.sum() / self.im.size:.1%}")
        print(f"Domain shape: {self.shape}")
        print(f"\nNetwork Properties:")
        print(f"  Number of pores: {pn.Np}")
        print(f"  Number of throats: {pn.Nt}")
        print(f"  Avg pore diameter: {pn['pore.diameter'].mean():.5f} voxels")
        print(f"  Avg throat diameter: {pn['throat.diameter'].mean():.5f} voxels")
        print(f"  Pore diameter range: {pn['pore.diameter'].min():.5f} - {pn['pore.diameter'].max():.5f}")
        print(f"  Throat diameter range: {pn['throat.diameter'].min():.5f} - {pn['throat.diameter'].max():.5f}")
        print(f"\nPhase Properties at {self.temperature}°C:")
        print(f"  Viscosity: {self.phase['pore.viscosity'][0]:.3e} Pa·s")
        print(f"  Diffusivity: {self.phase['pore.diffusivity'][0]:.3e} m²/s")
        print(f"\nBrewing Progress:")
        print(f"  Total brew time: {self.time_steps[-1]:.1f}s")
        if self.concentrations:
            for solute in self.solute_classes.keys():
                print(f" {solute} statistics:")
                print(f"  Final mean concentration: {self.concentrations[solute][-1].mean():.3f}")
                print(f"  Final max concentration: {self.concentrations[solute][-1].max():.3f}")
                print(f"  Mass in fluid: {np.sum(self.concentrations[solute][-1] * pn['pore.volume']):.15f}")
                print()
        print(f"{'='*60}\n")