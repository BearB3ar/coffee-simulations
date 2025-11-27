import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import scipy.stats

class Simulation:
    def __init__(self, domain_shape=[150,150,100], porosity = 0.4, temperature = 95, particle_size_dist = 'bimodal'):
        self.shape = domain_shape
        self.porosity = porosity
        self.temperature = temperature
        self.particle_size_dist = particle_size_dist
        self.time_steps = []
        self.concentrations = []
        self.pressures = []

    def generate_coffee_bed(self):
        if self.particle_size_dist == 'bimodal':
            size1 = scipy.stats.norm(loc=1.5, scale=0.3)
            size2 = scipy.stats.norm(loc=3.5, scale=0.5)

            im1 = ps.generators.polydisperse_spheres(shape=self.shape, r_min=1.0, porosity=self.porosity*0.4, dist=size1)
            im2 = ps.generators.polydisperse_spheres(shape=self.shape, r_min=2.0, porosity=self.porosity*0.6, dist=size2)
            im = np.logical_or(im1, im2).astype(int)
        elif self.particle_size_dist == 'lognormal':
            dist = scipy.stats.lognorm(s=0.5, scale=2.5)
            im = ps.generators.polydisperse_spheres(shape=self.shape, r_min=1.0, porosity=self.porosity, dist=dist)

        else: 
            # Default to monodisperse
            im = ps.generators.polydisperse_spheres(shape=self.shape, r_min=2.5, porosity=self.porosity)

        self.im = im
        return im
    
    def extract_network(self):
        snow_dict = ps.networks.snow2(self.im, voxel_size=1e-4) # 100 microns per voxel
        pn = op.io.network_from_porespy(snow_dict.network)
        pn['pore.diameter'] = pn['pore.inscribed_diameter']
        pn['throat.diameter'] = pn['throat.inscribed_diameter']

        self._ensure_connectivity(pn)
        self.pn = pn
        return pn
    
    def _ensure_connectivity(self, pn):
        adj_matrix = pn.create_adjacency_matrix(weights=None, fmt='csr')
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        
        if n_components > 1:
            largest_cluster = np.argmax(np.bincount(labels))
            pores_in_largest = np.where(labels == largest_cluster)[0]
            throats_in_largest = pn.find_neighbor_throats(pores=pores_in_largest, mode='intersection')
            
            pn_new = op.network.Network(
                conns=pn['throat.conns'][throats_in_largest],
                coords=pn['pore.coords'][pores_in_largest]
            )
            
            pn_new['pore.diameter'] = pn['pore.diameter'][pores_in_largest]
            pn_new['throat.diameter'] = pn['throat.diameter'][throats_in_largest]
            if 'pore.inscribed_diameter' in pn:
                pn_new['pore.inscribed_diameter'] = pn['pore.inscribed_diameter'][pores_in_largest]
            if 'throat.inscribed_diameter' in pn:
                pn_new['throat.inscribed_diameter'] = pn['throat.inscribed_diameter'][throats_in_largest]
            
            # Replace
            pn.clear()
            pn.update(pn_new)

    def add_geometry_models(self):
        pn = self.pn
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
        
    def brew(self, brew_time, pour_rate, num_steps=1):
        pn = self.pn
        phase = self.phase
        coords = pn.coords
        
        # Define boundary conditions based on V60 geometry
        tol = 1e-6
        inlet_pores = pn.pores()[coords[:, 2] >= coords[:, 2].max() - tol]
        outlet_pores = pn.pores()[coords[:, 2] <= coords[:, 2].min() + tol]
        
        dt = brew_time / num_steps
        self.dt = dt
        
        for step in range(num_steps):
            t = step * dt
            
            # Adjust inlet pressure based on pour rate
            # Pressure ~ pour_rate (proportional to flow rate)
            inlet_pressure = 1000 * (pour_rate / 50)  # Pa, scaled relative to 50 mL/s
            
            # Run Stokes flow
            flow = op.algorithms.StokesFlow(network=pn, phase=phase)
            flow.set_value_BC(pores=inlet_pores, values=inlet_pressure)
            flow.set_value_BC(pores=outlet_pores, values=0.0)
            flow.run()
            
            # Run advection-diffusion
            ad = op.algorithms.AdvectionDiffusion(network=pn, phase=phase)
            
            # Initial concentration for this step
            if step == 0:
                # Initialize with solute at inlet
                phase['pore.concentration'] = 0.0
                phase.update_model(propnames='pore.concentration')
            
            ad.set_value_BC(pores=inlet_pores, values=1.0)  # incoming water (high concentration)
            ad.set_value_BC(pores=outlet_pores, values=0.0)  # outlet
            ad.run()
            
            # Store results
            self.time_steps.append(t)
            self.concentrations.append(ad['pore.concentration'].copy())
            self.pressures.append(phase['pore.pressure'].copy())

    def plot_results(self):
        pn = self.pn
        coords = pn.coords
        
        n_steps = len(self.time_steps)
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Plot 1: Concentration distribution at final step
        C_final = self.concentrations[-1]
        axes[0, 0].hist(C_final, bins=40, edgecolor='black', alpha=0.7, color='brown')
        axes[0, 0].set_xlabel('Pore concentration (normalized)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Final Concentration Distribution (t={self.time_steps[-1]:.1f}s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Concentration evolution
        C_mean = [C.mean() for C in self.concentrations]
        C_max = [C.max() for C in self.concentrations]
        axes[0, 1].plot(self.time_steps, C_mean, 'o-', label='Mean concentration', linewidth=2)
        axes[0, 1].plot(self.time_steps, C_max, 's--', label='Max concentration', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Concentration (normalized)')
        axes[0, 1].set_title('Extraction Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Pressure profile (final)
        P_final = self.pressures[-1]
        axes[1, 0].scatter(coords[:, 2], P_final, alpha=0.5, s=10, color='red')
        axes[1, 0].set_xlabel('Z-coordinate (voxels)')
        axes[1, 0].set_ylabel('Pressure (Pa)')
        axes[1, 0].set_title('Pressure Profile Along Flow Direction (Final)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Concentration profile (final)
        axes[1, 1].scatter(coords[:, 2], C_final, alpha=0.5, s=10, color='brown')
        axes[1, 1].set_xlabel('Z-coordinate (voxels)')
        axes[1, 1].set_ylabel('Concentration (normalized)')
        axes[1, 1].set_title('Extraction Profile Along Flow (Final)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Network statistics
        pore_diam = pn['pore.diameter']
        throat_diam = pn['throat.diameter']
        axes[2, 0].hist(pore_diam, bins=30, alpha=0.6, label='Pores', edgecolor='black')
        axes[2, 0].hist(throat_diam, bins=30, alpha=0.6, label='Throats', edgecolor='black')
        axes[2, 0].set_xlabel('Diameter (voxels)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Pore/Throat Size Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Extraction kinetics (cumulative extraction)
        extraction_rate = [C.mean() for C in self.concentrations]
        axes[2, 1].plot(self.time_steps, extraction_rate, 'o-', linewidth=2, color='darkgreen')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Average extracted concentration')
        axes[2, 1].set_title('Extraction Kinetics')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].fill_between(self.time_steps, 0, extraction_rate, alpha=0.3, color='darkgreen')
        
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
        print(f"  Avg pore diameter: {pn['pore.diameter'].mean():.3f} voxels")
        print(f"  Avg throat diameter: {pn['throat.diameter'].mean():.3f} voxels")
        print(f"  Pore diameter range: {pn['pore.diameter'].min():.3f} - {pn['pore.diameter'].max():.3f}")
        print(f"  Throat diameter range: {pn['throat.diameter'].min():.3f} - {pn['throat.diameter'].max():.3f}")
        print(f"\nPhase Properties at {self.temperature}°C:")
        print(f"  Viscosity: {self.phase['pore.viscosity'][0]:.3e} Pa·s")
        print(f"  Diffusivity: {self.phase['pore.diffusivity'][0]:.3e} m²/s")
        print(f"\nBrewing Progress:")
        if self.concentrations:
            print(f"  Final mean concentration: {self.concentrations[-1].mean():.3f}")
            print(f"  Final max concentration: {self.concentrations[-1].max():.3f}")
            print(f"  Total brew time: {self.time_steps[-1]:.1f}s")
        print(f"{'='*60}\n")

