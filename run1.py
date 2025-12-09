import base_realistic_run 
import matplotlib.pyplot as plt

sim = base_realistic_run.Simulation(
    porosity = 0.42,
    temperature = 95,
    particle_size_dist = 'bimodal'
)

sim.generate_coffee_bed()

"""
plt.imshow(sim.im[:, :, sim.shape[2]//2])
plt.title('Packed spheres slice')
plt.show()
"""

sim.extract_network()
sim.add_geometry_models()
sim.phase()
sim.add_physics_models()


"""pn = sim.pn
print(f"\nDetailed Network Diagnostics:")
print(f"  Total pores: {pn.Np}")
print(f"  Total throats: {pn.Nt}")
print(f"  Avg coordination number: {2*pn.Nt/pn.Np:.2f}")
print(f"  Pore diameter: {pn['pore.diameter'].min():.3f} - {pn['pore.diameter'].max():.3f}")
print(f"  Throat diameter: {pn['throat.diameter'].min():.3f} - {pn['throat.diameter'].max():.3f}")

g = sim.phase["throat.hydraulic_conductance"]
print(f"  Conductance: {g.min():.3e} - {g.max():.3e}")
print(f"  Conductance ratio: {g.max()/g.min():.0e}")
print(f"  Zero/negative conductances: {(g <= 0).sum()}")

h = sim.phase["throat.diffusive_conductance"]
print(f"  Conductance: {h.min():.3e} - {h.max():.3e}")
print(f"  Conductance ratio: {h.max()/h.min():.0e}")
print(f"  Zero/negative conductances: {(h <= 0).sum()}")
"""

sim.brew(
    brew_time = 150,
    pour_rate = 1,
    num_pours = 200
)

print(f"\nTotal extracted solutes: {sim.total_extracted:.15f} mg")

#sim.plot_results()
sim.print_statistics()