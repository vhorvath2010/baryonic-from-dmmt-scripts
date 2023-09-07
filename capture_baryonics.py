import torch
import torch_geometric
import yt
import collections
import bisect

yt.enable_plugins()
enzo_data = yt.load("~jw254/data/SG256-v3/DD????/output_????") # Set to proper enzo dataset
graphs = torch.load('SG256_pruned.pt') # Load graphs from prune_and_gen step

# Load enzo snapshots
snapshots = {}
for snapshot in enzo_data:
    snapshot.add_particle_filter('p2')
    snapshot.add_particle_filter('p3')
    snapshots[snapshot.current_redshift] = snapshot
    
# Capture baryonics from Enzo snapshots
# Adjust to whatever values are in prune_and_gen_graphs logs
rVir_units = 'kpc'
pos_units = 'unitary'

for graph in graphs:
    y = []
    for halo_info in graph.x:
        # Select snapshot with closest redshift
        halo_redshift = halo_info[1].item()
        snapshot_redshift = min(snapshots.keys(), key=lambda k: abs(k - halo_redshift))
        
        if (abs(snapshot_redshift - halo_redshift) > 0.01):
            print(f"WARNING: High redshift mismatch detected: snapshot: {snapshot_redshift} vs halo: {halo_redshift}")
        
        # Load Stellar mass from pop2 mass
        halo_pos = snapshot.arr(halo_info[2:5].tolist(), pos_units)
        halo_rVir = snapshot.arr(halo_info[-1].item(), rVir_units)
        sph = snapshot.sphere(halo_pos, halo_rVir)
        stellar_mass = sph.quantities.total_quantity(('p2','particle_mass')) 
        
        # Acquire stellar mass in solar masses
        stellar_mass = stellar_mass.value.item() * 5.0000000025E-34
        y.append(stellar_mass)
    y = torch.tensor(y)
    graph.y = y

# Save output
print(f"Saving {len(graphs)} graphs...")
torch.save(graphs, 'SG256_Full_Graphs.pt') # Set output dir
print("Graphs saved with y values!")
print("Y is form: [stellar_mass (MSun)]")
