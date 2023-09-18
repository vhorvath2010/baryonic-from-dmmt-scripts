import torch
import torch_geometric
import yt
import collections
import bisect
import multiprocessing as mp
from functools import partial

# Function to load graph baryonics for a certain snapshot
def snapshot_baryonics(graphs, ys, snapshot):
    snapshot.add_particle_filter('p2')
    snapshot.add_particle_filter('p3')
    for graph_idx, graph in enumerate(graphs):
        for halo_idx, halo_info in enumerate(graph.x):
            # Ensure halo is within redshift threshold
            halo_redshift = halo_info[1].item()
            if 100 * abs(snapshot.current_redshift - halo_redshift) / halo_redshift > REDSHIFT_THRESHOLD:
                continue

            # Load Stellar mass from pop2 mass
            halo_pos = snapshot.arr(halo_info[2:5].tolist(), POS_UNITS)
            halo_rVir = snapshot.arr(halo_info[-1].item(), RVIR_UNITS)
            sph = snapshot.sphere(halo_pos, halo_rVir)
            stellar_mass = sph.quantities.total_quantity(('p2','particle_mass')) 
        
            # Acquire stellar mass in solar masses
            stellar_mass = stellar_mass.value.item() * 5.0000000025E-34
            ys[graph_idx][halo_idx] = stellar_mass
    print(f"Finished snapshot {snapshot}...")

yt.enable_plugins()
enzo_data = yt.load("~jw254/data/SG256-v3/DD????/output_????") # Set to proper enzo dataset
graphs = torch.load('SG256_pruned.pt') # Load graphs from prune_and_gen step

# Capture baryonics from Enzo snapshots
# Adjust to whatever values are in prune_and_gen_graphs logs
RVIR_UNITS = 'kpc'
POS_UNITS = 'unitary'

# The acceptance threshold (percentage) for a halo to query data from a snapshot
REDSHIFT_THRESHOLD = .25
manager = mp.Manager()

# create y's for all the graphs
ys = manager.list([torch.ones(len(graph.x)) * -1 for graph in graphs])

# multiprocessing pool
print(f'creating pool with {mp.cpu_count()} cpus')
pool = mp.Pool(mp.cpu_count())
snapshot_baryonics_partial = partial(snapshot_baryonics, graphs=graphs, ys=ys)

# Load enzo snapshot and its baryonics
# Run snapshot_baryonics in parallel
pool.map(snapshot_baryonics_partial, enzo_data)
pool.close()
pool.join()

# Save y values to graphs
for idx, y in enumerate(ys):
    graphs[idx].y = y

torch.save(graphs, 'SG256_Full_Graphs.pt')
print(f"f{len(graphs)} Graphs saved with y values!")
print("Y is form: [stellar_mass (MSun)]")

# Check to see if any halos didn't have a snapshot they matched to
for graph in graphs:
    if -1 in graph.y:
        print("At least one halo was not able to find a suitable snapshot")
        break
