import torch
import torch_geometric
import yt
import collections
import bisect
import multiprocessing as mp
from functools import partial

# The acceptance threshold (percentage) for a halo to query data from a snapshot
REDSHIFT_THRESHOLD = .25

# Function to load graph baryonics for a certain snapshot
# into a yt parallelism result
def snapshot_baryonics(graphs, sto, snapshot):
    # create y's for all the graphs
    ys = [torch.ones(len(graph.x)) * -1 for graph in graphs]

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
    sto.result = ys
    sto.result_id = snapshot.current_redshift
    print(f"Finished snapshot {snapshot.current_redshift}...")

yt.enable_plugins()
yt.enable_parallelism()
enzo_data = yt.load("~jw254/data/SG256-v3/DD????/output_????") # Set to proper enzo dataset
graphs = torch.load('SG256_pruned.pt') # Load graphs from prune_and_gen step

# Capture baryonics from Enzo snapshots
# Adjust to whatever values are in prune_and_gen_graphs logs
RVIR_UNITS = 'kpc'
POS_UNITS = 'unitary'

# multiprocessing cpus
cpus = mp.cpu_count()
print(f'yt parallelism with {cpus} cpus')

# Run snapshot_baryonics in parallel
parallelism_storage = {}

for sto, snapshot in yt.parallel_objects(enzo_data[0:10], cpus, storage=parallelism_storage):
    # Get the outputs for this snapshot
    snapshot_baryonics(graphs, sto, snapshot)

# Compile finalized y's (take outputs for a graph where the value isnt -1
combined_ys = [torch.ones(len(graph.x)) * -1 for graph in graphs]

for run in parallelism_storage.values():
    for y_idx, y in enumerate(run):
        for v_idx, v in enumerate(y):
            if v != -1:
                combined_ys[y_idx][v_idx] = v

# Save y values to graphs
for idx, y in enumerate(combined_ys):
    graphs[idx].y = y

torch.save(graphs, 'SG256_Full_Graphs.pt')
print(f"{len(graphs)} Graphs saved with y values!")
print("Y is form: [stellar_mass (MSun)]")

# Check to see if any halos didn't have a snapshot they matched to
for graph in graphs:
    if -1 in graph.y:
        print("At least one halo was not able to find a suitable snapshot")
        break
