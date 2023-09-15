import torch
import torch_geometric
import yt
import collections
import bisect

yt.enable_plugins()
enzo_data = yt.load("~jw254/data/SG256-v3/DD????/output_????") # Set to proper enzo dataset
graphs = torch.load('SG256_pruned.pt') # Load graphs from prune_and_gen step

# Initialize y values of all halos to -1 so we can validate all halos were queried
for graph in graphs:
    graph.y = torch.ones(len(graph.x)) * -1

# Capture baryonics from Enzo snapshots
# Adjust to whatever values are in prune_and_gen_graphs logs
RVIR_UNITS = 'kpc'
POS_UNITS = 'unitary'

# The acceptance threshold (percentage) for a halo to query data from a snapshot
REDSHIFT_THRESHOLD = .25

# Load enzo snapshot (currently looking at 300-309)
for snapshot in enzo_data:
    snapshot.add_particle_filter('p2')
    snapshot.add_particle_filter('p3')
    for graph in graphs:
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
            graph.y[halo_idx] = stellar_mass
    # Save output after every cycle
    print(f"Saving graphs for snapshot {snapshot}...")
    torch.save(graphs, 'SG256_Full_Graphs.pt') # Set output dir

print(f"f{len(graphs)} Graphs saved with y values!")
print("Y is form: [stellar_mass (MSun)]")

# Check to see if any halos didn't have a snapshot they matched to
for graph in graphs:
    if -1 in graph.y:
        print("At least one halo was not able to find a suitable snapshot")
        break
