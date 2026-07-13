import neurorhs.preprocessing.graph_to_arrays as ga
import networkx as nx
import numpy as np
import pandas as pd
import os

def process_params(path_to_full, type_groups, directedness, path_to_save, path_to_metadata):
    G = nx.read_gml(path_to_full)

#     type_groups = {'H': ['root', 'soma', 'branch',
#                              'slab', 'end'], 'S': ['connector']}
#     directedness = {'H': {'H': False, 'S': True}, 'S': {'H': True, 'S': True}}

    res = ga.process_graph_to_core_arrays(G, type_groups, directedness)

    metadata = pd.read_csv(path_to_metadata)
    global_mapping = res['mapping']
    metadata = metadata.fillna(10.0) # 10.0 as basic radius
    metadata['new_index'] = metadata.apply(lambda row:global_mapping['H'].get(str(row['node_id'])), axis = 1)
    metadata = metadata.dropna(subset=['new_index'])
    metadata = metadata.set_index('new_index').sort_index()


    all_somas = metadata[metadata['type'] == 'root']['node_id'].to_numpy()
    stom = [(int(soma), int(global_mapping['H'][str(soma)])) for soma in all_somas]
    stom = np.array(stom)

    ga.save_jax_arrays(res, path_to_save, {"stom":stom, # сома_global_id, сома_cabble_id
                                            'x':metadata['x'].to_numpy(),
                                            'y':metadata['y'].to_numpy(),
                                            'z':metadata['z'].to_numpy(),
                                            'r':metadata['radius'].to_numpy()})