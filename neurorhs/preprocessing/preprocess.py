"""Helpers for turning graph data and metadata into the saved JAX-ready context."""

import neurorhs.preprocessing.graph_to_arrays as ga
import networkx as nx
import numpy as np
import pandas as pd
import os


def process_params(path_to_full, type_groups, directedness, path_to_save, path_to_metadata):
    """Load a graph, enrich it with metadata, and save the resulting JAX context."""
    graph = nx.read_gml(path_to_full)

    processed_graph = ga.process_graph_to_core_arrays(
        graph, type_groups, directedness)

    metadata = pd.read_csv(path_to_metadata)
    global_mapping = processed_graph['mapping']

    # 10.0 is the default radius used by the package.
    metadata = metadata.fillna(10.0)
    metadata['new_index'] = metadata.apply(
        lambda row: global_mapping['H'].get(str(row['node_id'])), axis=1
    )
    metadata = metadata.dropna(subset=['new_index'])
    metadata = metadata.set_index('new_index').sort_index()

    all_somas = metadata[metadata['type'] == 'root']['node_id'].to_numpy()
    soma_pairs = [(int(soma_id), int(global_mapping['H'][str(soma_id)]))
                  for soma_id in all_somas]
    soma_pairs = np.array(soma_pairs)

    ga.save_jax_arrays(
        processed_graph,
        path_to_save,
        {
            'stom': soma_pairs,
            'x': metadata['x'].to_numpy(),
            'y': metadata['y'].to_numpy(),
            'z': metadata['z'].to_numpy(),
            'r': metadata['radius'].to_numpy(),
        },
    )
