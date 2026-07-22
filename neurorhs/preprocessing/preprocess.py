"""Helpers for turning graph data and metadata into the saved JAX-ready context."""

import neurorhs.preprocessing.graph_to_arrays as ga
import networkx as nx
import numpy as np
import pandas as pd
import os


def process_params(path_to_full, type_groups, directedness, path_to_save, path_to_metadata, id_column='node_id',
                   metadata_columns={
                       'x': 'x',
                       'y': 'y',
                       'z': 'z',
                       'r': 'radius'}, soma_criteria=lambda metadata: metadata['type'] == 'root', cable_key='H', features_config = None):
    """Load a graph, enrich it with metadata, and save the resulting JAX context."""

    # processing graph
    graph = nx.read_gml(path_to_full)
    processed_graph = ga.process_graph_to_core_arrays(
        graph, type_groups, directedness, features_config)
    global_mapping = processed_graph['mapping']

    # rpocessing metadata
    metadata = pd.read_csv(path_to_metadata)
    metadata['new_index'] = metadata.apply(
        lambda row: global_mapping[cable_key].get(str(row[id_column])), axis=1
    )

    metadata = metadata.dropna(subset=['new_index'])
    metadata = metadata.set_index('new_index').sort_index()

    all_somas = metadata[soma_criteria(metadata)][id_column].to_numpy()
    soma_pairs = [(int(soma_id), int(global_mapping[cable_key][str(soma_id)]))
                  for soma_id in all_somas]
    soma_pairs = np.array(soma_pairs)

    ga.save_context(
        processed_graph,
        path_to_save,
        {'stom': soma_pairs} | {
            k: metadata[v].to_numpy() for k, v in metadata_columns.items()
        },
    )
