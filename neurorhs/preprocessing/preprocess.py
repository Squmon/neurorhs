"""Helpers for turning graph data and metadata into the saved JAX-ready context."""

import neurorhs.preprocessing.graph_to_arrays as ga
from neurorhs.io import save_context
import networkx as nx
import numpy as np
import pandas as pd
import os


def collapse_nodes(graph: nx.Graph, mapping: dict) -> nx.Graph:
    """Collapse nodes in the graph based on the mapping dictionary.
    
    If mapping maps key->val, we delete key and transfer all its edges to val.
    Chained mappings (e.g. a->b, b->c) are resolved recursively to their final targets.
    """
    resolved_mapping = {}
    for node in graph.nodes():
        visited = set()
        curr = node
        while curr in mapping and curr not in visited:
            visited.add(curr)
            curr = mapping[curr]
        if curr != node:
            resolved_mapping[node] = curr

    new_graph = graph.__class__()
    new_graph.graph.update(graph.graph)

    for node, data in graph.nodes(data=True):
        if node not in resolved_mapping:
            new_graph.add_node(node, **data)

    is_multigraph = graph.is_multigraph()

    if is_multigraph:
        for u, v, key, data in graph.edges(keys=True, data=True):
            u_new = resolved_mapping.get(u, u)
            v_new = resolved_mapping.get(v, v)
            new_graph.add_edge(u_new, v_new, key=key, **data)
    else:
        for u, v, data in graph.edges(data=True):
            u_new = resolved_mapping.get(u, u)
            v_new = resolved_mapping.get(v, v)
            if new_graph.has_edge(u_new, v_new):
                existing_data = new_graph[u_new][v_new]
                merged_data = {**existing_data, **data}
                new_graph.add_edge(u_new, v_new, **merged_data)
            else:
                new_graph.add_edge(u_new, v_new, **data)

    return new_graph


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

    save_context(
        processed_graph,
        path_to_save,
        {'stom': soma_pairs} | {
            k: metadata[v].to_numpy() for k, v in metadata_columns.items()
        },
    )
