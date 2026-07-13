"""Helpers for converting NetworkX graphs into compact JAX-friendly arrays."""

from neurorhs.utils import flatten_dict, unflatten_dict
from copy import copy

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Set
import os
import itertools

# --- Типы данных ---
DirectednessMap = Dict[str, Dict[str, bool]]
GraphResults = Dict[str, Any]


def _get_group_name(graph: Union[nx.DiGraph, nx.MultiDiGraph], node_id: Any, node_type_groups: Dict[str, List[str]]) -> List[str]:
    """Return the group names associated with a node based on its ``type`` attribute."""
    node_type = graph.nodes[node_id].get('type')
    if not node_type:
        return []

    found_groups: List[str] = []
    for group_name, type_list in node_type_groups.items():
        if node_type in type_list:
            found_groups.append(group_name)
    return found_groups


def _create_mappings(graph: Union[nx.DiGraph, nx.MultiDiGraph], node_type_groups: Dict[str, List[str]]) -> Tuple[Dict[Any, int], Dict[str, Dict[int, int]], Dict[str, int]]:
    """Create global and local index maps for the graph nodes."""
    sorted_nodes = sorted(graph.nodes())

    global_mapping: Dict[Any, int] = {}
    global_idx = 0
    for node_id in sorted_nodes:
        global_mapping[node_id] = global_idx
        global_idx += 1

    local_maps: Dict[str, Dict[int, int]] = {
        group: {} for group in node_type_groups.keys()}
    num_nodes: Dict[str, int] = {group: 0 for group in node_type_groups.keys()}

    for node_id in sorted_nodes:
        group_names = _get_group_name(graph, node_id, node_type_groups)

        for group_name in group_names:
            if group_name in local_maps:
                global_id = global_mapping[node_id]

                if global_id not in local_maps[group_name]:
                    local_idx = num_nodes[group_name]
                    local_maps[group_name][global_id] = local_idx
                    num_nodes[group_name] += 1

    num_nodes = {k: v for k, v in num_nodes.items() if v > 0}
    local_maps = {k: v for k, v in local_maps.items() if v}

    return global_mapping, local_maps, num_nodes


def _create_edge_arrays(graph: Union[nx.DiGraph, nx.MultiDiGraph],
                        global_mapping: Dict[Any, int],
                        local_maps: Dict[str, Dict[int, int]],
                        num_nodes: Dict[str, int],
                        edge_directedness: DirectednessMap,
                        node_type_groups: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Create edge arrays using the local indices for each node group."""
    edge_arrays: Dict[str, np.ndarray] = {}
    groups = list(num_nodes.keys())

    for src_group, dst_group in itertools.product(groups, groups):
        edge_list: List[Tuple[int, int]] = []
        is_directed = edge_directedness.get(src_group, {}).get(dst_group, True)

        src_local_map = local_maps.get(src_group, {})
        dst_local_map = local_maps.get(dst_group, {})

        if not src_local_map or not dst_local_map:
            continue

        for u_old, v_old in graph.edges():
            u_groups = _get_group_name(graph, u_old, node_type_groups)
            v_groups = _get_group_name(graph, v_old, node_type_groups)

            found_match = False
            for u_group in u_groups:
                for v_group in v_groups:
                    if u_group == src_group and v_group == dst_group:
                        u_global_id = global_mapping[u_old]
                        v_global_id = global_mapping[v_old]

                        u_local_idx = src_local_map.get(u_global_id)
                        v_local_idx = dst_local_map.get(v_global_id)

                        if u_local_idx is not None and v_local_idx is not None:
                            edge_list.append((u_local_idx, v_local_idx))
                            found_match = True
                            break
                if found_match:
                    break

        if not is_directed and src_group == dst_group:
            undirected_edges = set(edge_list)
            for u, v in edge_list:
                undirected_edges.add((v, u))
            edge_list = list(undirected_edges)

        if edge_list:
            unique_edges = sorted(list(set(edge_list)))
            array = np.array(unique_edges, dtype=np.int32).T
            key = f'edges_{src_group}_to_{dst_group}'
            edge_arrays[key] = array

    return edge_arrays


def _calculate_user_mapping(
    num_nodes: Dict[str, int],
    global_map_old_to_new: Dict[Any, int],
    local_maps_global_to_local: Dict[str, Dict[int, int]]
) -> Dict[str, Dict[Any, int]]:
    """Build the user-facing mapping from original node IDs to local group indices."""
    final_mapping: Dict[str, Dict[Any, int]] = {
        k: {} for k in num_nodes.keys()}

    for original_id, global_id in global_map_old_to_new.items():
        for group_name in num_nodes.keys():
            local_map = local_maps_global_to_local.get(group_name, {})
            local_index = local_map.get(global_id)

            if local_index is not None:
                final_mapping[group_name][original_id] = local_index

    return final_mapping


def process_graph_to_core_arrays(
    graph: Union[nx.DiGraph, nx.MultiDiGraph],
    node_type_groups: Dict[str, List[str]],
    edge_directedness: DirectednessMap
) -> GraphResults:
    """Convert a graph into the core arrays and mappings used by the JAX pipeline."""
    print("Начало обработки графа (маппинги и ребра)...")

    global_mapping, local_maps, num_nodes = _create_mappings(
        graph, node_type_groups)
    edge_arrays = _create_edge_arrays(
        graph, global_mapping, local_maps, num_nodes, edge_directedness, node_type_groups)
    final_mapping = _calculate_user_mapping(
        num_nodes, global_mapping, local_maps)

    print(
        f"Обработка завершена. Найдено {len(num_nodes)} групп и {len(edge_arrays)} массивов ребер.")

    context: GraphResults = {
        'num_nodes': num_nodes,
        'mapping': final_mapping,
        **edge_arrays,
        '__RAW_GLOBAL_MAPPING__': global_mapping,
        '__RAW_LOCAL_MAPS__': local_maps,
    }

    return context


def extract_node_features(
    graph: Union[nx.DiGraph, nx.MultiDiGraph],
    context: GraphResults,
    feature_config: Dict[str, List[str]]
) -> Dict[str, np.ndarray]:
    """Extract requested node attributes from the graph into NumPy feature arrays."""
    feature_arrays: Dict[str, np.ndarray] = {}

    if not feature_config:
        return feature_arrays

    global_mapping = context['__RAW_GLOBAL_MAPPING__']
    local_maps = context['__RAW_LOCAL_MAPS__']
    num_nodes = context['num_nodes']

    for group_name, properties_list in feature_config.items():
        if group_name not in num_nodes or not properties_list:
            continue

        num = num_nodes[group_name]
        local_map = local_maps.get(group_name, {})

        num_features = len(properties_list)
        feature_array = np.zeros((num, num_features), dtype=np.float64)

        for node_id, global_id in global_mapping.items():
            local_idx = local_map.get(global_id)
            if local_idx is None:
                continue

            node_data = graph.nodes[node_id]

            for prop_idx, prop_key in enumerate(properties_list):
                value = node_data.get(prop_key, 0.0)
                try:
                    feature_array[local_idx, prop_idx] = float(value)
                except (ValueError, TypeError):
                    feature_array[local_idx, prop_idx] = 0.0

        key = f'features_{group_name}'
        feature_arrays[key] = feature_array

    return feature_arrays


TYPE_MAPPING = {
    'str': str, 'int': int, 'float': float, 'bool': lambda v: v == 'True'
}


def save_np_dict(x, path, separator='/'):
    """Save a nested dictionary to an NPZ file while preserving the key types."""
    flat_dict = flatten_dict(x, separator=separator)

    types_meta = {}

    def _build_meta(current_dict, parent_key=''):
        for key, value in current_dict.items():
            key_str = str(key)
            path_key = f"{parent_key}{separator}{key_str}" if parent_key else key_str
            if isinstance(value, dict):
                _build_meta(value, path_key)
            types_meta[path_key] = type(key).__name__

    _build_meta(x)

    meta_arr = np.array(list(types_meta.items()), dtype=object)
    np.savez(path, __meta__=meta_arr, **flat_dict)


def load_np_dict(path, separator='/'):
    """Load a nested dictionary from an NPZ file and restore the original key types."""
    if not path.endswith('.npz') and not os.path.exists(path):
        path += '.npz'

    with np.load(path, allow_pickle=True) as data:
        data_keys = [k for k in data.files if k != '__meta__']
        flat_dict = {k: data[k] for k in data_keys}

        for k, v in flat_dict.items():
            if v.ndim == 0:
                flat_dict[k] = v.item()

        if '__meta__' not in data.files:
            return unflatten_dict(flat_dict, separator=separator)

        meta_dict = dict(data['__meta__'])

    nested_dict = unflatten_dict(flat_dict, separator=separator)

    def _cast_keys_inplace(d, parent_key=''):
        cased_dict = {}
        for key_str, value in d.items():
            current_path = f"{parent_key}{separator}{key_str}" if parent_key else key_str
            type_name = meta_dict.get(current_path, 'str')
            actual_key = TYPE_MAPPING.get(type_name, str)(key_str)

            if isinstance(value, dict):
                cased_dict[actual_key] = _cast_keys_inplace(
                    value, current_path)
            else:
                cased_dict[actual_key] = value
        return cased_dict

    return _cast_keys_inplace(nested_dict)


def save_jax_arrays(
    context: GraphResults,
    save_path: str,
    additional_data: Dict = None
):
    """Persist the full graph context and optional arrays to an NPZ file."""
    print(f"Начало сохранения данных в {save_path}...")

    global_mapping = context.get('__RAW_GLOBAL_MAPPING__')
    local_maps = context.get('__RAW_LOCAL_MAPS__')

    if global_mapping is None or local_maps is None:
        raise ValueError(
            "В контексте отсутствуют обязательные raw-маппинги (__RAW_GLOBAL_MAPPING__ или __RAW_LOCAL_MAPS__). Убедитесь, что контекст создан с помощью process_graph_to_core_arrays.")

    data_to_save: Dict[str, Any] = {}
    data_to_save['root'] = copy(context)
    data_to_save['additional_data'] = additional_data
    save_np_dict(data_to_save, save_path)


def load_jax_context(load_path: str) -> GraphResults:
    """Load a previously saved graph context from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Файл данных не найден: {load_path}")

    return load_np_dict(load_path)


if __name__ == '__main__':
    SAVE_FILE = 'test_graph_arrays.npz'
    G = nx.DiGraph()

    type_groups = {'H': ['H_root', 'HS_hybrid'], 'S': ['S_conn', 'HS_hybrid']}

    node_configs = {
        1001: ('H_root', {'bias': 0.1, 'weight': 5.0}),
        1005: ('H_root', {'bias': 0.5, 'weight': 10.0}),
        2010: ('S_conn', {'bias': 0.9, 'initial_value': 2.0}),
        9000: ('HS_hybrid', {'bias': 0.2, 'weight': 8.0, 'initial_value': 1.5})
    }
    for oid, (ntype, attrs) in node_configs.items():
        G.add_node(oid, type=ntype, **attrs)

    G.add_edges_from([
        (1001, 1005),
        (9000, 1001),
        (2010, 9000),
        (1005, 2010),
    ])

    directedness = {'H': {'H': False, 'S': True}, 'S': {'H': True, 'S': True}}

    # Конфигурация признаков для извлечения
    feature_config_H = {'H': ['bias', 'weight']}
    feature_config_S = {'S': ['initial_value', 'bias']}

    print("================== ТЕСТ 1: ОБРАБОТКА (возвращает единый контекст) ==================")

    # 1. Получаем единый контекст
    context = process_graph_to_core_arrays(
        graph=G,
        node_type_groups=type_groups,
        edge_directedness=directedness
    )

    # 2. Извлечение признаков (теперь просто обновляем контекст)
    features_H = extract_node_features(G, context, feature_config_H)
    features_S = extract_node_features(G, context, feature_config_S)

    # 3. Произвольные данные (абстрактные маски)
    # Предположим, что H имеет 3 узла, а S - 2 узла (по данным num_nodes)
    custom_mask = {'mask_H': np.array([1, 0, 1], dtype=np.int32),
                   'mask_S': np.array([0, 1], dtype=np.int32)}

    # 4. Включаем все дополнительные данные в контекст
    context.update(features_H)
    context.update(features_S)
    context.update(custom_mask)

    # 5. Сохранение: просто передаем единый контекст
    print("\n================== ТЕСТ 2: СОХРАНЕНИЕ (принимает единый контекст) ==================")
    save_jax_arrays(
        context=context,
        save_path=SAVE_FILE,
        # additional_data здесь не нужен, так как все добавлено в context
    )

    print("\n================== ТЕСТ 3: ЗАГРУЗКА И ПРОВЕРКА КОНТЕКСТА ==================")

    # 6. Загрузка
    loaded_context = load_jax_context(SAVE_FILE)

    # Проверка структуры загруженного контекста
    print("\nСтруктура загруженного контекста:")
    print(f"  > num_nodes: {loaded_context['num_nodes']}")
    print(
        f"  > Ребра H_to_S: {loaded_context.get('edges_H_to_S').shape if loaded_context.get('edges_H_to_S') is not None else 'N/A'}")
    print(f"  > Произвольная маска H: {loaded_context.get('mask_H')}")
    print(
        f"  > Скрытые маппинги присутствуют: {'__RAW_GLOBAL_MAPPING__' in loaded_context}")

    # Проверка признаков и маппинга
    features_H = loaded_context.get('features_H')
    if features_H is not None:
        local_idx_9000_H = loaded_context['mapping']['H'].get(9000)
        if local_idx_9000_H is not None:
            print(
                f"  > Признаки 9000 в H (локальный индекс {local_idx_9000_H}): {features_H[local_idx_9000_H]}")

    # --- Очистка ---
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)
