import os
import pickle
from typing import Any, Dict

import jax
import numpy as np

from neurorhs.preprocessing.graph_to_arrays import GraphResults
from copy import copy


def save_pytree(x, path, compress=True, compression_level=3):
    """
    Сохраняет структуру PyTree (словарь) в один файл на максимальной скорости.

    Args:
        x: Словарь (или любой PyTree), который нужно сохранить.
        path: Путь к файлу для сохранения.
        compress: Если True, использует сжатие (Zstandard, если установлен, иначе zlib).
        compression_level: Уровень сжатия (для zstd рекомендуемый 3, для zlib 1-9).
    """
    # 1. Мгновенно «сплющиваем» дерево с помощью JAX.
    # JAX сам соберет структуру (treedef) и список листьев (leaves).
    leaves, treedef = jax.tree_util.tree_flatten(x)

    # 2. Переводим JAX-массивы в стандартные CPU NumPy-массивы.
    # Это гарантирует, что файл запишется быстро и его можно будет прочесть без привязки к GPU/TPU.
    leaves_np = [np.asarray(leaf) if isinstance(
        leaf, (jax.Array, np.ndarray)) else leaf for leaf in leaves]

    # 3. Сериализуем структуру и массивы через Pickle Protocol 5 (самый быстрый протокол без лишнего копирования)
    serialized = pickle.dumps((treedef, leaves_np), protocol=5)

    # 4. Опциональное сжатие
    if compress:
        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=compression_level)
            data = cctx.compress(serialized)
        except ImportError:
            # Если zstandard не установлен, откатываемся на стандартный zlib
            import zlib
            data = zlib.compress(serialized, level=compression_level)
    else:
        data = serialized

    with open(path, 'wb') as f:
        f.write(data)


def load_pytree(path, to_jax=False):
    """
    Загружает структуру PyTree из файла, полностью восстанавливая оригинальные типы ключей.

    Args:
        path: Путь к файлу.
        to_jax: Если True, все загруженные массивы будут автоматически перенесены 
                в память вашего JAX-девайса (GPU/TPU). Если False, останутся на CPU как NumPy.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден.")

    with open(path, 'rb') as f:
        data = f.read()

    # 1. Автоматическое определение сжатия по сигнатуре файла (magic bytes)
    if data.startswith(b'\x28\xb5\x2f\xfd'):  # Сигнатура Zstandard
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        serialized = dctx.decompress(data)
    elif data.startswith(b'\x78'):  # Сигнатура zlib
        import zlib
        serialized = zlib.decompress(data)
    else:
        # Файл не сжат
        serialized = data

    # 2. Десериализуем структуру JAX и листья
    treedef, leaves_np = pickle.loads(serialized)

    # 3. При необходимости переносим массивы обратно в JAX (на дефолтный ускоритель)
    if to_jax:
        leaves = [jax.device_put(leaf) if isinstance(
            leaf, np.ndarray) else leaf for leaf in leaves_np]
    else:
        leaves = leaves_np

    # 4. Восстанавливаем оригинальное дерево (JAX сам соберет структуру и вернет исходные типы ключей)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def save_context(
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
    save_pytree(data_to_save, save_path)


def load_context(load_path: str) -> GraphResults:
    """Load a previously saved graph context from disk."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Файл данных не найден: {load_path}")

    return load_pytree(load_path)
