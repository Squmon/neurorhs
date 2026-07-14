from neurorhs.preprocessing.preprocess import process_params
from neurorhs.preprocessing.graph_to_arrays import (
    load_jax_context,
    process_graph_to_core_arrays,
    save_jax_arrays,
)
import pytest
import numpy as np
import networkx as nx
import jax.numpy as jnp
import jax
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _equal_tree(x, y):
    return jax.tree_util.tree_map(
        lambda u, v: bool(jnp.array_equal(jnp.asarray(u), jnp.asarray(v))),
        x,
        y,
    )


def test_load_save_round_trip(graph, type_groups, directedness, generated_dir):
    output_path = generated_dir / "test_graph_context.jconn"

    ctx = process_graph_to_core_arrays(graph, type_groups, directedness)
    save_jax_arrays(ctx, str(output_path))

    loaded = load_jax_context(str(output_path))
    assert output_path.exists()
    assert _equal_tree(ctx, loaded["root"])


def test_preprocess_writes_generated_artifacts(
    graph_path,
    metadata_path,
    type_groups,
    directedness,
    generated_dir,
):
    output_path = generated_dir / "test_preprocess_output.jconn"

    process_params(
        str(graph_path),
        type_groups,
        directedness,
        str(output_path),
        str(metadata_path),
    )

    loaded = load_jax_context(str(output_path))
    assert output_path.exists()
    assert "root" in loaded
    assert "additional_data" in loaded

    additional_data = loaded["additional_data"]
    assert "stom" in additional_data
    assert "x" in additional_data
    assert "y" in additional_data
    assert "z" in additional_data
    assert "r" in additional_data

    assert len(additional_data["x"]) > 0
    assert len(additional_data["stom"]) > 0
