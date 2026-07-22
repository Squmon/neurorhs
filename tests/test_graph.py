from neurorhs.preprocessing.preprocess import process_params, collapse_nodes
from neurorhs.preprocessing.graph_to_arrays import (
    process_graph_to_core_arrays,
)
from neurorhs.io import load_context, save_context
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
    save_context(ctx, str(output_path))

    loaded = load_context(str(output_path))
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

    loaded = load_context(str(output_path))
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


def test_collapse_nodes():
    # Create a directed graph with nodes a, b, c, d
    g = nx.DiGraph()
    g.add_node("a", weight=1.0)
    g.add_node("b", weight=2.0)
    g.add_node("c", weight=3.0)
    g.add_node("d", weight=4.0)

    # Add edges
    g.add_edge("a", "d", label="ad")
    g.add_edge("c", "d", label="cd")
    g.add_edge("d", "a", label="da")
    g.add_edge("b", "d", label="bd")

    # Map: a->b, c->b
    mapping = {"a": "b", "c": "b"}
    collapsed = collapse_nodes(g, mapping)

    # a and c should be deleted, b and d remain
    assert set(collapsed.nodes()) == {"b", "d"}
    # b should retain its attributes, d also
    assert collapsed.nodes["b"]["weight"] == 2.0
    assert collapsed.nodes["d"]["weight"] == 4.0

    # Edges transfer check:
    # a->d becomes b->d, c->d becomes b->d. bd remains b->d.
    # d->a becomes d->b.
    # Therefore edges should be:
    # b->d (with merged label if overwritten, or at least exists)
    # d->b
    assert collapsed.has_edge("b", "d")
    assert collapsed.has_edge("d", "b")
    assert not collapsed.has_edge("a", "d")
    assert not collapsed.has_edge("c", "d")
    assert not collapsed.has_edge("d", "a")

