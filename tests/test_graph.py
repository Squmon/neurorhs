from neurorhs.preprocessing.graph_to_arrays import load_jax_context, process_graph_to_core_arrays
from neurorhs.preprocessing.graph_to_arrays import save_jax_arrays
import jax
import jax.numpy as jnp
import networkx as nx

def _equal_tree(x, y):
    # compare leaves using jnp.array_equal for arrays/scalars
    return jax.tree_util.tree_map(
        lambda u, v: bool(jnp.array_equal(jnp.asarray(u), jnp.asarray(v))), x, y
    )

def test_load_save():
    G = nx.read_gml("data/test_graph_3n.gml")

    type_groups = {'H': ['root', 'soma', 'branch', 'slab', 'end'], 'S': ['connector']}
    directedness = {'H': {'H': False, 'S': True}, 'S': {'H': True, 'S': True}}
    ctx = process_graph_to_core_arrays(G, type_groups, directedness)
    save_jax_arrays(ctx, "./test_res.npz")
    ctx_loaded = load_jax_context("./test_res.npz")['root']
    assert _equal_tree(ctx, ctx_loaded)