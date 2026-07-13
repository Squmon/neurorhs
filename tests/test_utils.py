import jax.numpy as jnp
import pytest

def test_zeros():
    from neurorhs.utils import get_dynamic_zeros
    import jax
    a = {
        "a": {
            "a": jnp.array([1, 2, 3]),
            "b": jnp.array([3, 2]),
        },
        "c": {
            "a": 1,
            "b": 0.0
        },
        "d": {
            "a": jnp.array([0.0, 1.0, 2.0]),
            "q": 1.0,
        }
    }
    m = {
        "a": {
            "a": True,
            "b": False,
        },
        "c": {
            "a": False,
            "b": False
        },
        "d": {
            "a": True,
            "q": True
        }
    }
    target = {
        "a": {
            "a": jnp.zeros_like(jnp.array([1, 2, 3])),
        },
        "d": {
            "a": jnp.zeros_like(jnp.array([0.0, 1.0, 2.0])),
            "q": jnp.zeros_like(1.0),
        }
    }
    y = get_dynamic_zeros(a, m)
    import jax

    def _equal_tree(x, y):
        # compare leaves using jnp.array_equal for arrays/scalars
        return jax.tree_util.tree_map(
            lambda u, v: bool(jnp.array_equal(jnp.asarray(u), jnp.asarray(v))), x, y
        )

    cmp = _equal_tree(y, target)
    leaves = jax.tree_util.tree_leaves(cmp)
    assert all(leaves)
