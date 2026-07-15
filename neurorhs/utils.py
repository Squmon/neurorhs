"""Utility helpers for working with nested dictionaries and JAX-compatible values."""

import jax
import jax.numpy as jnp
from jax.lax import scan, fori_loop
from jax.scipy.sparse.linalg import cg
import iree

def iree_compile(foo, input_sample, **compiler_options):
    lowered = jax.jit(foo).lower(*input_sample)
    hlo_code = lowered.as_text('stablehlo')
    return iree.compiler.compile_str(
        hlo_code, 
        **compiler_options
    )

def copy_dict_struct(d, value=None):
    """Create a nested structure matching ``d`` and populate each leaf with ``value``."""
    return {k: copy_dict_struct(v) if isinstance(v, dict) else value for k, v in d.items()}


def flatten_dict(x, separator='/'):
    """Convert a nested dictionary into a flat dictionary using ``separator``."""
    result = {}

    def recurse(current_dict, parent_key=''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dict):
                recurse(value, new_key)
            else:
                result[new_key] = value

    recurse(x)
    return result


def unflatten_dict(x, separator='/'):
    """Restore a flat dictionary into a nested dictionary structure."""
    result = {}
    for key, value in x.items():
        parts = key.split(separator)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def get_dynamic_zeros(x, is_dynamic):
    """Create a nested structure of zeros for the dynamic leaves of ``x``."""
    source_items = flatten_dict(x)
    dynamic_marks = flatten_dict(is_dynamic)
    dynamic_values = {}

    for (key, value), (expected_key, is_dynamic_value) in zip(source_items.items(), dynamic_marks.items()):
        assert key == expected_key
        if is_dynamic_value:
            dynamic_values[key] = jnp.zeros_like(value)

    return unflatten_dict(dynamic_values)


def get_dynamic_static_parts(x, is_dynamic):
    """Split a nested structure into dynamic and static parts based on the flags."""
    source_items = flatten_dict(x)
    dynamic_marks = flatten_dict(is_dynamic)
    dynamic_part = {}
    static_part = {}

    for key, value in source_items.items():
        if dynamic_marks[key]:
            dynamic_part[key] = value
        else:
            static_part[key] = value

    return unflatten_dict(dynamic_part), unflatten_dict(static_part)


def combine_parts(dynamic_part, static_part, is_dynamic):
    """Recombine a dynamic part and a static part using the dynamic flags."""
    marks = flatten_dict(is_dynamic)
    dynamic_items = flatten_dict(dynamic_part)
    static_items = flatten_dict(static_part)
    output = {}

    for key, is_dynamic_value in marks.items():
        if is_dynamic_value:
            output[key] = dynamic_items[key]
        else:
            output[key] = static_items[key]

    return unflatten_dict(output)
