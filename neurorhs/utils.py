import jax
import jax.numpy as jnp
from jax.lax import scan, fori_loop
from jax.scipy.sparse.linalg import cg

def copy_dict_struct(d, value = None):
    return {k: copy_dict_struct(v) if isinstance(v, dict) else value for k, v in d.items()}

def flatten_dict(x, separator='/'):
    """
    Рекурсивно превращает вложенный словарь в плоский.
    {'a': {'b': 1}} -> {'a/b': 1}
    """
    result = {}
    
    def recurse(current_dict, parent_key=''):
        for key, value in current_dict.items():
            # Формируем новый ключ
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict):
                recurse(value, new_key)
            else:
                result[new_key] = value
                
    recurse(x)
    return result

def unflatten_dict(x, separator='/'):
    """
    Восстанавливает плоский словарь обратно во вложенную структуру.
    {'a/b': 1} -> {'a': {'b': 1}}
    """
    result = {}
    for key, value in x.items():
        parts = key.split(separator)
        current = result
        # Идем по цепочке ключей, создавая вложенные словари
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Записываем значение в последний ключ
        current[parts[-1]] = value
    return result

def get_dynamic_zeros(x, is_dynamic):
    a = flatten_dict(x)
    b = flatten_dict(is_dynamic)
    q = dict()
    for (k, v), (k1, s) in zip(a.items(), b.items()):
        assert k == k1
        if s:
            q[k] = jnp.zeros_like(v)
    return unflatten_dict(q)


def get_dynamic_static_parts(x, is_dynamic):
    a = flatten_dict(x)
    b = flatten_dict(is_dynamic)
    dynamic_part = dict()
    static_part = dict()
    for k, v in a.items():
        if b[k]:
            dynamic_part[k] = v
        else:
            static_part[k] = v
                
    return unflatten_dict(dynamic_part), unflatten_dict(static_part)

def combine_parts(dynamic_part, static_part, is_dynamic):
    marks = flatten_dict(is_dynamic)
    dp = flatten_dict(dynamic_part)
    sp = flatten_dict(static_part)
    output = dict()
    for k, s in marks.items():
        if s:
            output[k] = dp[k]
        else:
            output[k] = sp[k]
    return unflatten_dict(output)

