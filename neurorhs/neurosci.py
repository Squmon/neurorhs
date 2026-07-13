import jax
from neurorhs.utils import *
import jax.numpy as jnp
from jax import jit


def save_exp(x, max_value: float = 100.0):
    x = jnp.clip(x, -jnp.inf, max_value)
    return jnp.exp(x)


def m_gate(v):
    alpha = 0.1 * _vtrap(-(v + 40), 10)
    beta = 4.0 * save_exp(-(v + 65) / 18)
    return alpha, beta

def h_gate(v):
    alpha = 0.07 * save_exp(-(v + 65) / 20)
    beta = 1.0 / (save_exp(-(v + 35) / 10) + 1)
    return alpha, beta

def n_gate(v):
    alpha = 0.01 * _vtrap(-(v + 55), 10)
    beta = 0.125 * save_exp(-(v + 65) / 80)
    return alpha, beta


def _vtrap(x, y, epsilon=1e-12):
    return x / (save_exp(x / y) - 1.0 + epsilon)
def calculate_cable_scaling(edges, ro, x, y, z, r):
    S = jnp.pi * r**2
    dx = x.at[edges[:, 1]].get() - x.at[edges[:, 0]].get()
    dy = y.at[edges[:, 1]].get() - y.at[edges[:, 0]].get()
    dz = z.at[edges[:, 1]].get() - z.at[edges[:, 0]].get()    
    L = (dx**2 + dy**2 + dz**2)**0.5
    R = (ro*L/S.at[edges[:, 1]].get())
    return R

def get_cabble_pipeline(
    edges
):  # edges должны быть не ореинтированны и не повторятся
    q = jnp.array(edges, jnp.int32)
    static_sources = q[:, 0]
    static_targets = q[:, 1]
    def graph_evolution_fn_with_scaling(X: jnp.ndarray, dx_dt) -> jnp.ndarray:
        x = X['morphology']['position']['x']
        y = X['morphology']['position']['y']
        z = X['morphology']['position']['z']
        r = X['morphology']['r']
        ro = X['morphology']['ro']
        scaling = calculate_cable_scaling(edges, ro, x, y, z, r)
        potential_diff = (
            X['V'].at[static_targets].get() - X['V'].at[static_sources].get()
        )
        dx_dt['V'] = dx_dt['V'].at[static_sources].add(potential_diff*scaling.at[static_sources].get())
        dx_dt['V'] = dx_dt['V'].at[static_targets].add(-potential_diff*scaling.at[static_targets].get())
        return dx_dt
    graph_evolution_fn = graph_evolution_fn_with_scaling

    return jax.jit(graph_evolution_fn)



def get_Na_channel_pipeline():
    """
    Sodium (Na) channel pipeline.
    Reads/writes states matching structure.txt:
      state['V'], state['morphology']['C']
      state['morphology']['Na']['m'], state['morphology']['Na']['h']
      state['morphology']['Na']['gNa'], state['morphology']['Na']['eNa']
    """
    @jax.jit
    def Na_pipeline(state, ds_dt):
        V = state['V']
        m = state['morphology']['Na']['m']
        h = state['morphology']['Na']['h']
        gNa = state['morphology']['Na']['gNa']
        eNa = state['morphology']['Na']['eNa']
        C = state['morphology']['C']

        alpha_m, beta_m = m_gate(V)
        dm = alpha_m * (1.0 - m) - beta_m * m

        alpha_h, beta_h = h_gate(V)
        dh = alpha_h * (1.0 - h) - beta_h * h

        INa = gNa * h * (m**3) * (V - eNa)

        # Update ds_dt
        ds_dt['morphology']['Na']['m'] = ds_dt['morphology']['Na']['m'] + dm
        ds_dt['morphology']['Na']['h'] = ds_dt['morphology']['Na']['h'] + dh
        ds_dt['V'] = ds_dt['V'] - INa / C
        return ds_dt

    return Na_pipeline


def get_K_channel_pipeline():
    """
    Potassium (K) channel pipeline.
    Reads/writes states matching structure.txt:
      state['V'], state['morphology']['C']
      state['morphology']['K']['n']
      state['morphology']['K']['gK'], state['morphology']['K']['eK']
    """
    @jax.jit
    def K_pipeline(state, ds_dt):
        V = state['V']
        n = state['morphology']['K']['n']
        gK = state['morphology']['K']['gK']
        eK = state['morphology']['K']['eK']
        C = state['morphology']['C']

        alpha_n, beta_n = n_gate(V)
        dn = alpha_n * (1.0 - n) - beta_n * n

        IK = gK * (n**4) * (V - eK)

        # Update ds_dt
        ds_dt['morphology']['K']['n'] = ds_dt['morphology']['K']['n'] + dn
        ds_dt['V'] = ds_dt['V'] - IK / C
        return ds_dt

    return K_pipeline


def get_leak_channel_pipeline():
    """
    Leak channel pipeline.
    Reads/writes states matching structure.txt:
      state['V'], state['morphology']['C']
      state['morphology']['leak']['gLeak'], state['morphology']['leak']['eLeak']
    """
    @jax.jit
    def leak_pipeline(state, ds_dt):
        V = state['V']
        gL = state['morphology']['leak']['gLeak']
        eL = state['morphology']['leak']['eLeak']
        C = state['morphology']['C']

        Ileak = gL * (V - eL)

        # Update ds_dt
        ds_dt['V'] = ds_dt['V'] - Ileak / C
        return ds_dt

    return leak_pipeline


def get_stub_synapse_pipeline(pre_syn_edges, post_syn_edges):
    """
    Stub synapse pipeline using a sigmoid activation function.
    
    Args:
        pre_syn_edges: Array of shape (2, N) where:
            pre_syn_edges[0, :] is presynaptic cable segment indices
            pre_syn_edges[1, :] is synapse indices
        post_syn_edges: Array of shape (2, N) where:
            post_syn_edges[0, :] is synapse indices
            post_syn_edges[1, :] is postsynaptic cable segment indices
    """
    pre_cable_idx = jnp.array(pre_syn_edges[0, :], dtype=jnp.int32)
    syn_pre_idx = jnp.array(pre_syn_edges[1, :], dtype=jnp.int32)

    syn_post_idx = jnp.array(post_syn_edges[0, :], dtype=jnp.int32)
    post_cable_idx = jnp.array(post_syn_edges[1, :], dtype=jnp.int32)

    @jax.jit
    def synapse_pipeline(state, ds_dt):
        V_cable = state['V']
        V_syn = state['connectors']['stub']['V']
        weight = state['connectors']['stub']['weight']
        C = state['morphology']['C']

        # 1. Update synapse V to sigmoid of presynaptic V
        V_pre = V_cable[pre_cable_idx]
        target_V_syn = jax.nn.sigmoid(V_pre)
        V_syn_active = V_syn[syn_pre_idx]
        ds_dt['connectors']['stub']['V'] = ds_dt['connectors']['stub']['V'].at[syn_pre_idx].add(target_V_syn - V_syn_active)

        # 2. Synchronize postsynaptic V with synapse V, scaled by weight
        V_syn_mapped = V_syn[syn_post_idx]
        V_post = V_cable[post_cable_idx]
        weight_mapped = weight[syn_post_idx]
        C_post = C[post_cable_idx]

        # Synchronization term: (V_syn - V_post) * weight
        sync_term = (V_syn_mapped - V_post) * weight_mapped / C_post
        
        # Accumulate changes to V_cable
        ds_dt['V'] = ds_dt['V'].at[post_cable_idx].add(sync_term)
        return ds_dt

    return synapse_pipeline