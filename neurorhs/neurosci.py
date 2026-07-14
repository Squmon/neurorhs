"""JAX-based neuron model pipelines for channel and synapse updates.

The module preserves the existing public API while exposing the internal
state updates through small helper functions and descriptive docstrings.
"""

import jax
from neurorhs.utils import *
import jax.numpy as jnp
from jax import jit


def save_exp(x, max_value: float = 100.0):
    """Clamp values before applying the exponential to avoid overflow."""
    clipped_x = jnp.clip(x, -jnp.inf, max_value)
    return jnp.exp(clipped_x)


def m_gate(v):
    """Return the activation and decay rates for the sodium activation gate."""
    alpha = 0.1 * _vtrap(-(v + 40), 10)
    beta = 4.0 * save_exp(-(v + 65) / 18)
    return alpha, beta


def h_gate(v):
    """Return the activation and decay rates for the sodium inactivation gate."""
    alpha = 0.07 * save_exp(-(v + 65) / 20)
    beta = 1.0 / (save_exp(-(v + 35) / 10) + 1)
    return alpha, beta


def n_gate(v):
    """Return the activation and decay rates for the potassium gate."""
    alpha = 0.01 * _vtrap(-(v + 55), 10)
    beta = 0.125 * save_exp(-(v + 65) / 80)
    return alpha, beta


def _vtrap(x, y, epsilon=1e-12):
    """Compute the voltage-dependent trap term used by the gating equations."""
    return x / (save_exp(x / y) - 1.0 + epsilon)


def calculate_cable_scaling(edges, ro, x, y, z, r):
    """Compute per-edge axial resistance from cable geometry."""
    cross_section_area = jnp.pi * r**2
    source_nodes = edges[:, 0]
    target_nodes = edges[:, 1]

    dx = x.at[target_nodes].get() - x.at[source_nodes].get()
    dy = y.at[target_nodes].get() - y.at[source_nodes].get()
    dz = z.at[target_nodes].get() - z.at[source_nodes].get()

    cable_length = (dx**2 + dy**2 + dz**2)**0.5
    return (ro * cable_length) / cross_section_area.at[target_nodes].get()


def get_cabble_pipeline(edges):
    """Build a JIT-compiled cable propagation pipeline for the provided edges."""
    edge_array = jnp.array(edges, jnp.int32)
    source_nodes = edge_array[:, 0]
    target_nodes = edge_array[:, 1]

    def graph_evolution_fn_with_scaling(state, dx_dt):
        x = state['morphology']['position']['x']
        y = state['morphology']['position']['y']
        z = state['morphology']['position']['z']
        radius = state['morphology']['r']
        resistivity = state['morphology']['ro']

        scaling = calculate_cable_scaling(edges, resistivity, x, y, z, radius)
        potential_difference = (
            state['V'].at[target_nodes].get(
            ) - state['V'].at[source_nodes].get()
        )

        # Apply the same cable update symmetrically at both ends of each edge.
        dx_dt['V'] = dx_dt['V'].at[source_nodes].add(
            potential_difference * scaling.at[source_nodes].get()
        )
        dx_dt['V'] = dx_dt['V'].at[target_nodes].add(
            -potential_difference * scaling.at[target_nodes].get()
        )
        return dx_dt

    return jax.jit(graph_evolution_fn_with_scaling)


def get_Na_channel_pipeline():
    """Return the JIT-compiled sodium channel update pipeline."""

    @jax.jit
    def Na_pipeline(state, ds_dt):
        membrane_voltage = state['V']
        m = state['morphology']['Na']['m']
        h = state['morphology']['Na']['h']
        sodium_conductance = state['morphology']['Na']['gNa']
        sodium_reversal = state['morphology']['Na']['eNa']
        membrane_capacitance = state['morphology']['C']

        alpha_m, beta_m = m_gate(membrane_voltage)
        dm = alpha_m * (1.0 - m) - beta_m * m

        alpha_h, beta_h = h_gate(membrane_voltage)
        dh = alpha_h * (1.0 - h) - beta_h * h

        sodium_current = sodium_conductance * h * \
            (m**3) * (membrane_voltage - sodium_reversal)

        ds_dt['morphology']['Na']['m'] = ds_dt['morphology']['Na']['m'] + dm
        ds_dt['morphology']['Na']['h'] = ds_dt['morphology']['Na']['h'] + dh
        ds_dt['V'] = ds_dt['V'] - sodium_current / membrane_capacitance
        return ds_dt

    return Na_pipeline


def get_K_channel_pipeline():
    """Return the JIT-compiled potassium channel update pipeline."""

    @jax.jit
    def K_pipeline(state, ds_dt):
        membrane_voltage = state['V']
        n = state['morphology']['K']['n']
        potassium_conductance = state['morphology']['K']['gK']
        potassium_reversal = state['morphology']['K']['eK']
        membrane_capacitance = state['morphology']['C']

        alpha_n, beta_n = n_gate(membrane_voltage)
        dn = alpha_n * (1.0 - n) - beta_n * n

        potassium_current = potassium_conductance * \
            (n**4) * (membrane_voltage - potassium_reversal)

        ds_dt['morphology']['K']['n'] = ds_dt['morphology']['K']['n'] + dn
        ds_dt['V'] = ds_dt['V'] - potassium_current / membrane_capacitance
        return ds_dt

    return K_pipeline


def get_leak_channel_pipeline():
    """Return the JIT-compiled leak-current update pipeline."""

    @jax.jit
    def leak_pipeline(state, ds_dt):
        membrane_voltage = state['V']
        leak_conductance = state['morphology']['leak']['gLeak']
        leak_reversal = state['morphology']['leak']['eLeak']
        membrane_capacitance = state['morphology']['C']

        leak_current = leak_conductance * (membrane_voltage - leak_reversal)

        ds_dt['V'] = ds_dt['V'] - leak_current / membrane_capacitance
        return ds_dt

    return leak_pipeline


def get_stub_synapse_pipeline(pre_syn_edges, post_syn_edges):
    """Return a JIT-compiled stub synapse update pipeline."""
    pre_cable_idx = jnp.array(pre_syn_edges[0, :], dtype=jnp.int32)
    syn_pre_idx = jnp.array(pre_syn_edges[1, :], dtype=jnp.int32)

    syn_post_idx = jnp.array(post_syn_edges[0, :], dtype=jnp.int32)
    post_cable_idx = jnp.array(post_syn_edges[1, :], dtype=jnp.int32)

    @jax.jit
    def synapse_pipeline(state, ds_dt):
        cable_voltage = state['V']
        synapse_voltage = state['connectors']['stub']['V']
        synapse_weights = state['connectors']['stub']['weight']
        membrane_capacitance = state['morphology']['C']

        # Update synapse voltage from the presynaptic cable using a sigmoid.
        presynaptic_voltage = cable_voltage[pre_cable_idx]
        target_synapse_voltage = jax.nn.sigmoid(presynaptic_voltage)
        active_synapse_voltage = synapse_voltage[syn_pre_idx]
        ds_dt['connectors']['stub']['V'] = ds_dt['connectors']['stub']['V'].at[syn_pre_idx].add(
            target_synapse_voltage - active_synapse_voltage
        )

        # Synchronize postsynaptic voltage with the synapse state, scaled by weight.
        mapped_synapse_voltage = synapse_voltage[syn_post_idx]
        postsynaptic_voltage = cable_voltage[post_cable_idx]
        mapped_weights = synapse_weights[syn_post_idx]
        postsynaptic_capacitance = membrane_capacitance[post_cable_idx]

        sync_term = (mapped_synapse_voltage - postsynaptic_voltage) * \
            mapped_weights / postsynaptic_capacitance
        ds_dt['V'] = ds_dt['V'].at[post_cable_idx].add(sync_term)
        return ds_dt

    return synapse_pipeline


@jax.jit
def shift_operator(z: jnp.ndarray, tau_r, tau_d) -> jnp.ndarray:
    # 1. Главная диагональ: -1/tau_r * z_i для всех элементов
    # Это дает базовое экспоненциальное затухание во всех узлах
    out = - (1.0 / tau_r) * z
    
    # 2. Субдиагональ (перенос вперед): 1/tau_d * z_{i-1} -> записывается в z_i
    # Элемент z_0 переносится в z_1, z_1 в z_2 и т.д.
    # Мы сдвигаем исходный вектор z вправо и масштабируем его
    forward_flow = (1.0 / tau_d) * z[:-1]
    out = out.at[1:].add(forward_flow)
    
    # 3. Супердиагональ (обратное влияние): -1/tau_d * z_{i+1} -> записывается в z_i
    # Элемент z_2 переносится в z_1, z_3 в z_2 и т.д.
    # Мы сдвигаем исходный вектор z влево и масштабируем его
    backward_flow = - (1.0 / tau_d) * z[1:]
    out = out.at[:-1].add(backward_flow)
    
    return out

 
def get_dummy_delay_synapse_pipeline(pre_syn_edges, post_syn_edges): # edges_H_to_S, edges_S_to_H
    pre_cable_idx = jnp.array(pre_syn_edges[0, :], dtype=jnp.int32)
    syn_pre_idx = jnp.array(pre_syn_edges[1, :], dtype=jnp.int32)

    syn_post_idx = jnp.array(post_syn_edges[0, :], dtype=jnp.int32)
    post_cable_idx = jnp.array(post_syn_edges[1, :], dtype=jnp.int32)



    @jax.jit
    def synapse_pipeline(state, ds_dt):
        cable_voltage = state['V']
        # membrane_capacitance = state['morphology']['C']
        tau_d = state['connectors']['dummy_delay']['tau_d']
        tau_r = state['connectors']['dummy_delay']['tau_r']
        weight = state['connectors']['dummy_delay']['weight']
        slope = state['connectors']['dummy_delay']['slope']
        bias = state['connectors']['dummy_delay']['bias']
        z = state['connectors']['dummy_delay']['z']

        presynaptic_voltage = cable_voltage[pre_cable_idx]
        ds_dt['connectors']['dummy_delay']['z'] = ds_dt['connectors']['dummy_delay']['z'].at[syn_pre_idx, 0].add(presynaptic_voltage)

        ds_dt['connectors']['dummy_delay']['z'] = ds_dt['connectors']['dummy_delay']['z'] + jax.vmap(shift_operator)(z, tau_r, tau_d)

        F_out = weight * jax.nn.sigmoid(z[:, -1]*slope - bias)

        ds_dt['V'] = ds_dt['V'].at[post_cable_idx].add(F_out[syn_post_idx])
        return ds_dt

    return synapse_pipeline