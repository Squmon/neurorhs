"""JAX-based neuron model pipelines for channel and synapse updates.

The module preserves the existing public API while exposing the internal
state updates through small helper functions and descriptive docstrings.
"""

from typing import Callable

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

    def graph_evolution_fn_with_scaling(state, ds_dt, t):
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
        ds_dt['V'] = ds_dt['V'].at[source_nodes].add(
            potential_difference * scaling.at[source_nodes].get()
        )
        ds_dt['V'] = ds_dt['V'].at[target_nodes].add(
            -potential_difference * scaling.at[target_nodes].get()
        )
        return ds_dt

    return jax.jit(graph_evolution_fn_with_scaling)


def get_Na_channel_pipeline():
    """Return the JIT-compiled sodium channel update pipeline."""

    @jax.jit
    def Na_pipeline(state, ds_dt, t):
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
    def K_pipeline(state, ds_dt, t):
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
    def leak_pipeline(state, ds_dt, t):
        membrane_voltage = state['V']
        leak_conductance = state['morphology']['leak']['gLeak']
        leak_reversal = state['morphology']['leak']['eLeak']
        membrane_capacitance = state['morphology']['C']

        leak_current = leak_conductance * (membrane_voltage - leak_reversal)

        ds_dt['V'] = ds_dt['V'] - leak_current / membrane_capacitance
        return ds_dt

    return leak_pipeline


def get_stim_pipeline(schedule: tuple[jnp.array, Callable]):

    def filt(state, ds_dt, t):
        for inds, c in schedule:
            C = state['morphology']['C'].at[inds].get()
            ds_dt['V'] = ds_dt['V'].at[inds].add(c(state, ds_dt, t)/C)
        return ds_dt

    return filt


def get_stim_pipeline_from_original_ids(mapping, schedule: tuple[tuple[str], Callable]):
    schd = []
    for inds, c in schedule:
        mapped = jnp.array([mapping[ind] for ind in inds], dtype=jnp.int32)
        schd.append((mapped, c))
    return get_stim_pipeline(tuple(schd))


def get_stub_synapse_pipeline(pre_syn_edges, post_syn_edges):
    """Return a JIT-compiled stub synapse update pipeline."""
    pre_cable_idx = jnp.array(pre_syn_edges[0, :], dtype=jnp.int32)
    syn_pre_idx = jnp.array(pre_syn_edges[1, :], dtype=jnp.int32)

    syn_post_idx = jnp.array(post_syn_edges[0, :], dtype=jnp.int32)
    post_cable_idx = jnp.array(post_syn_edges[1, :], dtype=jnp.int32)

    @jax.jit
    def synapse_pipeline(state, ds_dt, t):
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


# edges_H_to_S, edges_S_to_H
def get_dummy_delay_synapse_pipeline(pre_syn_edges, post_syn_edges):
    pre_cable_idx = jnp.array(pre_syn_edges[0, :], dtype=jnp.int32)
    syn_pre_idx = jnp.array(pre_syn_edges[1, :], dtype=jnp.int32)

    syn_post_idx = jnp.array(post_syn_edges[0, :], dtype=jnp.int32)
    post_cable_idx = jnp.array(post_syn_edges[1, :], dtype=jnp.int32)

    @jax.jit
    def synapse_pipeline(state, ds_dt, t):
        cable_voltage = state['V']
        # membrane_capacitance = state['morphology']['C']
        tau_d = state['connectors']['dummy_delay']['tau_d']
        tau_r = state['connectors']['dummy_delay']['tau_r']
        weight = state['connectors']['dummy_delay']['weight']
        slope = state['connectors']['dummy_delay']['slope']
        bias = state['connectors']['dummy_delay']['bias']
        z = state['connectors']['dummy_delay']['z']

        presynaptic_voltage = cable_voltage[pre_cable_idx]
        ds_dt['connectors']['dummy_delay']['z'] = ds_dt['connectors']['dummy_delay']['z'].at[syn_pre_idx, 0].add(
            presynaptic_voltage)

        ds_dt['connectors']['dummy_delay']['z'] = ds_dt['connectors']['dummy_delay']['z'] + \
            jax.vmap(shift_operator)(z, tau_r, tau_d)

        F_out = weight * jax.nn.sigmoid(z[:, -1]*slope - bias)

        ds_dt['V'] = ds_dt['V'].at[post_cable_idx].add(F_out[syn_post_idx])
        return ds_dt

    return synapse_pipeline


def __L(V_pre, s):
    L_max = s['L_max']
    V_p = s['V_p']
    K_p = s['K_p']
    return L_max/(1 + jnp.exp(
        -(V_pre - V_p)/K_p
    ))


# TODO using sum_i pi = 1, reduce dims to N - 1
def get_kinetic_synapce_pipeline(
    Q: dict[str, Callable],
    g_syn,
    name,
    pre_syn_edges, post_syn_edges, __L = __L
):
    pre_cable_idx = jnp.array(
        pre_syn_edges[0, :], dtype=jnp.int32)  # те идут в синапс
    # те которые принимают кабель
    syn_pre_idx = jnp.array(pre_syn_edges[1, :], dtype=jnp.int32)

    # те которые идут в кабель
    syn_post_idx = jnp.array(post_syn_edges[0, :], dtype=jnp.int32)
    # те кабеля, которые принимают синапс
    post_cable_idx = jnp.array(post_syn_edges[1, :], dtype=jnp.int32)

    Ps = set()
    for k, r in Q.items():
        a, b = k.split('->')
        assert a != b
        Ps.add(a)
        Ps.add(b)

    @jax.jit
    def synapse_pipeline(state, ds_dt, t):
        # assert state['connectors'][name]['E'] is not None
        # for k in Ps:
        #    assert state['connectors'][name]['P'][k] is not None

        cable_voltage = state['V']
        membrane_capacitance = state['morphology']['C']
        post_membrane_capacitance = membrane_capacitance[post_cable_idx]

        synapce_state = state['connectors'][name]
        synapce_ds_dt = ds_dt['connectors'][name]

        presynaptic_voltage = cable_voltage[pre_cable_idx]
        postsynaptic_voltage = cable_voltage[post_cable_idx]
        L = __L(presynaptic_voltage, synapce_state)[syn_pre_idx] # concentration

        for k, r in Q.items():
            a, b = k.split('->')
            synapce_ds_dt['P'][b] = synapce_ds_dt['P'][b].at[syn_pre_idx].add(
                + r(L, synapce_state)*synapce_state['P'][a][syn_pre_idx])
            synapce_ds_dt['P'][a] = synapce_ds_dt['P'][a].at[syn_pre_idx].add(
                - r(L, synapce_state)*synapce_state['P'][a][syn_pre_idx])

        I = g_syn(synapce_state)[
            syn_post_idx]*(synapce_state['E'][syn_post_idx] - postsynaptic_voltage)

        ds_dt['V'] = ds_dt['V'].at[post_cable_idx].add(
            I/post_membrane_capacitance)
        ds_dt['connectors'][name] = synapce_ds_dt
        return ds_dt

    return synapse_pipeline


def get_component2_syn(pre_syn_edges, post_syn_edges):
    Q = {
        'C->O': lambda L, s: s['r1']*L,
        'O->C': lambda L, s: s['r2']
    }
    name = 'comp2'
    return get_kinetic_synapce_pipeline(Q, lambda s: s['g']*s['P']['O'], name, pre_syn_edges, post_syn_edges)
