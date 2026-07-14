from neurorhs.configs.default import *
from neurorhs.preprocessing.graph_to_arrays import process_graph_to_core_arrays, load_jax_context
import networkx as nx
import matplotlib.pyplot as plt


from neurorhs.neurosci import (
    get_Na_channel_pipeline,
    get_K_channel_pipeline,
    get_leak_channel_pipeline,
    get_stub_synapse_pipeline,
    get_cabble_pipeline,
)
import diffrax


class FooFromNpz(FooConfig):
    def __init__(self, npz_path):
        root_ctx = load_jax_context(npz_path)
        ctx = root_ctx['root']

        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']

        x = root_ctx['additional_data']['x']
        y = root_ctx['additional_data']['y']
        z = root_ctx['additional_data']['z']
        r = root_ctx['additional_data']['r']

        default_arguments = {
            'V': jnp.ones((num_H,), dtype=jnp.float32) * -65.0,
            'time': 0.0,
            'morphology': {
                'C': jnp.ones((num_H,), dtype=jnp.float32),
                'position': {
                    'x': x,
                    'y': y,
                    'z': z
                },

                'r': r,
                'ro': 1.0,
                'Na': {
                    'm': jnp.ones((num_H,), dtype=jnp.float32) * 0.0220,
                    'h': jnp.ones((num_H,), dtype=jnp.float32) * 0.9840,
                    'gNa': jnp.ones((num_H,), dtype=jnp.float32) * 120.0,
                    'eNa': jnp.ones((num_H,), dtype=jnp.float32) * 50.0,
                },
                'K': {
                    'n': jnp.ones((num_H,), dtype=jnp.float32) * 0.0773,
                    'gK': jnp.ones((num_H,), dtype=jnp.float32) * 36.0,
                    'eK': jnp.ones((num_H,), dtype=jnp.float32) * -77.0,
                },
                'leak': {
                    'gLeak': jnp.ones((num_H,), dtype=jnp.float32) * 0.3,
                    'eLeak': jnp.ones((num_H,), dtype=jnp.float32) * -54.4,
                }
            },
            'connectors': {
                'stub': {
                    'V': jnp.zeros((num_S,), dtype=jnp.float32),
                    'weight': jnp.ones((num_S,), dtype=jnp.float32) * 0.5,
                }
            }
        }
        is_dynamic = jax.tree_util.tree_map(lambda x: True, default_arguments)
        is_dynamic['morphology']['position']['x'] = False
        is_dynamic['morphology']['position']['y'] = False
        is_dynamic['morphology']['position']['z'] = False
        is_dynamic['morphology']['r'] = False
        is_dynamic['morphology']['C'] = False
        is_dynamic['morphology']['Na']['gNa'] = False
        is_dynamic['morphology']['Na']['eNa'] = False
        is_dynamic['morphology']['K']['gK'] = False
        is_dynamic['morphology']['K']['eK'] = False
        is_dynamic['morphology']['leak']['gLeak'] = False
        is_dynamic['morphology']['leak']['eLeak'] = False

        is_dynamic['connectors']['stub']['weight'] = False
        super().__init__(ctx, default_arguments, is_dynamic)

    def get_f_implicit(self):
        cable_m = self.ctx['edges_H_to_H']
        cabble_pipe = get_cabble_pipeline(cable_m)

        def f_implicit(t, y, args):
            s, ds_dt = self.setup(y)
            ds_dt = jax.tree_util.tree_map(jnp.zeros_like, y)
            ds_dt = cabble_pipe(s, ds_dt)
            return ds_dt
        return f_implicit

    def get_f_explicit(self):
        pre_syn = self.ctx['edges_H_to_S']
        post_syn = self.ctx['edges_S_to_H']
        na_pipe = get_Na_channel_pipeline()
        k_pipe = get_K_channel_pipeline()
        leak_pipe = get_leak_channel_pipeline()

        syn_pipe = get_stub_synapse_pipeline(pre_syn, post_syn)

        def f_explicit(t, y, args):
            s, ds_dt = self.setup(y)
            ds_dt = na_pipe(s, ds_dt)
            ds_dt = k_pipe(s, ds_dt)
            ds_dt = leak_pipe(s, ds_dt)
            ds_dt = syn_pipe(s, ds_dt)
            return ds_dt
        return f_explicit


def test_simulation_pipeline(
    generated_dir
):
    npz_path = generated_dir / "test_preprocess_output.bin"
    img_path = generated_dir / "sim_result.png"
    foo = FooFromNpz(str(npz_path))
    sim = DefaultSim(foo)
    sol = sim.solve(0, 200, num=200)
    plt.plot(sol.ts, sol.ys['V'])
    plt.savefig(str(img_path))
