from neurorhs.configs.default import *
from neurorhs.preprocessing.graph_to_arrays import process_graph_to_core_arrays, load_context
import networkx as nx
import matplotlib.pyplot as plt


from neurorhs.neurosci import *
import diffrax


class AbstractHHSimulation(FooConfig):
    def __init__(self, root_ctx, default_r=10.0, stimulus=None):
        ctx = root_ctx['root']
        self.stimulus = stimulus

        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']

        x = root_ctx['additional_data']['x']
        y = root_ctx['additional_data']['y']
        z = root_ctx['additional_data']['z']
        r = root_ctx['additional_data']['r']
        r = jnp.nan_to_num(r, nan=default_r)

        default_arguments = {
            'V': jnp.ones((num_H,), dtype=jnp.float32) * -65.0,
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
            }
        }


        is_dynamic = jax.tree_util.tree_map(
            lambda x: True, default_arguments)

        groups = jax.tree_util.tree_map(lambda x: 'H', default_arguments)
        super().__init__(ctx, default_arguments, is_dynamic, groups)

    def construct_f_implicit(self):
        cable_m = self.ctx['edges_H_to_H']
        cabble_pipe = get_cabble_pipeline(cable_m)

        def f_implicit(s, ds_dt, t):
            ds_dt = cabble_pipe(s, ds_dt, t)
            return ds_dt
        return f_implicit

    def construct_f_explicit(self):
        na_pipe = get_Na_channel_pipeline()
        k_pipe = get_K_channel_pipeline()
        leak_pipe = get_leak_channel_pipeline()

        def f_explicit(s, ds_dt, t):
            ds_dt = na_pipe(s, ds_dt, t)
            ds_dt = k_pipe(s, ds_dt, t)
            ds_dt = leak_pipe(s, ds_dt, t)
            if self.stimulus is not None:
                ds_dt = self.stimulus(s, ds_dt, t)
            return ds_dt
        return f_explicit


class DDSsynFoo(AbstractHHSimulation):
    def __init__(self, root_ctx, default_r=10, N_ddp=5, stimulus=None):
        super().__init__(root_ctx, default_r, stimulus)
        ctx = root_ctx['root']
        self.stimulus = stimulus

        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']
        conn_state = {'dummy_delay': {
            'z': jnp.ones((num_S, N_ddp), dtype=jnp.float32),
            'tau_d': jnp.ones((num_S, ), dtype=jnp.float32),
            'tau_r': jnp.ones((num_S, ), dtype=jnp.float32)*0.1,
            'weight': jnp.ones((num_S, ), dtype=jnp.float32)*70.0,
            'slope': jnp.ones((num_S, ), dtype=jnp.float32),
            'bias': jnp.ones((num_S, ), dtype=jnp.float32)*0,
        }}
        self.default_arguments['connectors'] = conn_state
        self.is_dynamic['connectors'] = jax.tree_util.tree_map(lambda x: True, conn_state)
        self.groups['connectors'] = jax.tree_util.tree_map(lambda x: 'S', conn_state)

    def construct_f_explicit(self):
        pre_syn = self.ctx['edges_H_to_S']
        post_syn = self.ctx['edges_S_to_H']
        sp = super().construct_f_explicit()
        syn_pipe = get_dummy_delay_synapse_pipeline(pre_syn, post_syn)

        def f_explicit_generated(s, ds_dt, t):
            ds_dt = sp(s, ds_dt, t)
            ds_dt = syn_pipe(s, ds_dt, t)
            return ds_dt

        return f_explicit_generated


class StubSynFoo(AbstractHHSimulation):
    def __init__(self, root_ctx, default_r=10, stimulus=None):
        super().__init__(root_ctx, default_r, stimulus)
        ctx = root_ctx['root']
        self.stimulus = stimulus

        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']
        conn_state = {
                'stub': {
                    'V': jnp.zeros((num_S,), dtype=jnp.float32),
                    'weight': jnp.ones((num_S,), dtype=jnp.float32) * 0.5,
                }
            }
        self.default_arguments['connectors'] = conn_state
        self.is_dynamic['connectors'] = jax.tree_util.tree_map(lambda x: True, conn_state)
        self.groups['connectors'] = jax.tree_util.tree_map(lambda x: 'S', conn_state)

    def construct_f_explicit(self):
        pre_syn = self.ctx['edges_H_to_S']
        post_syn = self.ctx['edges_S_to_H']
        sp = super().construct_f_explicit()
        syn_pipe = get_stub_synapse_pipeline(pre_syn, post_syn)

        def f_explicit_generated(s, ds_dt, t):
            ds_dt = sp(s, ds_dt, t)
            ds_dt = syn_pipe(s, ds_dt, t)
            return ds_dt

        return f_explicit_generated


class DDSsynFoo_static_params(DDSsynFoo):
    def __init__(self, root_ctx, default_r=10, N_ddp=5, stimulus=None):
        super().__init__(root_ctx, default_r, N_ddp, stimulus)
        self.is_dynamic['morphology']['position']['x'] = False
        self.is_dynamic['morphology']['position']['y'] = False
        self.is_dynamic['morphology']['position']['z'] = False
        self.is_dynamic['morphology']['r'] = False
        self.is_dynamic['morphology']['C'] = False
        self.is_dynamic['morphology']['Na']['gNa'] = False
        self.is_dynamic['morphology']['Na']['eNa'] = False
        self.is_dynamic['morphology']['K']['gK'] = False
        self.is_dynamic['morphology']['K']['eK'] = False
        self.is_dynamic['morphology']['leak']['gLeak'] = False
        self.is_dynamic['morphology']['leak']['eLeak'] = False

        self.is_dynamic['connectors']['dummy_delay']['tau_d'] = False
        self.is_dynamic['connectors']['dummy_delay']['tau_r'] = False
        self.is_dynamic['connectors']['dummy_delay']['weight'] = False
        self.is_dynamic['connectors']['dummy_delay']['slope'] = False
        self.is_dynamic['connectors']['dummy_delay']['bias'] = False

class StubSynFoo_static_params(StubSynFoo):
    def __init__(self, root_ctx, default_r=10, stimulus=None):
        super().__init__(root_ctx, default_r, stimulus)
        self.is_dynamic['morphology']['position']['x'] = False
        self.is_dynamic['morphology']['position']['y'] = False
        self.is_dynamic['morphology']['position']['z'] = False
        self.is_dynamic['morphology']['r'] = False
        self.is_dynamic['morphology']['C'] = False
        self.is_dynamic['morphology']['Na']['gNa'] = False
        self.is_dynamic['morphology']['Na']['eNa'] = False
        self.is_dynamic['morphology']['K']['gK'] = False
        self.is_dynamic['morphology']['K']['eK'] = False
        self.is_dynamic['morphology']['leak']['gLeak'] = False
        self.is_dynamic['morphology']['leak']['eLeak'] = False

        self.is_dynamic['connectors']['stub']['weight'] = False


def test_basic_simulation_pipeline(
    generated_dir
):
    npz_path = generated_dir / "test_preprocess_output.jconn"
    img_path = generated_dir / "basic_sim_result.png"
    root_ctx = load_context(str(npz_path))
    foo = StubSynFoo_static_params(root_ctx)
    sim = DefaultSim(foo)
    sol = sim.solve(0, 200, num=200)
    plt.plot(sol.ts, sol.ys['V'])
    plt.savefig(str(img_path))



def test_dds_simulation_pipeline(
    generated_dir
):
    npz_path = generated_dir / "test_preprocess_output.jconn"
    img_path = generated_dir / "dds_sim_result.png"
    root_ctx = load_context(str(npz_path))
    foo = DDSsynFoo_static_params(root_ctx, N_ddp = 1)
    sim = DefaultSim(foo)
    sol = sim.solve(0, 200, num=200)
    plt.plot(sol.ts, sol.ys['V'])
    plt.savefig(str(img_path))


def test_basic_simulation_pipeline_with_stimulus(
    generated_dir
):
    npz_path = generated_dir / "test_preprocess_output.jconn"
    img_path = generated_dir / "basic_sim_result_stimula.png"
    img_path1 = generated_dir / "basic_sim_result_stimula_only_stimulated.png"
    root_ctx = load_context(str(npz_path))

    iclamp = lambda state, ds_dt, t: 70*(t > 20)
    mapping = root_ctx['root']['mapping']['H']
    stimula = get_stim_pipeline_from_original_ids(mapping, 
                                           ((['25019976'], iclamp), ))
    
    foo = StubSynFoo_static_params(root_ctx, stimulus = stimula)

    sim = DefaultSim(foo)
    sol = sim.solve(0, 100, num=200)
    plt.plot(sol.ts, sol.ys['V'])
    plt.savefig(str(img_path))
    plt.clf()

    plt.plot(sol.ts, sol.ys['V'].T[mapping['25019976']])
    plt.savefig(str(img_path1))

