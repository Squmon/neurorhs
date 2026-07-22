from neurorhs.configs.default import *
from neurorhs.preprocessing.graph_to_arrays import process_graph_to_core_arrays
from neurorhs.io import load_context
from neurorhs.preprocessing.preprocess import collapse_nodes, process_params
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from neurorhs.utils import apply_mappers, Mapper


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


class PointAbstractHHSimulation(AbstractHHSimulation):
    def construct_f_implicit(self):
        return lambda s, ds_dt, t:ds_dt

class KineticSyn(AbstractHHSimulation):
    def __init__(self, root_ctx, model, name, P_defaut, other_states, default_r=10, stimulus=None):
        super().__init__(root_ctx, default_r, stimulus)
        ctx = root_ctx['root']
        self.stimulus = stimulus
        self.model = model
        self.model_name = name

        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']

        Q = jnp.zeros_like(num_S, dtype=jnp.float32)
        for k, v in P_defaut.items():
            Q += v
        assert all(Q == jnp.ones(num_S, ))
        conn_state = {name: {
            'E': jnp.zeros(num_S),
            'L_max': 2.84,
            'V_p': 2,
            'K_p': 5,
            'P': P_defaut
        } | other_states
        }
        self.default_arguments['connectors'] = conn_state
        self.is_dynamic['connectors'] = jax.tree_util.tree_map(
            lambda x: True, conn_state)
        self.groups['connectors'] = jax.tree_util.tree_map(
            lambda x: 'S', conn_state)

    def construct_f_explicit(self):
        pre_syn = self.ctx['edges_H_to_S']
        post_syn = self.ctx['edges_S_to_H']
        sp = super().construct_f_explicit()
        syn_pipe = self.model(pre_syn, post_syn)

        def f_explicit_generated(s, ds_dt, t):
            ds_dt = sp(s, ds_dt, t)
            ds_dt = syn_pipe(s, ds_dt, t)
            return ds_dt

        return f_explicit_generated

class KineticSynPoint(PointAbstractHHSimulation):
    def __init__(self, root_ctx, model, name, P_defaut, other_states, default_r=10, stimulus=None):
        super().__init__(root_ctx, default_r, stimulus)
        ctx = root_ctx['root']
        self.stimulus = stimulus
        self.model = model
        self.model_name = name

        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']

        Q = jnp.zeros_like(num_S, dtype=jnp.float32)
        for k, v in P_defaut.items():
            Q += v
        assert all(Q == jnp.ones(num_S, ))
        conn_state = {name: {
            'E': jnp.zeros(num_S),
            'L_max': 2.84,
            'V_p': 2,
            'K_p': 5,
            'P': P_defaut
        } | other_states
        }
        self.default_arguments['connectors'] = conn_state
        self.is_dynamic['connectors'] = jax.tree_util.tree_map(
            lambda x: True, conn_state)
        self.groups['connectors'] = jax.tree_util.tree_map(
            lambda x: 'S', conn_state)

    def construct_f_explicit(self):
        pre_syn = self.ctx['edges_H_to_S']
        post_syn = self.ctx['edges_S_to_H']
        sp = super().construct_f_explicit()
        syn_pipe = self.model(pre_syn, post_syn)

        def f_explicit_generated(s, ds_dt, t):
            ds_dt = sp(s, ds_dt, t)
            ds_dt = syn_pipe(s, ds_dt, t)
            return ds_dt

        return f_explicit_generated

class Syn2Comp(KineticSyn):
    def __init__(self, root_ctx, default_r=10, stimulus=None):
        ctx = root_ctx['root']
        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']
        P_defaut = {
            'C': jnp.ones(num_S, dtype=jnp.float32),
            'O': jnp.zeros(num_S, dtype=jnp.float32),
        }
        other_states = {
            'r1': 0.1,
            'r2': 0.01,
            'g': 2.0,
        }
        super().__init__(root_ctx, get_component2_syn, 'comp2', P_defaut,
                         other_states=other_states, default_r=default_r, stimulus=stimulus)


class Syn2CompPoint(KineticSynPoint):
    def __init__(self, root_ctx, default_r=10, stimulus=None):
        ctx = root_ctx['root']
        num_H = ctx['num_nodes']['H']
        num_S = ctx['num_nodes']['S']
        P_defaut = {
            'C': jnp.ones(num_S, dtype=jnp.float32),
            'O': jnp.zeros(num_S, dtype=jnp.float32),
        }
        other_states = {
            'r1': 0.1,
            'r2': 0.01,
            'g': 2.0,
        }
        super().__init__(root_ctx, get_component2_syn, 'comp2', P_defaut,
                         other_states=other_states, default_r=default_r, stimulus=stimulus)

class Syn2Comp_static_params(Syn2Comp):
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

        self.is_dynamic['connectors']['comp2']['r1'] = False
        self.is_dynamic['connectors']['comp2']['r2'] = False
        self.is_dynamic['connectors']['comp2']['E'] = False
        self.is_dynamic['connectors']['comp2']['L_max'] = False
        self.is_dynamic['connectors']['comp2']['V_p'] = False
        self.is_dynamic['connectors']['comp2']['K_p'] = False
        self.is_dynamic['connectors']['comp2']['g'] = False

class Syn2Comp_static_paramsPoint(Syn2CompPoint):
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

        self.is_dynamic['connectors']['comp2']['r1'] = False
        self.is_dynamic['connectors']['comp2']['r2'] = False
        self.is_dynamic['connectors']['comp2']['E'] = False
        self.is_dynamic['connectors']['comp2']['L_max'] = False
        self.is_dynamic['connectors']['comp2']['V_p'] = False
        self.is_dynamic['connectors']['comp2']['K_p'] = False
        self.is_dynamic['connectors']['comp2']['g'] = False


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
        self.is_dynamic['connectors'] = jax.tree_util.tree_map(
            lambda x: True, conn_state)
        self.groups['connectors'] = jax.tree_util.tree_map(
            lambda x: 'S', conn_state)

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
        self.is_dynamic['connectors'] = jax.tree_util.tree_map(
            lambda x: True, conn_state)
        self.groups['connectors'] = jax.tree_util.tree_map(
            lambda x: 'S', conn_state)

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
    foo = DDSsynFoo_static_params(root_ctx, N_ddp=1)
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

    def iclamp(state, ds_dt, t): return 70*(t > 20)
    mapping = root_ctx['root']['mapping']['H']

    neurites_to_stimul = [
        '10675427',
        '11281421']

    stimula = get_stim_pipeline_from_original_ids(
        mapping, ((neurites_to_stimul, iclamp),))

    foo = StubSynFoo_static_params(root_ctx, stimulus=stimula)

    sim = DefaultSim(foo)
    sol = sim.solve(0, 100, num=200)
    plt.plot(sol.ts, sol.ys['V'])
    plt.savefig(str(img_path))
    plt.clf()

    mapped = apply_mappers(sol.ys, foo.get_dynamic_static_parts(
        foo.groups)[0], foo.ctx['mapping'])

    for i in neurites_to_stimul:
        plt.plot(sol.ts, mapped['V'][..., i], label=i)
    plt.legend()
    plt.savefig(str(img_path1))


def test_basic_simulation_pipeline_with_stimulus_2comp(
    generated_dir
):
    npz_path = generated_dir / "test_preprocess_output.jconn"
    img_path = generated_dir / "basic_sim_result_stimula_2comp.png"
    img_path1 = generated_dir / "basic_sim_result_stimula_only_stimulated_2comp.png"
    img_path2 = generated_dir / "basic_sim_result_stimula_only_stimulated_2comp_C.png"
    img_path3 = generated_dir / "basic_sim_result_stimula_only_stimulated_2comp_O.png"
    root_ctx = load_context(str(npz_path))

    def iclamp(state, ds_dt, t): return 70*(t > 20)
    mapping = root_ctx['root']['mapping']['H']

    neurites_to_stimul = [
        '10675427',
        '11281421']

    stimula = get_stim_pipeline_from_original_ids(
        mapping, ((neurites_to_stimul, iclamp),))

    foo = Syn2Comp_static_params(root_ctx, stimulus=stimula)

    sim = DefaultSim(foo)
    sol = sim.solve(0, 100, num=200)
    plt.plot(sol.ts, sol.ys['V'])
    plt.savefig(str(img_path))
    plt.clf()

    mapped = apply_mappers(sol.ys, foo.get_dynamic_static_parts(
        foo.groups)[0], foo.ctx['mapping'])

    for i in neurites_to_stimul:
        plt.plot(sol.ts, mapped['V'][..., i], label=i)
    plt.legend()
    plt.savefig(str(img_path1))
    plt.clf()

    plt.title('C')
    plt.plot(sol.ts, sol.ys['connectors']['comp2']['P']['C'])
    plt.savefig(str(img_path2))
    plt.clf()

    plt.title('O')
    plt.plot(sol.ts, sol.ys['connectors']['comp2']['P']['O'])
    plt.savefig(str(img_path3))
    plt.clf()


def test_point_simulation(
    graph_path,
    metadata_path,
    type_groups,
    directedness,
    generated_dir
):
    # 1. Load the original graph
    g = nx.read_gml(graph_path)

    # 2. Build mapping: collapse all morphology nodes to their "name" property (neuron ID)
    mapping = {}
    for node, data in g.nodes(data=True):
        node_type = data.get('type')
        if node_type != 'connector':
            neuron_id = data.get('name')
            if neuron_id:
                mapping[node] = str(neuron_id)

    # 3. Collapse the nodes
    collapsed_g = collapse_nodes(g, mapping)

    # Set type="root" for all the new morphology nodes
    for node in list(collapsed_g.nodes()):
        node_type = collapsed_g.nodes[node].get('type')
        if node_type != 'connector':
            collapsed_g.nodes[node]['type'] = 'root'

    # Save collapsed graph to a temp path
    collapsed_gml_path = generated_dir / "collapsed_20n.gml"
    nx.write_gml(collapsed_g, str(collapsed_gml_path))

    # 4. Process the metadata
    # Load metadata and filter to only keep rows with type == 'root'
    df = pd.read_csv(metadata_path)
    df_root = df[df['type'] == 'root'].copy()
    
    # We write it to a temporary csv file
    temp_metadata_path = generated_dir / "temp_collapsed_metadata.csv"
    df_root.to_csv(temp_metadata_path, index=False)

    # 5. Process parameters to save .jconn file
    jconn_path = generated_dir / "collapsed_sim.jconn"
    process_params(
        path_to_full=str(collapsed_gml_path),
        type_groups=type_groups,
        directedness=directedness,
        path_to_save=str(jconn_path),
        path_to_metadata=str(temp_metadata_path),
        id_column='neuron_id',
        soma_criteria=lambda metadata: metadata['type'] == 'root'
    )

    # 6. Run simulation and save plot
    root_ctx = load_context(str(jconn_path))
    foo = Syn2Comp_static_paramsPoint(root_ctx)
    sim = DefaultExplicitSim(foo)
    sol = sim.solve(0, 100, num=200)

    # Plot results
    img_path = generated_dir / "collapsed_sim_result.png"
    plt.figure()
    plt.plot(sol.ts, sol.ys['V'])
    plt.title("Simulation with Collapsed Single-Compartment Neurons")
    plt.xlabel("Time")
    plt.ylabel("Voltage (V)")
    plt.savefig(str(img_path))
    plt.clf()
    assert img_path.exists()

