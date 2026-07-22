from neurorhs.neurosci import *

PlaceHolderParams = {
    get_cabble_pipeline: {
        'ro': 100.0
    },
    get_Na_channel_pipeline: {
        'm': 0.0220,
        'h': 0.9840,
        'gNa': 120.0,
        'eNa': 50.0,
    },
    get_K_channel_pipeline: {
        'n': 0.0773,
        'gK': 36.0,
        'eK': -77.0,
    },
    get_leak_channel_pipeline: {
        'gLeak': 0.3,
        'eLeak': -54.4,
    },
    get_stub_synapse_pipeline: {
        'V': 0.0,
        'weight': 0.5
    },
    get_dummy_delay_synapse_pipeline: {
        'z': 1.0,
        'tau_d': 1.0,
        'tau_r': 0.1,
        'weight': 70.0,
        'slope': 1.0,
        'bias': 0,
    },

    get_kinetic_synapce_pipeline: {
        'E': 0.0,
        'L_max': 2.84,
        'V_p': 2,
        'K_p': 5,
    },
    get_component2_syn: {
        'r1': 0.1,
        'r2': 0.01,
        'g': 2.0,
        'P_defaut': {
            'C': 1.0,
            'O': 0.0,
        }
    }
}