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
    },
    get_ampa_kainate_2state_syn: {
        'r1': 1100.0,
        'r2': 190.0,
        'g': 1.0,
        'E': 0.0,
        'P_defaut': {
            'C': 1.0,
            'O': 0.0,
        }
    },
    get_ampa_kainate_6state_syn: {
        'r1': 2e4,
        'r2': 1300.0,
        'r3': 1e4,
        'r4': 2600.0,
        'r5': 900.0,
        'r6': 500.0,
        'r7': 1e4,
        'r8': 0.2,
        'r9': 2.0,
        'r10': 0.1,
        'g': 1.0,
        'E': 0.0,
        'P_defaut': {
            'C': 1.0,
            'C1': 0.0,
            'C2': 0.0,
            'O': 0.0,
            'D1': 0.0,
            'D2': 0.0,
        }
    },
    # get_nmda_2state_syn: {
    #     'r1': 72.0,
    #     'r2': 6.6,
    #     'Mg': 1.0,
    #     'g': 1.0,
    #     'E': 0.0,
    #     'P_defaut': {
    #         'C': 1.0,
    #         'O': 0.0,
    #     }
    # },
    get_gaba_a_2state_syn: {
        'r1': 530.0,
        'r2': 180.0,
        'g': 1.0,
        'E': -80.0,
        'P_defaut': {
            'C': 1.0,
            'O': 0.0,
        }
    },
    get_gaba_b_syn: {
        'r1': 16.0,
        'r2': 4.7,
        'n': 4.0,
        'Kd': 100.0,
        'g': 1.0,
        'E': -95.0,
        'P_defaut': {
            'C': 1.0,
            'O': 0.0,
        }
    }
}