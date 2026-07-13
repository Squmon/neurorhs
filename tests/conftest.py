from pathlib import Path
import sys

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path('./')

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest

from neurorhs.preprocessing.graph_to_arrays import (
    load_jax_context,
    process_graph_to_core_arrays,
    save_jax_arrays,
)
from neurorhs.preprocessing.preprocess import process_params

@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def graph_path(project_root):
    return project_root / "data" / "gml" / "20n.gml"


@pytest.fixture
def metadata_path(project_root):
    return project_root / "data" / "metadata" / "20n_nodes_metadata.csv"


@pytest.fixture
def generated_dir(project_root):
    path = project_root / "data" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def type_groups():
    return {"H": ["root", "soma", "branch", "slab", "end"], "S": ["connector"]}


@pytest.fixture
def directedness():
    return {"H": {"H": False, "S": True}, "S": {"H": True, "S": True}}


@pytest.fixture
def graph(graph_path):
    return nx.read_gml(graph_path)

@pytest.fixture
def context_path(project_root):
    project_root / 'data' / 'generated' / "test_preprocess_output.npz"