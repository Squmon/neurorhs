from neurorhs.preprocessing.preprocess import process_params
from neurorhs.preprocessing.graph_to_arrays import (
    process_graph_to_core_arrays,
)
from neurorhs.io import load_context, save_context
import pytest
import numpy as np
import networkx as nx
import jax.numpy as jnp
import jax
from pathlib import Path
import sys

PROJECT_ROOT = Path('./')


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
    project_root / 'data' / 'generated' / "test_preprocess_output.jconn"
