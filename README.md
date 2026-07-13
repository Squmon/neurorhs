# neurorhs

neurorhs is a small Python project for working with graph-based neuron morphology data and running simple JAX-powered simulation pipelines. The package provides utilities to preprocess NetworkX graphs, convert them into compact arrays for downstream modeling, and run channel/synapse update pipelines.

## Features

- Load and process graph data from GML files
- Map node metadata onto graph nodes and save a JAX-friendly context
- Build reusable pipelines for:
  - sodium channels
  - potassium channels
  - leak channels
  - stub synapses
  - cable propagation
- Support round-trip serialization of processed graph context through NPZ files

## Project structure

- `neurorhs/` – core package code
  - `neurosci.py` – channel and synapse update pipelines
  - `utils.py` – helper utilities for nested dictionaries
  - `preprocessing/` – graph preprocessing and array export logic
  - `configs/` – default configuration helpers
- `tests/` – pytest-based regression tests
- `data/` – example graph and metadata files

## Installation

Install the package and its dependencies with:

```bash
pip install -r requirements.txt
```

For development tools and tests:

```bash
pip install -r dev-requirements.txt
```

## Usage

### Preprocess graph data

The preprocessing workflow reads a graph file and metadata file, then writes a serialized context that can be consumed by the simulation code:

```python
from neurorhs.preprocessing.preprocess import process_params

process_params(
    "data/gml/20n.gml",
    {"H": ["root", "soma", "branch", "slab", "end"], "S": ["connector"]},
    {"H": {"H": False, "S": True}, "S": {"H": True, "S": True}},
    "data/generated/test_preprocess_output.npz",
    "data/metadata/20n_nodes_metadata.csv",
)
```

### Run the example simulation

The package includes a simulation example built around the generated NPZ context. Tests and example flows use the generated artifacts from the preprocessing step.

## Testing

Run the test suite with:

```bash
pytest -q
```

## Notes

This repository is still fairly lightweight and is primarily intended as an experimental research-oriented codebase for graph-driven neuron modeling.
