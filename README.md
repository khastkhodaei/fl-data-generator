# fl-data-generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A lightweight, framework-agnostic Python library for **IID** and **Non-IID** data partitioning in **Federated Learning (FL)**.

---

## Overview

In Federated Learning research and benchmarks, simulating realistic data heterogeneity across distributed clients is crucial. **fl-data-generator** simplifies partitioning arbitrary datasets into realistic client distributions without tying you to any specific deep learning framework.

Whether you are using PyTorch, TensorFlow, JAX, or federated frameworks like Flower, FedML, or PySyft, `fl-data-generator` provides clean indices and partitions for your clients.

## Key Features

- **Framework-Agnostic**: Operates directly on dataset labels, targets, or arrays (NumPy-compatible). Zero hard dependencies on deep learning frameworks.
- **Comprehensive Non-IID Partitioning**:
  - **IID (Independent and Identically Distributed)**: Uniform random partitioning across all clients.
  - **Dirichlet Distribution ($\text{Dir}(\alpha)$)**: Continuous label distribution skew parameterized by concentration $\alpha$.
  - **Pathological / Shard-Based Label Skew**: Restrict clients to a fixed number of classes (e.g., 2 classes per client).
  - **Quantity Skew (Volume Heterogeneity)**: Distribute varying dataset sizes across clients (e.g., via Dirichlet or power-law distributions).
  - **Feature Skew (Covariate Shift)**: Partition data based on input features or synthetic noise.
- **Lightweight & Fast**: Vectorized NumPy operations for lightning-fast partitioning of large datasets.
- **Reproducible**: Deterministic partitioning with explicit random seed control.

---

## Installation

### From Source

```bash
git clone https://github.com/khastkhodaei/fl-data-generator.git
cd fl-data-generator
pip install -e .
```

### With Optional Dependencies

```bash
# Development dependencies (pytest, ruff, etc.)
pip install -e ".[dev]"

# Visualization support (matplotlib)
pip install -e ".[viz]"
```

---

## Quick Start

```python
import numpy as np

# Example: Partitioning synthetic labels across 10 clients using Dirichlet distribution
# targets: array of class labels for your dataset
num_samples = 1000
num_classes = 10
targets = np.random.randint(0, num_classes, size=num_samples)

# Partition indices will be mapped per client
# client_dict[client_id] -> array of sample indices
```

---

## Project Structure

```text
fl-data-generator/
├── src/
│   └── fl_data_generator/   # Core library package
│       ├── __init__.py      # Package initialization
│       └── iid.py           # IID data partitioner
├── tests/                   # Test suite
│   └── test_iid.py
├── pyproject.toml           # Build configuration and dependency specification
├── LICENSE                  # MIT License
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
