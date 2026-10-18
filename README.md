# fl-data-generator

[![PyPI](https://img.shields.io/pypi/v/fl-data-generator.svg)](https://pypi.org/project/fl-data-generator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/khastkhodaei/fl-data-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/khastkhodaei/fl-data-generator/actions)

A lightweight, framework-agnostic Python library for **IID** and **Non-IID** data partitioning in **Federated Learning (FL)**.

---

## Overview

In Federated Learning research and benchmarks, simulating realistic data heterogeneity across distributed clients is essential. **fl-data-generator** simplifies partitioning arbitrary datasets into realistic client distributions without tying you to any specific deep learning framework.

Whether you are using PyTorch, TensorFlow, JAX, or federated frameworks like Flower, FedML, or PySyft, `fl-data-generator` provides clean sample indices and dataset wrappers for your clients.

---

## Features

- **IID Partitioning**: Uniform random sample allocation across all clients.
- **Dirichlet Partitioning**: Continuous label distribution skew ($\text{Dir}(\alpha)$) over class proportions, with client minimum size constraints (`min_size`).
- **Pathological Non-IID Partitioning**: Allocation of a fixed number of classes ($k$ classes per client, following the NIID-Bench benchmark scheme).
- **Quantity Skew Partitioning**: Volume heterogeneity where client dataset sizes vary according to a Dirichlet distribution ($\text{Dir}(\beta)$).
- **Hybrid Partitioning (Experimental)**: Sequential joint label skew and quantity skew allocation combining Dirichlet $\alpha$ and $\beta$.
- **FederatedDataset Wrapper**: High-level wrapper supporting built-in computer vision datasets (`mnist`, `cifar10`, `cifar100`, `fashion_mnist`) as well as custom `(X, y)` tuples and PyTorch `Dataset` instances.

---

## Installation

### From PyPI (Recommended)

```bash
pip install fl-data-generator
```

### With Optional Extras

```bash
# PyTorch and torchvision integration
pip install "fl-data-generator[torch]"

# All optional dependencies (torch + visualization)
pip install "fl-data-generator[all]"

# Development dependencies (pytest, hypothesis, ruff, etc.)
pip install "fl-data-generator[dev]"

# Visualization support (matplotlib)
pip install "fl-data-generator[viz]"
```

### From Source

```bash
git clone https://github.com/khastkhodaei/fl-data-generator.git
cd fl-data-generator
pip install -e .
```

---

## Runnable Examples

### 1. IID Partitioning

Uniformly partitions sample indices across clients.

```python
import numpy as np
from fl_data_generator import iid_partition

labels = np.array([0, 1] * 50)  # 100 samples
client_dict = iid_partition(labels, num_clients=4, seed=42)

print("Client sample counts:", [len(indices) for indices in client_dict.values()])
# Real Output: Client sample counts: [25, 25, 25, 25]
```

### 2. Dirichlet Label Skew Partitioning

Simulates non-IID label skew parameterized by concentration parameter `alpha`. Smaller $\alpha$ generates severe distribution skew, while larger $\alpha$ approaches uniform class distributions.

```python
import numpy as np
from fl_data_generator import dirichlet_partition

labels = np.repeat(np.arange(4), 50)  # 200 samples across 4 classes
client_dict = dirichlet_partition(labels, num_clients=3, alpha=0.5, seed=42)

print("Client sample counts:", [len(indices) for indices in client_dict.values()])
# Real Output: Client sample counts: [114, 42, 44]
```

### 3. Pathological Non-IID Partitioning ($k$ Classes per Client)

Partitions data such that each client receives samples from exactly $k$ distinct classes (NIID-Bench #C=k scheme).

```python
import numpy as np
from fl_data_generator import pathological_partition

labels = np.repeat(np.arange(4), 50)  # 200 samples across 4 classes
client_dict = pathological_partition(labels, num_clients=4, classes_per_client=2, seed=42)

client_0_labels = labels[client_dict[0]]
print("Client 0 distinct classes:", np.unique(client_0_labels).tolist())
# Real Output: Client 0 distinct classes: [1, 2]
```

### 4. Quantity Skew Partitioning (Volume Heterogeneity)

Allocates varying numbers of samples to clients according to a Dirichlet distribution parameterized by concentration parameter `beta`.

```python
import numpy as np
from fl_data_generator import quantity_skew_partition

labels = np.arange(200)  # 200 samples
client_dict = quantity_skew_partition(labels, num_clients=4, beta=0.5, seed=42)

print("Client sample counts:", [len(indices) for indices in client_dict.values()])
# Real Output: Client sample counts: [52, 72, 16, 60]
```

### 5. Hybrid Partitioning (Experimental)

Combines quantity skew ($\beta$) and label skew ($\alpha$) via sequential allocation.

```python
import numpy as np
from fl_data_generator import hybrid_partition

labels = np.repeat(np.arange(4), 50)  # 200 samples across 4 classes
client_dict = hybrid_partition(labels, num_clients=4, alpha=0.5, beta=0.5, seed=42)

print("Client sample counts:", [len(indices) for indices in client_dict.values()])
# Real Output: Client sample counts: [56, 89, 2, 53]
```

### 6. FederatedDataset Interface

High-level interface for creating client-specific datasets and PyTorch DataLoaders.

```python
import numpy as np
from fl_data_generator import FederatedDataset

X = np.random.default_rng(42).standard_normal((100, 5))
y = np.array([0, 1] * 50)

fed_data = FederatedDataset((X, y), num_clients=4, partition="iid", seed=42)
client_0_ds = fed_data.get_client_dataset(0)

print(f"Client 0 sample count: {len(client_0_ds)}")
features, label = client_0_ds[0]
print(f"Feature shape: {features.shape}, Label: {label}")
# Real Output:
# Client 0 sample count: 25
# Feature shape: (5,), Label: 1
```

---

## Roadmap

The following experimental features and partition strategies are planned for future releases:
- **Feature Skew (Covariate Shift)**: Partitioning based on continuous feature space clusters or synthetic domain perturbations.
- **Power-Law Quantity Skew**: Heavy-tailed sample volume distribution across clients.
- **Asymmetric Class Topologies**: Custom directed class sharing graphs.

---

## Project Structure

```text
fl-data-generator/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI matrix workflow
├── .gitattributes               # Line ending normalization
├── src/
│   └── fl_data_generator/       # Core library package
│       ├── __init__.py          # Public API exports
│       ├── _validation.py       # Input validation utilities
│       ├── dataset.py           # FederatedDataset interface & torchvision loaders
│       ├── dirichlet.py         # Dirichlet label skew partitioner
│       ├── hybrid.py            # Hybrid label & quantity skew partitioner
│       ├── iid.py               # Uniform IID partitioner
│       ├── pathological.py      # Pathological k-class partitioner
│       └── quantity.py          # Quantity skew partitioner
├── tests/                       # Unit, integration, and property test suite
│   ├── test_dataset.py
│   ├── test_dirichlet.py
│   ├── test_hybrid.py
│   ├── test_iid.py
│   ├── test_pathological.py
│   ├── test_properties.py
│   ├── test_quantity.py
│   ├── test_readme.py
│   └── test_validation.py
├── CHANGELOG.md                 # Release changelog (Keep a Changelog)
├── CITATION.cff                 # Citation metadata
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
└── pyproject.toml               # Package build configuration & Ruff settings
```

---

## Citation

If you use `fl-data-generator` in your research or benchmarks, please cite it using:

```bibtex
@software{khastkhodaei2026fldatagenerator,
  author = {Khastkhodaei, Mostafa},
  title = {fl-data-generator: A Lightweight Federated Learning Data Partitioning Library},
  url = {https://github.com/khastkhodaei/fl-data-generator},
  version = {0.1.1},
  year = {2026}
}
```

You can also use the citation metadata provided in [CITATION.cff](CITATION.cff).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
