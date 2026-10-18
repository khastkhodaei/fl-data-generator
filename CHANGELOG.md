# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-18

### Added
- Core partitioning strategies:
  - `iid_partition`: Independent and Identically Distributed partitioning across clients.
  - `dirichlet_partition`: Dirichlet distribution-based label skew partitioning with Dirichlet resampling, validation of `num_clients <= num_samples`, and `min_size` / `max_retries` constraints.
  - `pathological_partition`: Non-IID label skew assigning a fixed number of classes per client with strict validation of `shards_per_client` and client capacity.
  - `quantity_partition`: Dirichlet distribution-based quantity skew for uneven sample size allocations.
  - `hybrid_partition`: Combined label and quantity skew partitioning (experimental status, with scalar `alpha`/`beta` validation).
- `FederatedDataset` wrapper supporting PyTorch dataset compatibility, lazy loading, and built-in datasets (`MNIST`, `CIFAR10`, `CIFAR100`, `FashionMNIST`, `SVHN`) with pure Python / NumPy fallback (`_NonTorchSubset`) when PyTorch is unavailable.
- Comprehensive test suite covering edge cases, property-based tests via Hypothesis, and runnable documentation tests.
- CI/CD workflow testing Python matrix 3.9 through 3.13 and lowest-direct dependency resolution with `uv`.

[Unreleased]: https://github.com/khastkhodaei/fl-data-generator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/khastkhodaei/fl-data-generator/releases/tag/v0.1.0
