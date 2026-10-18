# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-10-18

### Fixed
- Fixed `TypeError: len() of unsized object` when passing NumPy scalars (e.g. `np.float32`, 0-d arrays) as `alpha` in `dirichlet_partition` or `beta` in `quantity_skew_partition`.
- Hardened `validate_labels` to explicitly reject complex numbers (`complex64`, `complex128`).
- Prevented silent integer overflow in `validate_labels` for float inputs exceeding `int64` bounds.

## [0.1.1] - 2026-10-18

### Fixed
- Fixed probe imports of `torch` and `torchvision` in `dataset.py` with `# noqa: F401` to ensure strict linter passing.
- Enabled strict CI gating by removing `continue-on-error` from linter workflow.
- Corrected inaccurate 0.1.0 release notes: the function is `quantity_skew_partition` (not `quantity_partition`), the parameter is `classes_per_client` (not `shards_per_client`), and `SVHN` is not a supported built-in dataset.

### Changed
- Migrated project metadata to modern PEP 639 specification (`license = "MIT"` and `setuptools>=77`).
- Added optional extras `torch` (`["torch", "torchvision"]`) and `all` (`["fl-data-generator[torch,viz]"]`) to `pyproject.toml`.
- Added `.gitattributes` to standardize LF line endings across platforms.

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

[Unreleased]: https://github.com/khastkhodaei/fl-data-generator/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/khastkhodaei/fl-data-generator/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/khastkhodaei/fl-data-generator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/khastkhodaei/fl-data-generator/releases/tag/v0.1.0
