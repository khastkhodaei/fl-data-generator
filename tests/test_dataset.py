import numpy as np
import pytest

from fl_data_generator import FederatedDataset

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def test_federated_dataset_custom_arrays_iid():
    rng = np.random.default_rng(200)
    X = rng.standard_normal((200, 10))
    y = rng.integers(0, 4, size=200)

    fed_data = FederatedDataset(
        dataset=(X, y),
        num_clients=5,
        partition="iid",
        seed=42,
    )

    assert len(fed_data) == 200
    assert len(fed_data.client_dict) == 5
    assert sum(fed_data.client_sizes.values()) == 200

    # Test client dataset
    client_0_ds = fed_data.get_client_dataset(0)
    assert len(client_0_ds) == fed_data.client_sizes[0]
    sample_x, sample_y = client_0_ds[0]
    assert sample_x.shape == (10,)

    # Test indices
    idx_0 = fed_data.get_client_indices(0)
    assert len(idx_0) == fed_data.client_sizes[0]

    if _HAS_TORCH:
        loader = fed_data.get_client_dataloader(0, batch_size=8)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] <= 8
        assert batch_y.shape[0] <= 8


@pytest.mark.parametrize("strategy", ["dirichlet", "pathological", "quantity", "hybrid"])
def test_federated_dataset_partition_strategies(strategy):
    rng = np.random.default_rng(201)
    X = rng.standard_normal((300, 5))
    y = rng.integers(0, 6, size=300)

    fed_data = FederatedDataset(
        dataset=(X, y),
        num_clients=6,
        partition=strategy,
        alpha=0.5,
        beta=0.5,
        classes_per_client=2,
        seed=42,
    )

    assert len(fed_data) == 300
    assert len(fed_data.client_dict) == 6
    assert sum(fed_data.client_sizes.values()) == 300


def test_federated_dataset_transforms():
    X = np.ones((50, 4))
    y = np.zeros(50, dtype=int)

    fed_data = FederatedDataset(
        dataset=(X, y),
        num_clients=2,
        partition="iid",
        transform=lambda x: x * 5.0,
        target_transform=lambda t: t + 1,
        seed=42,
    )

    ds_0 = fed_data.get_client_dataset(0)
    sample_x, sample_y = ds_0[0]
    np.testing.assert_array_equal(sample_x, np.ones(4) * 5.0)
    assert sample_y == 1


def test_federated_dataset_repr_and_properties():
    rng = np.random.default_rng(202)
    X = rng.standard_normal((80, 2))
    y = rng.integers(0, 2, size=80)

    fed_data = FederatedDataset((X, y), num_clients=4, partition="iid", seed=42)
    repr_str = repr(fed_data)
    assert "num_clients=4" in repr_str
    assert "partition='iid'" in repr_str
    assert "total_samples=80" in repr_str


def test_federated_dataset_invalid_inputs():
    rng = np.random.default_rng(203)
    X = rng.standard_normal((20, 2))
    y = rng.integers(0, 2, size=20)

    with pytest.raises(ValueError, match="Unsupported partition"):
        FederatedDataset((X, y), num_clients=2, partition="unknown")

    with pytest.raises(ValueError, match="num_clients must be positive"):
        FederatedDataset((X, y), num_clients=0)

    with pytest.raises(ValueError, match="Mismatch between features"):
        FederatedDataset((X, y[:10]), num_clients=2)

    fed_data = FederatedDataset((X, y), num_clients=2)
    with pytest.raises(IndexError):
        fed_data.get_client_indices(5)

    with pytest.raises(ValueError, match="Unknown built-in dataset"):
        FederatedDataset("non_existent_dataset_name", num_clients=2)
