from unittest.mock import MagicMock
import numpy as np
import pytest

from fl_data_generator import FederatedDataset
from fl_data_generator.dataset import NumpyDataset


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
    assert sample_y in [0, 1, 2, 3]

    # Test indices
    idx_0 = fed_data.get_client_indices(0)
    assert len(idx_0) == fed_data.client_sizes[0]


def test_federated_dataset_dataloader_with_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(205)
    X = rng.standard_normal((100, 8))
    y = rng.integers(0, 2, size=100)

    fed_data = FederatedDataset((X, y), num_clients=4, partition="iid", seed=42)
    loader = fed_data.get_client_dataloader(0, batch_size=8, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[0] <= 8
    assert batch_y.shape[0] <= 8
    assert isinstance(batch_x, torch.Tensor)


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

    with pytest.raises(TypeError, match="dataset must be a string name"):
        FederatedDataset(12345, num_clients=2)


def test_federated_dataset_builtin_without_torchvision_raises(monkeypatch):
    monkeypatch.setattr("fl_data_generator.dataset._HAS_TORCHVISION", False)
    with pytest.raises(ImportError, match="torchvision is required to download and load"):
        FederatedDataset("mnist", num_clients=2)


def test_federated_dataset_builtin_mocked_torchvision_targets_numpy(monkeypatch):
    mock_tv_datasets = MagicMock()
    mock_instance = MagicMock()
    mock_targets = MagicMock()
    mock_targets.numpy.return_value = np.array([0, 1, 0, 1, 0, 1] * 10)
    mock_instance.targets = mock_targets
    mock_instance.__len__.return_value = 60
    mock_instance.__getitem__.side_effect = lambda idx: (np.zeros(5), int(idx % 2))

    mock_tv_datasets.MNIST.return_value = mock_instance

    monkeypatch.setattr("fl_data_generator.dataset._HAS_TORCHVISION", True)
    monkeypatch.setattr("fl_data_generator.dataset.tv_datasets", mock_tv_datasets)

    fed_data = FederatedDataset("mnist", num_clients=3, partition="iid", seed=42)
    assert len(fed_data) == 60
    assert len(fed_data.client_dict) == 3


def test_federated_dataset_builtin_mocked_torchvision_targets_list_and_fallback(monkeypatch):
    mock_tv_datasets = MagicMock()
    mock_instance_list = MagicMock()
    mock_instance_list.targets = [0, 1] * 20
    mock_instance_list.__len__.return_value = 40
    mock_instance_list.__getitem__.side_effect = lambda idx: (np.zeros(3), int(idx % 2))

    mock_tv_datasets.CIFAR10.return_value = mock_instance_list

    monkeypatch.setattr("fl_data_generator.dataset._HAS_TORCHVISION", True)
    monkeypatch.setattr("fl_data_generator.dataset.tv_datasets", mock_tv_datasets)

    fed_cifar = FederatedDataset("cifar10", num_clients=4, partition="iid", seed=42)
    assert len(fed_cifar) == 40

    # Also test when dataset instance has no targets attribute (falls back to __getitem__)
    mock_instance_no_targets = MagicMock()
    del mock_instance_no_targets.targets
    mock_instance_no_targets.__len__.return_value = 30
    mock_instance_no_targets.__getitem__.side_effect = lambda idx: (np.zeros(4), int(idx % 3))
    mock_tv_datasets.FashionMNIST.return_value = mock_instance_no_targets

    fed_fashion = FederatedDataset("fashion_mnist", num_clients=3, partition="iid", seed=42)
    assert len(fed_fashion) == 30


def test_federated_dataset_generic_dataset_objects():
    # Generic dataset with .targets (has .numpy())
    class TargetsNumpyDataset:
        def __init__(self):
            targets_mock = MagicMock()
            targets_mock.numpy.return_value = np.array([0, 1] * 20)
            self.targets = targets_mock

        def __len__(self):
            return 40

        def __getitem__(self, idx):
            return np.ones(3), idx % 2

    fed_data1 = FederatedDataset(TargetsNumpyDataset(), num_clients=2, seed=42)
    assert len(fed_data1) == 40

    # Generic dataset with .labels attribute
    class LabelsDataset:
        def __init__(self):
            self.labels = [0, 1, 2] * 10

        def __len__(self):
            return 30

        def __getitem__(self, idx):
            return np.ones(2), idx % 3

    fed_data2 = FederatedDataset(LabelsDataset(), num_clients=3, seed=42)
    assert len(fed_data2) == 30

    # Generic dataset without targets or labels (extract via __getitem__)
    class PlainDataset:
        def __len__(self):
            return 20

        def __getitem__(self, idx):
            return np.zeros(2), idx % 2

    fed_data3 = FederatedDataset(PlainDataset(), num_clients=2, seed=42)
    assert len(fed_data3) == 20


def test_federated_dataset_non_torch_fallback(monkeypatch):
    """Verify _NonTorchSubset behaves as expected when torch is unavailable."""
    monkeypatch.setattr("fl_data_generator.dataset._HAS_TORCH", False)
    monkeypatch.setattr("fl_data_generator.dataset.Subset", None)
    monkeypatch.setattr("fl_data_generator.dataset.DataLoader", None)

    X = np.arange(50).reshape(25, 2)
    y = np.array([0, 1] * 12 + [0])

    fed_data = FederatedDataset((X, y), num_clients=2, partition="iid", seed=42)

    # get_client_dataset should return _NonTorchSubset
    client_ds = fed_data.get_client_dataset(0)
    assert len(client_ds) == fed_data.client_sizes[0]
    sample_x, sample_y = client_ds[0]
    first_idx = fed_data.get_client_indices(0)[0]
    np.testing.assert_array_equal(sample_x, X[first_idx])
    assert sample_y == y[first_idx]

    # get_client_dataloader should raise ImportError
    with pytest.raises(ImportError, match="PyTorch is required to create a DataLoader"):
        fed_data.get_client_dataloader(0)


def test_numpy_dataset_direct():
    X = np.ones((10, 3))
    y = np.arange(10)

    ds = NumpyDataset(X, y)
    assert len(ds) == 10
    sample_x, sample_y = ds[3]
    np.testing.assert_array_equal(sample_x, np.ones(3))
    assert sample_y == 3


def test_federated_dataset_lazy_torch_import():
    from pathlib import Path
    import subprocess
    import sys

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    code = (
        f"import sys; sys.path.insert(0, {src_dir!r}); "
        "import fl_data_generator; "
        "assert 'torch' not in sys.modules, 'torch is unexpectedly loaded!'; "
        "assert 'torchvision' not in sys.modules, 'torchvision is unexpectedly loaded!'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Lazy import failed: {result.stderr}"


def test_numpy_dataset_torch_interop():
    torch = pytest.importorskip("torch")

    X = np.ones((10, 3))
    y = np.arange(10)
    ds = NumpyDataset(X, y)

    assert isinstance(ds, torch.utils.data.Dataset)


def test_federated_dataset_num_clients_types():
    X = np.ones((10, 2))
    y = np.zeros(10, dtype=int)

    with pytest.raises(ValueError):
        FederatedDataset((X, y), num_clients=2.5)

    with pytest.raises(ValueError):
        FederatedDataset((X, y), num_clients=True)
