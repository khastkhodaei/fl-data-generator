import numpy as np
import pytest

from fl_data_generator import pathological_partition

_RNG = np.random.default_rng(4)


def test_pathological_partition_classes_per_client():
    labels = np.repeat(np.arange(10), 100)
    num_clients = 10
    classes_per_client = 2

    client_dict = pathological_partition(
        labels, num_clients=num_clients, classes_per_client=classes_per_client, seed=42
    )

    assert len(client_dict) == num_clients

    for cid, indices in client_dict.items():
        client_labels = labels[indices]
        unique_classes = np.unique(client_labels)
        assert len(unique_classes) == classes_per_client


def test_pathological_partition_all_indices_covered():
    labels = _RNG.integers(0, 6, size=300)
    num_clients = 6
    classes_per_client = 2

    client_dict = pathological_partition(
        labels, num_clients=num_clients, classes_per_client=classes_per_client, seed=42
    )

    all_indices = []
    for indices in client_dict.values():
        all_indices.extend(indices)

    assert len(all_indices) == len(labels)
    assert set(all_indices) == set(range(len(labels)))


def test_pathological_partition_reproducibility():
    labels = np.repeat(np.arange(5), 40)
    split_a = pathological_partition(labels, 5, classes_per_client=2, seed=123)
    split_b = pathological_partition(labels, 5, classes_per_client=2, seed=123)
    split_c = pathological_partition(labels, 5, classes_per_client=2, seed=999)

    for cid in range(5):
        np.testing.assert_array_equal(split_a[cid], split_b[cid])

    different = any(
        not np.array_equal(split_a[cid], split_c[cid]) for cid in range(5)
    )
    assert different


def test_pathological_partition_single_class_per_client():
    labels = np.repeat(np.arange(4), 50)
    client_dict = pathological_partition(
        labels, num_clients=4, classes_per_client=1, seed=42
    )

    for indices in client_dict.values():
        assert len(np.unique(labels[indices])) == 1


def test_pathological_partition_invalid_inputs():
    labels = np.repeat(np.arange(4), 10)

    with pytest.raises(ValueError):
        pathological_partition(labels, num_clients=0)

    with pytest.raises(ValueError):
        pathological_partition(labels, num_clients=4, classes_per_client=0)

    with pytest.raises(ValueError):
        pathological_partition(labels, num_clients=4, classes_per_client=10)

    with pytest.raises(ValueError):
        # 2 clients * 1 class = 2 slots < 4 unique classes
        pathological_partition(labels, num_clients=2, classes_per_client=1)

    with pytest.raises(ValueError):
        pathological_partition([], num_clients=2)
