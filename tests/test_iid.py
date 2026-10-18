import numpy as np
import pytest

from fl_data_generator import iid_partition


def test_iid_partition_sample_count():
    rng = np.random.default_rng(400)
    labels = rng.integers(0, 10, size=100)
    num_clients = 10
    client_dict = iid_partition(labels, num_clients, seed=42)

    assert len(client_dict) == num_clients

    all_indices = []
    for cid, indices in client_dict.items():
        assert len(indices) == 10
        all_indices.extend(indices)

    assert len(all_indices) == len(labels)
    assert set(all_indices) == set(range(len(labels)))


def test_iid_partition_uneven_samples():
    labels = list(range(103))
    num_clients = 5
    client_dict = iid_partition(labels, num_clients, seed=42)

    lengths = [len(idx) for idx in client_dict.values()]
    assert sum(lengths) == 103
    assert max(lengths) - min(lengths) <= 1


def test_iid_partition_reproducibility():
    labels = np.arange(50)
    split_a = iid_partition(labels, 4, seed=123)
    split_b = iid_partition(labels, 4, seed=123)
    split_c = iid_partition(labels, 4, seed=999)

    for cid in range(4):
        np.testing.assert_array_equal(split_a[cid], split_b[cid])

    different = any(
        not np.array_equal(split_a[cid], split_c[cid]) for cid in range(4)
    )
    assert different


def test_iid_partition_invalid_inputs():
    labels = [0, 1, 2]
    with pytest.raises(ValueError):
        iid_partition(labels, num_clients=0)

    with pytest.raises(ValueError):
        iid_partition(labels, num_clients=10)

    with pytest.raises(ValueError):
        iid_partition(labels, num_clients=2.5)

    with pytest.raises(ValueError):
        iid_partition(labels, num_clients=True)



def test_iid_partition_2d_labels_raises():
    labels = np.array([[0], [1], [2]])
    with pytest.raises(
        ValueError,
        match=r"labels must be 1-dimensional; consider using labels\.ravel\(\)",
    ):
        iid_partition(labels, num_clients=2)


def test_iid_partition_float_labels_raises():
    labels = np.array([0.5, 1.2, 3.4])
    with pytest.raises(ValueError, match=r"labels must be integer-valued"):
        iid_partition(labels, num_clients=2)

