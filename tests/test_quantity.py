import numpy as np
import pytest

from fl_data_generator import quantity_skew_partition


def test_quantity_skew_partition_all_indices_allocated():
    rng = np.random.default_rng(600)
    labels = rng.integers(0, 5, size=500)
    num_clients = 8
    client_dict = quantity_skew_partition(labels, num_clients=num_clients, beta=0.5, seed=42)

    assert len(client_dict) == num_clients

    all_indices = []
    for cid, indices in client_dict.items():
        assert len(indices) >= 1
        all_indices.extend(indices)

    assert len(all_indices) == len(labels)
    assert set(all_indices) == set(range(len(labels)))


def test_quantity_skew_partition_varying_sizes():
    labels = np.arange(1000)
    num_clients = 10

    skewed = quantity_skew_partition(labels, num_clients=num_clients, beta=0.1, seed=42)
    uniform = quantity_skew_partition(labels, num_clients=num_clients, beta=100.0, seed=42)

    skewed_sizes = [len(idx) for idx in skewed.values()]
    uniform_sizes = [len(idx) for idx in uniform.values()]

    # Severe skew should exhibit higher standard deviation than uniform
    assert np.std(skewed_sizes) > np.std(uniform_sizes)
    # Range of sample counts should be substantially wider in skewed setting
    assert (max(skewed_sizes) - min(skewed_sizes)) > (max(uniform_sizes) - min(uniform_sizes))


def test_quantity_skew_partition_reproducibility():
    labels = list(range(200))
    split_a = quantity_skew_partition(labels, num_clients=5, beta=0.5, seed=777)
    split_b = quantity_skew_partition(labels, num_clients=5, beta=0.5, seed=777)
    split_c = quantity_skew_partition(labels, num_clients=5, beta=0.5, seed=888)

    for cid in range(5):
        np.testing.assert_array_equal(split_a[cid], split_b[cid])

    different = any(
        not np.array_equal(split_a[cid], split_c[cid]) for cid in range(5)
    )
    assert different


def test_quantity_skew_partition_vector_beta():
    labels = list(range(100))
    beta_vec = [0.1, 0.5, 1.0, 2.0]
    client_dict = quantity_skew_partition(labels, num_clients=4, beta=beta_vec, seed=42)

    assert len(client_dict) == 4
    total_assigned = sum(len(idx) for idx in client_dict.values())
    assert total_assigned == len(labels)


def test_quantity_skew_partition_invalid_inputs():
    labels = [0, 1, 2, 3]

    with pytest.raises(ValueError):
        quantity_skew_partition(labels, num_clients=0)

    with pytest.raises(ValueError):
        quantity_skew_partition(labels, num_clients=2, beta=0)

    with pytest.raises(ValueError):
        quantity_skew_partition(labels, num_clients=2, beta=-1.0)

    with pytest.raises(ValueError):
        quantity_skew_partition([], num_clients=2)

    with pytest.raises(ValueError):
        quantity_skew_partition(labels, num_clients=10)

    with pytest.raises(ValueError):
        quantity_skew_partition(labels, num_clients=2, beta=[0.5, 0.5, 0.5])
