import numpy as np
import pytest

from fl_data_generator import hybrid_partition


def test_hybrid_partition_index_coverage():
    rng = np.random.default_rng(300)
    labels = rng.integers(0, 5, size=400)
    num_clients = 6
    client_dict = hybrid_partition(labels, num_clients=num_clients, alpha=0.5, beta=0.5, seed=42)

    assert len(client_dict) == num_clients

    all_indices = []
    for cid, indices in client_dict.items():
        assert len(indices) >= 1
        all_indices.extend(indices)

    assert len(all_indices) == len(labels)
    assert set(all_indices) == set(range(len(labels)))


def test_hybrid_partition_quantity_skew_effect():
    rng = np.random.default_rng(301)
    labels = rng.integers(0, 4, size=600)
    num_clients = 8

    skewed_sizes = [
        len(idx)
        for idx in hybrid_partition(labels, num_clients, alpha=1.0, beta=0.1, seed=42).values()
    ]
    uniform_sizes = [
        len(idx)
        for idx in hybrid_partition(labels, num_clients, alpha=1.0, beta=100.0, seed=42).values()
    ]

    assert np.std(skewed_sizes) > np.std(uniform_sizes)
    assert (max(skewed_sizes) - min(skewed_sizes)) > (max(uniform_sizes) - min(uniform_sizes))


def test_hybrid_partition_label_skew_effect():
    labels = np.repeat(np.arange(5), 100)
    num_clients = 5

    # With extreme label skew (alpha=0.01), some clients should have 0 samples of certain classes
    skewed = hybrid_partition(labels, num_clients, alpha=0.01, beta=10.0, seed=42)
    uniform = hybrid_partition(labels, num_clients, alpha=100.0, beta=10.0, seed=42)

    skewed_zeros = 0
    for indices in skewed.values():
        client_labels = labels[indices]
        for c in range(5):
            if np.sum(client_labels == c) == 0:
                skewed_zeros += 1

    uniform_zeros = 0
    for indices in uniform.values():
        client_labels = labels[indices]
        for c in range(5):
            if np.sum(client_labels == c) == 0:
                uniform_zeros += 1

    assert skewed_zeros > 0
    assert uniform_zeros == 0


def test_hybrid_partition_reproducibility():
    rng = np.random.default_rng(302)
    labels = rng.integers(0, 4, size=150)
    split_a = hybrid_partition(labels, 4, alpha=0.5, beta=0.5, seed=123)
    split_b = hybrid_partition(labels, 4, alpha=0.5, beta=0.5, seed=123)
    split_c = hybrid_partition(labels, 4, alpha=0.5, beta=0.5, seed=999)

    for cid in range(4):
        np.testing.assert_array_equal(split_a[cid], split_b[cid])

    different = any(
        not np.array_equal(split_a[cid], split_c[cid]) for cid in range(4)
    )
    assert different


def test_hybrid_partition_invalid_inputs():
    labels = [0, 1, 1, 0]

    with pytest.raises(ValueError):
        hybrid_partition(labels, num_clients=0)

    with pytest.raises(ValueError):
        hybrid_partition(labels, num_clients=2, alpha=0)

    with pytest.raises(ValueError):
        hybrid_partition(labels, num_clients=2, beta=-0.1)

    with pytest.raises(ValueError):
        hybrid_partition([], num_clients=2)

    with pytest.raises(ValueError):
        hybrid_partition(labels, num_clients=10)


def test_hybrid_partition_2d_labels_raises():
    labels = np.array([[0], [1], [2]])
    with pytest.raises(
        ValueError,
        match=r"labels must be 1-dimensional; consider using labels\.ravel\(\)",
    ):
        hybrid_partition(labels, num_clients=2)


def test_hybrid_partition_float_labels_raises():
    labels = np.array([0.5, 1.2, 3.4])
    with pytest.raises(ValueError, match=r"labels must be integer-valued"):
        hybrid_partition(labels, num_clients=2)


@pytest.mark.parametrize("invalid_alpha", [[0.5, 0.5], (0.5, 0.5), np.array([0.5, 0.5])])
def test_hybrid_partition_sequence_alpha_raises_value_error(invalid_alpha):
    labels = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="alpha must be a scalar"):
        hybrid_partition(labels, num_clients=2, alpha=invalid_alpha)


@pytest.mark.parametrize("invalid_beta", [[0.5, 0.5], (0.5, 0.5), np.array([0.5, 0.5])])
def test_hybrid_partition_sequence_beta_raises_value_error(invalid_beta):
    labels = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="beta must be a scalar"):
        hybrid_partition(labels, num_clients=2, beta=invalid_beta)


