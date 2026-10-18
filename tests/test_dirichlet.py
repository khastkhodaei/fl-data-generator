import numpy as np
import pytest

from fl_data_generator import dirichlet_partition

_RNG = np.random.default_rng(0)


def test_dirichlet_partition_all_indices_allocated():
    labels = _RNG.integers(0, 5, size=200)
    num_clients = 8
    client_dict = dirichlet_partition(labels, num_clients, alpha=0.5, seed=42)

    assert len(client_dict) == num_clients

    all_indices = []
    for indices in client_dict.values():
        all_indices.extend(indices)

    assert len(all_indices) == len(labels)
    assert set(all_indices) == set(range(len(labels)))


def test_dirichlet_partition_reproducibility():
    labels = np.repeat(np.arange(4), 25)
    split_a = dirichlet_partition(labels, 4, alpha=0.5, seed=123)
    split_b = dirichlet_partition(labels, 4, alpha=0.5, seed=123)
    split_c = dirichlet_partition(labels, 4, alpha=0.5, seed=456)

    for cid in range(4):
        np.testing.assert_array_equal(split_a[cid], split_b[cid])

    different = any(
        not np.array_equal(split_a[cid], split_c[cid]) for cid in range(4)
    )
    assert different


def test_dirichlet_partition_skew_levels():
    labels = np.repeat(np.arange(5), 200)
    num_clients = 5

    skewed = dirichlet_partition(labels, num_clients, alpha=0.01, seed=42)
    uniform = dirichlet_partition(labels, num_clients, alpha=100.0, seed=42)

    # In extreme skew (alpha=0.01), some clients will have 0 samples of certain classes
    skewed_zeros = 0
    for indices in skewed.values():
        client_labels = labels[indices]
        for c in range(5):
            if np.sum(client_labels == c) == 0:
                skewed_zeros += 1

    # In uniform distribution (alpha=100.0), all clients should have samples from all classes
    uniform_zeros = 0
    for indices in uniform.values():
        client_labels = labels[indices]
        for c in range(5):
            if np.sum(client_labels == c) == 0:
                uniform_zeros += 1

    assert skewed_zeros > 0
    assert uniform_zeros == 0


def test_dirichlet_partition_vector_alpha():
    labels = _RNG.integers(0, 3, size=90)
    alpha_vec = [0.1, 0.5, 1.0]
    client_dict = dirichlet_partition(labels, num_clients=3, alpha=alpha_vec, seed=42)

    assert len(client_dict) == 3
    total_assigned = sum(len(idx) for idx in client_dict.values())
    assert total_assigned == len(labels)


def test_dirichlet_partition_invalid_inputs():
    labels = [0, 1, 1, 0]

    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=0)

    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=2, alpha=0)

    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=2, alpha=-0.5)

    with pytest.raises(ValueError):
        dirichlet_partition([], num_clients=2)

    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=2, alpha=[0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    "alpha_vec",
    [
        [0.5, 0.0, 0.5],   # zero value
        [0.5, -1.0, 0.5],  # negative value
    ],
)
def test_dirichlet_partition_vector_alpha_nonpositive(alpha_vec):
    """Vector alpha containing 0 or negative value must raise ValueError (covers line 52)."""
    labels = np.repeat(np.arange(3), 30)
    with pytest.raises(ValueError, match="All alpha values must be positive"):
        dirichlet_partition(labels, num_clients=3, alpha=alpha_vec)
