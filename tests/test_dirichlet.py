import numpy as np
import pytest

from fl_data_generator import dirichlet_partition


def test_dirichlet_partition_all_indices_allocated():
    rng = np.random.default_rng(100)
    labels = rng.integers(0, 5, size=200)
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
    rng = np.random.default_rng(101)
    labels = rng.integers(0, 3, size=90)
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


def test_dirichlet_partition_2d_labels_raises():
    labels = np.array([[0], [1], [2]])
    with pytest.raises(
        ValueError,
        match=r"labels must be 1-dimensional; consider using labels\.ravel\(\)",
    ):
        dirichlet_partition(labels, num_clients=2)


def test_dirichlet_partition_float_labels_raises():
    labels = np.array([0.5, 1.2, 3.4])
    with pytest.raises(ValueError, match=r"labels must be integer-valued"):
        dirichlet_partition(labels, num_clients=2)


def test_dirichlet_partition_num_clients_exceeds_samples_raises():
    labels = [0, 1, 2, 0]
    num_clients = 10
    with pytest.raises(
        ValueError,
        match=r"^num_clients \(10\) cannot exceed number of samples \(4\)$",
    ):
        dirichlet_partition(labels, num_clients=num_clients)


@pytest.mark.parametrize(
    "invalid_min_size",
    [
        -1,
        1.5,
        True,
        False,
        "1",
    ],
)
def test_dirichlet_partition_invalid_min_size_type_or_negative(invalid_min_size):
    labels = [0, 1, 0, 1]
    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=2, min_size=invalid_min_size)


def test_dirichlet_partition_min_size_times_num_clients_exceeds_samples_raises():
    labels = [0, 1, 0, 1, 0]
    with pytest.raises(ValueError):
        dirichlet_partition(labels, num_clients=3, min_size=2)


@pytest.mark.parametrize("invalid_max_retries", [0, -1, 1.5, True, False, "10"])
def test_dirichlet_partition_invalid_max_retries(invalid_max_retries):
    labels = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="max_retries must be a positive integer"):
        dirichlet_partition(labels, num_clients=2, max_retries=invalid_max_retries)


def test_dirichlet_partition_numpy_scalar_min_size_and_max_retries():
    labels = np.repeat(np.arange(3), 20)
    res = dirichlet_partition(
        labels, num_clients=3, min_size=np.int64(2), max_retries=np.int64(500), seed=42
    )
    assert len(res) == 3



def test_dirichlet_partition_max_retries_exceeded_raises_runtime_error():
    labels = np.array([0, 1] * 5)
    with pytest.raises(
        RuntimeError,
        match=r"(?=.*alpha)(?=.*min_size)",
    ):
        dirichlet_partition(labels, num_clients=5, alpha=1e-5, min_size=2, max_retries=1, seed=0)


def test_dirichlet_partition_regression_seeds_0_to_19():
    labels = np.repeat(np.arange(10), 50)
    for seed in range(20):
        split_default = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=seed)
        split_min_zero = dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=seed, min_size=0)
        for cid in range(10):
            np.testing.assert_array_equal(split_default[cid], split_min_zero[cid])


def test_dirichlet_partition_min_size_zero_allows_empty_client():
    labels = np.array([0, 1] * 10)
    split = dirichlet_partition(labels, num_clients=5, alpha=0.1, seed=1, min_size=0)
    sizes = [len(split[cid]) for cid in range(5)]
    assert any(s == 0 for s in sizes)
    assert sum(sizes) == len(labels)


def test_dirichlet_partition_default_min_size_eliminates_empty_clients():
    labels = np.array([0, 1] * 10)
    split = dirichlet_partition(labels, num_clients=5, alpha=0.1, seed=1)
    sizes = [len(split[cid]) for cid in range(5)]
    assert all(s >= 1 for s in sizes)
    assert sum(sizes) == len(labels)


def test_dirichlet_partition_numpy_scalar_alpha():
    labels = np.array([0, 1] * 10)
    res1 = dirichlet_partition(labels, 2, alpha=np.float32(0.5), seed=42)
    assert len(res1) == 2
    res2 = dirichlet_partition(labels, 2, alpha=np.array(0.5), seed=42)
    assert len(res2) == 2


def test_dirichlet_partition_bool_alpha():
    labels = [0, 1] * 5
    with pytest.raises(ValueError, match="alpha cannot be a boolean"):
        dirichlet_partition(labels, 2, alpha=True)




