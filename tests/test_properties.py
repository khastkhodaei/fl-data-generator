import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fl_data_generator import (
    dirichlet_partition,
    hybrid_partition,
    iid_partition,
    pathological_partition,
    quantity_skew_partition,
)


def assert_partition_invariants(client_dict, num_samples, pre_state, partition_fn, *args, seed=42, **kwargs):
    """Assert all 5 core invariants for a partitioner output:

    1. Global np.random state unchanged
    2. Dtype is int64
    3. Disjoint partitions
    4. Complete coverage
    5. Same seed -> same output
    """
    # 1. Global np.random state unchanged
    post_state = np.random.get_state()
    for s1, s2 in zip(pre_state, post_state):
        if isinstance(s1, np.ndarray):
            np.testing.assert_array_equal(s1, s2)
        else:
            assert s1 == s2

    # 2. Dtype is int64
    for cid, arr in client_dict.items():
        assert isinstance(arr, np.ndarray), f"Client {cid} output must be a numpy array"
        assert arr.dtype == np.int64, f"Client {cid} array dtype {arr.dtype} != int64"

    # 3. Disjoint partitions
    all_indices = []
    for arr in client_dict.values():
        all_indices.extend(arr.tolist())
    assert len(all_indices) == len(set(all_indices)), "Partitions are not disjoint across clients"

    # 4. Complete coverage
    assert len(all_indices) == num_samples, f"Total allocated {len(all_indices)} != {num_samples}"
    assert set(all_indices) == set(range(num_samples)), "Allocated indices do not match range(num_samples)"

    # 4b. No empty client produced (unless min_size=0 explicitly allowed)
    if kwargs.get("min_size", 1) != 0:
        assert all(len(arr) > 0 for arr in client_dict.values()), "Empty client produced"

    # 5. Same seed -> same output
    if seed is not None:
        second_dict = partition_fn(*args, seed=seed, **kwargs)
        assert len(client_dict) == len(second_dict)
        for cid in client_dict:
            np.testing.assert_array_equal(client_dict[cid], second_dict[cid])


@settings(max_examples=50, deadline=None)
@given(
    num_clients=st.integers(min_value=1, max_value=15),
    extra_samples=st.integers(min_value=0, max_value=150),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_iid_partition_properties(num_clients, extra_samples, seed):
    num_samples = num_clients + extra_samples
    labels = np.random.default_rng(seed).integers(0, 5, size=num_samples)

    pre_state = np.random.get_state()
    client_dict = iid_partition(labels, num_clients, seed=seed)
    assert_partition_invariants(client_dict, num_samples, pre_state, iid_partition, labels, num_clients, seed=seed)


@settings(max_examples=50, deadline=None)
@given(
    num_clients=st.integers(min_value=1, max_value=10),
    extra_samples=st.integers(min_value=0, max_value=100),
    alpha=st.floats(min_value=0.1, max_value=20.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_dirichlet_partition_properties(num_clients, extra_samples, alpha, seed):
    num_samples = num_clients + extra_samples
    labels = np.random.default_rng(seed).integers(0, 5, size=num_samples)

    pre_state = np.random.get_state()
    client_dict = dirichlet_partition(labels, num_clients, alpha=alpha, seed=seed, min_size=0)
    assert_partition_invariants(
        client_dict,
        num_samples,
        pre_state,
        dirichlet_partition,
        labels,
        num_clients,
        alpha=alpha,
        seed=seed,
        min_size=0,
    )


@pytest.mark.xfail(
    strict=True,
    reason="Dirichlet partitioner with default min_size=1 raises RuntimeError when classes have single samples because split_points truncate to 0",
)
@settings(max_examples=20, deadline=None)
@given(
    num_clients=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_dirichlet_partition_default_min_size_fails_on_single_sample_classes(num_clients, seed):
    # Minimal counterexample: labels where each class has only 1 sample
    labels = np.arange(num_clients)
    dirichlet_partition(labels, num_clients, alpha=1.0, seed=seed)


@settings(max_examples=50, deadline=None)
@given(
    num_clients=st.integers(min_value=1, max_value=10),
    extra_samples=st.integers(min_value=0, max_value=100),
    beta=st.floats(min_value=0.1, max_value=20.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_quantity_skew_partition_properties(num_clients, extra_samples, beta, seed):
    num_samples = num_clients + extra_samples
    labels = np.random.default_rng(seed).integers(0, 5, size=num_samples)

    pre_state = np.random.get_state()
    client_dict = quantity_skew_partition(labels, num_clients, beta=beta, seed=seed)
    assert_partition_invariants(
        client_dict,
        num_samples,
        pre_state,
        quantity_skew_partition,
        labels,
        num_clients,
        beta=beta,
        seed=seed,
    )


@settings(max_examples=50, deadline=None)
@given(
    num_clients=st.integers(min_value=1, max_value=10),
    extra_samples=st.integers(min_value=0, max_value=100),
    alpha=st.floats(min_value=0.1, max_value=20.0),
    beta=st.floats(min_value=0.1, max_value=20.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_hybrid_partition_properties(num_clients, extra_samples, alpha, beta, seed):
    num_samples = num_clients + extra_samples
    labels = np.random.default_rng(seed).integers(0, 5, size=num_samples)

    pre_state = np.random.get_state()
    client_dict = hybrid_partition(labels, num_clients, alpha=alpha, beta=beta, seed=seed)
    assert_partition_invariants(
        client_dict,
        num_samples,
        pre_state,
        hybrid_partition,
        labels,
        num_clients,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )


@settings(max_examples=50, deadline=None)
@given(
    num_classes=st.integers(min_value=2, max_value=6),
    num_clients=st.integers(min_value=2, max_value=10),
    classes_per_client=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_pathological_partition_properties(num_classes, num_clients, classes_per_client, seed):
    # Ensure classes_per_client <= num_classes
    classes_per_client = min(classes_per_client, num_classes)
    total_slots = num_clients * classes_per_client
    if total_slots < num_classes:
        num_clients = (num_classes + classes_per_client - 1) // classes_per_client
        total_slots = num_clients * classes_per_client

    # Ensure every class has at least (total_slots // num_classes + 2) samples
    min_per_class = (total_slots // num_classes) + 2
    samples_per_class = min_per_class + 5
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    num_samples = len(labels)

    pre_state = np.random.get_state()
    client_dict = pathological_partition(
        labels,
        num_clients=num_clients,
        classes_per_client=classes_per_client,
        seed=seed,
    )
    assert_partition_invariants(
        client_dict,
        num_samples,
        pre_state,
        pathological_partition,
        labels,
        num_clients,
        classes_per_client,
        seed=seed,
    )
