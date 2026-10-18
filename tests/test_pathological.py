import numpy as np
import pytest

from fl_data_generator import pathological_partition


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
    rng = np.random.default_rng(500)
    labels = rng.integers(0, 6, size=300)
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


def test_pathological_grid_invariants():
    num_clients_list = [3, 5, 7, 10, 13]
    num_classes_list = [2, 3, 5, 10]
    classes_per_client_list = [1, 2, 3]
    seeds = range(20)

    violations = []
    for nc in num_clients_list:
        for ncls in num_classes_list:
            for cpc in classes_per_client_list:
                for s in seeds:
                    labels = np.repeat(np.arange(ncls), 40)
                    n_samples = 40 * ncls
                    try:
                        result = pathological_partition(
                            labels=labels,
                            num_clients=nc,
                            classes_per_client=cpc,
                            seed=s,
                        )
                    except ValueError:
                        continue
                    except Exception as e:  # noqa: BLE001
                        violations.append(
                            f"nc={nc}, ncls={ncls}, cpc={cpc}, seed={s}: CRASH {type(e).__name__}: {e}"
                        )
                        continue

                    all_indices = []
                    for cid in range(nc):
                        if cid not in result:
                            violations.append(
                                f"nc={nc}, ncls={ncls}, cpc={cpc}, seed={s}: client {cid} missing from result"
                            )
                        else:
                            indices = result[cid]
                            all_indices.extend(indices)
                            distinct = len(np.unique(labels[indices]))
                            if distinct != cpc:
                                violations.append(
                                    f"nc={nc}, ncls={ncls}, cpc={cpc}, seed={s}: client {cid} has {distinct} classes, expected {cpc}"
                                )

                    if set(all_indices) != set(range(n_samples)):
                        violations.append(
                            f"nc={nc}, ncls={ncls}, cpc={cpc}, seed={s}: MISSING_INDICES ({len(set(range(n_samples)) - set(all_indices))} lost)"
                        )
                    if len(all_indices) != len(set(all_indices)):
                        violations.append(
                            f"nc={nc}, ncls={ncls}, cpc={cpc}, seed={s}: DUPLICATES ({len(all_indices) - len(set(all_indices))} duplicates)"
                        )

    assert not violations, f"Grid violations ({len(violations)} total):\n" + "\n".join(violations[:10])


def test_pathological_class_with_fewer_samples_than_shards():
    """Class with fewer samples than assigned shards must raise ValueError."""
    labels = np.array([0] * 1 + [1] * 15 + [2] * 14)
    num_clients = 10
    classes_per_client = 2
    seed = 42

    with pytest.raises(
        ValueError,
        match=r"Class.*0.*has 1 sample\(s\).*fewer than.*assigned.*shard",
    ):
        pathological_partition(
            labels, num_clients=num_clients, classes_per_client=classes_per_client, seed=seed
        )


def test_pathological_boundary_sample_count_sufficient():
    """Each class has exactly 2 samples for 2 assigned shards: succeeds."""
    labels = np.array([0, 0, 1, 1, 2, 2])
    client_dict = pathological_partition(labels, num_clients=3, classes_per_client=2, seed=42)
    assert len(client_dict) == 3
    all_indices = []
    for indices in client_dict.values():
        assert len(np.unique(labels[indices])) == 2
        all_indices.extend(indices)
    assert set(all_indices) == set(range(6))


def test_pathological_boundary_sample_count_insufficient():
    """Class 0 has 1 sample but 2 shards assigned: raises ValueError."""
    labels = np.array([0, 1, 1, 2, 2, 2])
    with pytest.raises(
        ValueError,
        match=r"Class.*0.*has 1 sample\(s\).*fewer than.*assigned.*2 shard",
    ):
        pathological_partition(labels, num_clients=3, classes_per_client=2, seed=42)


def test_pathological_partition_global_rng_untouched():
    """Global np.random state must not be modified by pathological_partition."""
    labels = np.repeat(np.arange(3), 20)
    state_before = np.random.get_state()
    pathological_partition(labels, num_clients=3, classes_per_client=2, seed=42)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_pathological_partition_2d_labels_raises():
    labels = np.array([[0], [1], [2]])
    with pytest.raises(
        ValueError,
        match=r"labels must be 1-dimensional; consider using labels\.ravel\(\)",
    ):
        pathological_partition(labels, num_clients=2, classes_per_client=1)


def test_pathological_partition_float_labels_raises():
    labels = np.array([0.5, 1.2, 3.4])
    with pytest.raises(ValueError, match=r"labels must be integer-valued"):
        pathological_partition(labels, num_clients=2, classes_per_client=1)


def test_pathological_partition_seed_compatibility_regression():
    labels = np.repeat(np.arange(10), 50)
    res = pathological_partition(labels, num_clients=10, classes_per_client=2, seed=42)
    expected_client_0_prefix = np.array([181, 394, 366, 186, 370])
    np.testing.assert_array_equal(res[0][:5], expected_client_0_prefix)



