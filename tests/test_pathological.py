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


@pytest.mark.xfail(
    strict=True,
    reason="pathological_partition leaves shards unassigned / wrong class count; fixed in #9 (T2)",
)
def test_pathological_class_with_fewer_samples_than_shards():
    labels = np.array([0] * 1 + [1] * 15 + [2] * 14)
    num_clients = 10
    classes_per_client = 2
    seed = 42

    result = pathological_partition(
        labels, num_clients=num_clients, classes_per_client=classes_per_client, seed=seed
    )

    all_indices = []
    for cid in range(num_clients):
        assert cid in result
        indices = result[cid]
        all_indices.extend(indices)
        distinct = len(np.unique(labels[indices]))
        assert distinct == classes_per_client, (
            f"Client {cid} has {distinct} distinct classes, expected {classes_per_client}"
        )

    assert set(all_indices) == set(range(len(labels)))
    assert len(all_indices) == len(labels)

