from typing import Dict, Optional, Sequence, Union

import numpy as np

from ._validation import validate_labels, validate_partitioner_int


def pathological_partition(
    labels: Union[Sequence, np.ndarray],
    num_clients: int,
    classes_per_client: int = 2,
    seed: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Partition dataset indices among clients with pathological non-IID label skew.

    Assigns k distinct classes per client (NIID-Bench #C=k, Li et al. 2021),
    NOT the McMahan et al. 2017 sort-and-shard scheme. Each class's samples are
    split into shards, and clients receive shards corresponding to their assigned
    classes.

    Parameters
    ----------
    labels : Sequence or np.ndarray
        Dataset class labels or targets.
    num_clients : int
        Number of clients to partition data among.
    classes_per_client : int, default=2
        Number of distinct classes assigned to each client.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict of int to np.ndarray
        Mapping from client ID to array of assigned sample indices.

    Raises
    ------
    ValueError
        If labels is invalid (empty, not 1D, non-integer, or contains NaN/Inf).
        If num_clients <= 0 or classes_per_client <= 0.
        If classes_per_client exceeds the number of unique classes.
        If total client slots is less than the number of unique classes.
        If total samples is less than total class shards.
        If any class has fewer samples than the number of shards assigned to it.
    """
    labels_arr = validate_labels(labels)
    num_samples = len(labels_arr)

    num_clients = validate_partitioner_int("num_clients", num_clients, min_value=1)
    classes_per_client = validate_partitioner_int(
        "classes_per_client", classes_per_client, min_value=1
    )


    unique_classes = np.unique(labels_arr)
    num_classes = len(unique_classes)

    if classes_per_client > num_classes:
        raise ValueError(
            f"classes_per_client ({classes_per_client}) cannot exceed number of unique classes ({num_classes})"
        )

    total_slots = num_clients * classes_per_client
    if total_slots < num_classes:
        raise ValueError(
            f"Total client class slots ({num_clients} * {classes_per_client} = {total_slots}) "
            f"must be at least the number of unique classes ({num_classes}) to allocate all samples."
        )

    if num_samples < total_slots:
        raise ValueError(
            f"Total samples ({num_samples}) must be at least total class shards ({total_slots})"
        )

    rng = np.random.default_rng(seed)

    # Distribute class slots across clients
    counts = np.ones(num_classes, dtype=int)
    remaining = total_slots - num_classes
    base = remaining // num_classes
    rem = remaining % num_classes
    counts += base
    counts[:rem] += 1

    # Check overall feasibility against the multiset of available shard counts:
    # Sort class sample counts and counts to ensure classes with more samples get higher or equal shard requests
    class_sample_counts = np.array([int(np.count_nonzero(labels_arr == c)) for c in unique_classes])
    
    # Sort classes by sample count descending, so larger classes receive the larger shard counts (base + 1)
    sorted_by_samples = np.argsort(-class_sample_counts)
    assigned_counts = np.zeros(num_classes, dtype=int)
    assigned_counts[sorted_by_samples] = counts  # counts is already sorted descending [base+1..., base...]

    # If even the most favorable assignment cannot satisfy shard requirements, raise deterministic ValueError
    for c_idx, cls_val in enumerate(unique_classes):
        sample_cnt = class_sample_counts[c_idx]
        needed_shards = assigned_counts[c_idx]
        if sample_cnt < needed_shards:
            raise ValueError(
                f"Class '{cls_val}' has {sample_cnt} sample(s), which is fewer than "
                f"the assigned {needed_shards} shard(s). Cannot partition without producing empty shards."
            )
    counts = assigned_counts


    client_classes = [set() for _ in range(num_clients)]
    sorted_class_indices = np.argsort(-counts)

    for c_idx in sorted_class_indices:
        cls_val = unique_classes[c_idx]
        needed = counts[c_idx]

        candidates = [
            cid
            for cid in range(num_clients)
            if cls_val not in client_classes[cid]
            and len(client_classes[cid]) < classes_per_client
        ]
        candidates.sort(key=lambda cid: (len(client_classes[cid]), rng.random()))

        for cid in candidates[:needed]:
            client_classes[cid].add(cls_val)

    # Split each class's samples into the assigned number of shards
    class_shards: Dict[object, list] = {}
    for c_idx, cls_val in enumerate(unique_classes):
        cls_indices = np.where(labels_arr == cls_val)[0]
        rng.shuffle(cls_indices)
        num_shards = counts[c_idx]
        class_shards[cls_val] = list(np.array_split(cls_indices, num_shards))

    # Assign shards to clients
    result: Dict[int, np.ndarray] = {}
    for cid in range(num_clients):
        cid_shards = []
        for cls_val in client_classes[cid]:
            cid_shards.append(class_shards[cls_val].pop())

        merged = np.concatenate(cid_shards)
        rng.shuffle(merged)
        result[cid] = merged

    return result
