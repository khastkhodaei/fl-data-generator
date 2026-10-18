from typing import Dict, Optional, Sequence, Union

import numpy as np


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
        If num_clients <= 0 or classes_per_client <= 0.
        If labels is empty.
        If classes_per_client exceeds the number of unique classes.
        If total client slots is less than the number of unique classes.
        If total samples is less than total class shards.
        If any class has fewer samples than the number of shards assigned to it.
    """
    labels_arr = np.asarray(labels)
    num_samples = len(labels_arr)

    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if classes_per_client <= 0:
        raise ValueError(
            f"classes_per_client must be positive, got {classes_per_client}"
        )
    if num_samples == 0:
        raise ValueError("labels cannot be empty")

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
    rng.shuffle(counts)

    # Validate that every class has at least as many samples as assigned shards
    for c_idx, cls_val in enumerate(unique_classes):
        class_sample_count = int(np.count_nonzero(labels_arr == cls_val))
        needed_shards = int(counts[c_idx])
        if class_sample_count < needed_shards:
            raise ValueError(
                f"Class '{cls_val}' has {class_sample_count} sample(s), which is fewer than "
                f"the assigned {needed_shards} shard(s). Cannot partition without producing empty shards."
            )


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
