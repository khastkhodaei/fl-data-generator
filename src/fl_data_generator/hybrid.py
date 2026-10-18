from typing import Dict, Optional, Sequence, Union

import numpy as np

from ._validation import validate_labels


def hybrid_partition(
    labels: Union[Sequence, np.ndarray],
    num_clients: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    seed: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Partition dataset indices combining label skew and quantity skew (experimental).

    .. warning::
        This method is experimental and not a standard published partitioning method.
        It uses sequential allocation: client sample capacities are first determined
        via Dirichlet distribution with parameter beta (quantity skew). Samples are then
        allocated sequentially to clients in a randomly permuted client order based on class
        preference distributions drawn from a Dirichlet distribution with parameter alpha
        (label skew). Because samples are drawn without replacement from a shared finite pool,
        there is a client-order effect: earlier clients in the sequence have access to the
        full class distribution, while later clients may receive residual classes that deviate
        from their drawn preferences.

    Parameters
    ----------
    labels : Sequence or np.ndarray
        Dataset class labels or targets.
    num_clients : int
        Number of clients to partition data among.
    alpha : float, default=0.5
        Concentration parameter for the Dirichlet distribution governing label skew.
        Must be a positive scalar. Smaller values induce stronger class preference heterogeneity.
    beta : float, default=0.5
        Concentration parameter for the Dirichlet distribution governing quantity skew.
        Must be a positive scalar. Smaller values induce greater disparity in sample volume across clients.
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
        If num_clients <= 0, alpha <= 0, or beta <= 0.
        If num_clients exceeds num_samples.
        If alpha or beta is a sequence rather than a scalar.
    """
    labels_arr = validate_labels(labels)
    num_samples = len(labels_arr)

    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if num_clients > num_samples:
        raise ValueError(
            f"num_clients ({num_clients}) cannot exceed number of samples ({num_samples})"
        )

    if isinstance(alpha, (Sequence, np.ndarray)) or np.ndim(alpha) > 0 or not isinstance(alpha, (int, float, np.number)) or isinstance(alpha, bool):
        raise ValueError(f"alpha must be a scalar positive number, got {alpha}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    if isinstance(beta, (Sequence, np.ndarray)) or np.ndim(beta) > 0 or not isinstance(beta, (int, float, np.number)) or isinstance(beta, bool):
        raise ValueError(f"beta must be a scalar positive number, got {beta}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels_arr)
    num_classes = len(unique_classes)
    cls_to_idx = {c: i for i, c in enumerate(unique_classes)}

    # 1. Determine client sample sizes via quantity skew (Dirichlet with parameter beta)
    q_props = rng.dirichlet(np.repeat(beta, num_clients))
    remaining = num_samples - num_clients
    splits = (np.cumsum(q_props)[:-1] * remaining).astype(int)
    client_sizes = np.diff(np.concatenate(([0], splits, [remaining]))) + 1

    # Organize available sample indices by class
    class_indices: Dict[object, list] = {}
    for cls in unique_classes:
        idx = np.where(labels_arr == cls)[0]
        rng.shuffle(idx)
        class_indices[cls] = list(idx)

    # 2. Allocate samples to each client proportionally based on label skew (Dirichlet alpha)
    client_partitions: Dict[int, list] = {cid: [] for cid in range(num_clients)}
    client_order = rng.permutation(num_clients)

    for cid in client_order:
        needed = client_sizes[cid]
        p = rng.dirichlet(np.repeat(alpha, num_classes))

        while needed > 0:
            avail_classes = [c for c in unique_classes if len(class_indices[c]) > 0]
            if not avail_classes:
                break

            avail_p = np.array([p[cls_to_idx[c]] for c in avail_classes])
            p_sum = avail_p.sum()
            if p_sum == 0:
                avail_p = np.ones(len(avail_classes))
                p_sum = len(avail_classes)
            avail_p = avail_p / p_sum

            counts = (avail_p * needed).astype(int)
            if counts.sum() == 0:
                counts[np.argmax(avail_p)] = 1

            allocated_this_round = 0
            for c, cnt in zip(avail_classes, counts):
                take = min(cnt, len(class_indices[c]), needed)
                if take > 0:
                    for _ in range(take):
                        client_partitions[cid].append(class_indices[c].pop())
                    needed -= take
                    allocated_this_round += take

            if allocated_this_round == 0:
                c = avail_classes[0]
                client_partitions[cid].append(class_indices[c].pop())
                needed -= 1

    result: Dict[int, np.ndarray] = {}
    for cid in range(num_clients):
        arr = np.array(client_partitions[cid], dtype=int)
        rng.shuffle(arr)
        result[cid] = arr

    return result
