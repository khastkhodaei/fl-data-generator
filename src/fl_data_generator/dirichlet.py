from typing import Dict, Optional, Sequence, Union

import numpy as np

from ._validation import validate_labels, validate_partitioner_int


def dirichlet_partition(
    labels: Union[Sequence, np.ndarray],
    num_clients: int,
    alpha: Union[float, Sequence[float]] = 0.5,
    seed: Optional[int] = None,
    min_size: int = 1,
    max_retries: int = 100,
) -> Dict[int, np.ndarray]:
    """Partition dataset indices among clients using a Dirichlet distribution over classes.

    Simulates non-IID label skew across clients in Federated Learning. Smaller alpha values
    result in higher distribution skew (more non-IID), while larger alpha values approach
    an IID distribution.

    Parameters
    ----------
    labels : Sequence or np.ndarray
        Dataset class labels or targets.
    num_clients : int
        Number of clients to partition data among.
    alpha : float or Sequence[float], default=0.5
        Concentration parameter for the Dirichlet distribution.
    seed : int, optional
        Random seed for reproducibility.
    min_size : int, default=1
        Minimum number of samples required per client.
    max_retries : int, default=100
        Maximum number of Dirichlet resampling attempts to satisfy min_size.

    Returns
    -------
    dict of int to np.ndarray
        Mapping from client ID to array of assigned sample indices.

    Raises
    ------
    ValueError
        If labels is invalid (empty, not 1D, non-integer, or contains NaN/Inf).
        If num_clients <= 0 or alpha <= 0.
        If num_clients exceeds num_samples.
        If alpha is a sequence and its length does not match num_clients.
        If min_size < 0, not an integer, or min_size * num_clients > num_samples.
    RuntimeError
        If unable to partition data such that all clients receive at least min_size samples
        within max_retries attempts.
    """
    labels_arr = validate_labels(labels)
    num_samples = len(labels_arr)

    num_clients = validate_partitioner_int("num_clients", num_clients, min_value=1)
    if num_clients > num_samples:
        raise ValueError(
            f"num_clients ({num_clients}) cannot exceed number of samples ({num_samples})"
        )

    if not isinstance(min_size, int) or isinstance(min_size, bool):
        raise ValueError(f"min_size must be an integer, got {type(min_size).__name__}")  # noqa: TRY004
    if min_size < 0:
        raise ValueError(f"min_size must be non-negative, got {min_size}")
    if min_size * num_clients > num_samples:
        raise ValueError(
            f"min_size * num_clients ({min_size * num_clients}) cannot exceed number of samples ({num_samples})"
        )
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries <= 0:
        raise ValueError(f"max_retries must be a positive integer, got {max_retries}")

    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha cannot be a boolean")

    if isinstance(alpha, (int, float, np.number)) or (isinstance(alpha, np.ndarray) and alpha.ndim == 0):
        alpha_val = float(alpha)
        if not np.isfinite(alpha_val) or alpha_val <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        alpha_vec = np.repeat(alpha_val, num_clients)
    else:
        alpha_vec = np.asarray(alpha, dtype=float)
        if alpha_vec.shape != (num_clients,):
            raise ValueError(
                f"alpha length must match num_clients ({num_clients}), got {alpha_vec.size}"
            )
        if (alpha_vec <= 0).any() or not np.all(np.isfinite(alpha_vec)):
            raise ValueError("All alpha values must be positive and finite")

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels_arr)

    for _ in range(max_retries):
        client_indices: Dict[int, list] = {cid: [] for cid in range(num_clients)}

        for cls in unique_classes:
            cls_idx = np.where(labels_arr == cls)[0]
            rng.shuffle(cls_idx)

            proportions = rng.dirichlet(alpha_vec)
            split_points = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            splits = np.split(cls_idx, split_points)

            for cid, split in enumerate(splits):
                if len(split) > 0:
                    client_indices[cid].append(split)

        if all(sum(len(s) for s in client_indices[cid]) >= min_size for cid in range(num_clients)):
            break
    else:
        raise RuntimeError(
            f"Could not partition data such that all clients receive at least min_size={min_size} samples "
            f"after {max_retries} retries. Try using a larger alpha or a smaller min_size."
        )

    result = {}
    for cid in range(num_clients):
        if client_indices[cid]:
            merged = np.concatenate(client_indices[cid])
            rng.shuffle(merged)
            result[cid] = merged
        else:
            result[cid] = np.array([], dtype=int)

    return result
