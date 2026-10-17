from typing import Dict, Optional, Sequence, Union
import numpy as np


def dirichlet_partition(
    labels: Union[Sequence, np.ndarray],
    num_clients: int,
    alpha: Union[float, Sequence[float]] = 0.5,
    seed: Optional[int] = None,
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

    Returns
    -------
    dict of int to np.ndarray
        Mapping from client ID to array of assigned sample indices.
    """
    labels_arr = np.asarray(labels)
    num_samples = len(labels_arr)

    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if num_samples == 0:
        raise ValueError("labels cannot be empty")

    if isinstance(alpha, (int, float)):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        alpha_vec = np.repeat(float(alpha), num_clients)
    else:
        alpha_vec = np.asarray(alpha, dtype=float)
        if alpha_vec.shape != (num_clients,):
            raise ValueError(
                f"alpha length must match num_clients ({num_clients}), got {len(alpha_vec)}"
            )
        if (alpha_vec <= 0).any():
            raise ValueError("All alpha values must be positive")

    rng = np.random.default_rng(seed)
    client_indices: Dict[int, list] = {cid: [] for cid in range(num_clients)}

    unique_classes = np.unique(labels_arr)

    for cls in unique_classes:
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(alpha_vec)
        split_points = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        splits = np.split(cls_idx, split_points)

        for cid, split in enumerate(splits):
            if len(split) > 0:
                client_indices[cid].append(split)

    result = {}
    for cid in range(num_clients):
        if client_indices[cid]:
            merged = np.concatenate(client_indices[cid])
            rng.shuffle(merged)
            result[cid] = merged
        else:
            result[cid] = np.array([], dtype=int)

    return result
