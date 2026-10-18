from typing import Dict, Optional, Sequence, Union

import numpy as np

from ._validation import validate_labels


def quantity_skew_partition(
    labels: Union[Sequence, np.ndarray],
    num_clients: int,
    beta: Union[float, Sequence[float]] = 0.5,
    seed: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Partition dataset indices among clients with quantity skew (varying sample sizes).

    Simulates realistic volume heterogeneity across clients in Federated Learning.
    The proportion of total samples assigned to each client is drawn from a Dirichlet
    distribution parameterized by beta. Smaller beta values produce severe volume
    disparities, while larger beta values approach uniform sample sizes.

    Parameters
    ----------
    labels : Sequence or np.ndarray
        Dataset labels or target sequence.
    num_clients : int
        Number of clients to partition data among.
    beta : float or Sequence[float], default=0.5
        Concentration parameter for the Dirichlet distribution governing client sample sizes.
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
        If num_clients <= 0 or beta <= 0.
        If num_clients exceeds num_samples.
        If beta is a sequence and its length does not match num_clients.
    """
    labels_arr = validate_labels(labels)
    num_samples = len(labels_arr)

    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if num_clients > num_samples:

        raise ValueError(
            f"num_clients ({num_clients}) cannot exceed number of samples ({num_samples})"
        )

    if isinstance(beta, (bool, np.bool_)):
        raise ValueError("beta cannot be a boolean")

    if isinstance(beta, (int, float, np.number)) or (isinstance(beta, np.ndarray) and beta.ndim == 0):
        beta_val = float(beta)
        if not np.isfinite(beta_val) or beta_val <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        beta_vec = np.repeat(beta_val, num_clients)
    else:
        beta_vec = np.asarray(beta, dtype=float)
        if beta_vec.shape != (num_clients,):
            raise ValueError(
                f"beta length must match num_clients ({num_clients}), got {beta_vec.size}"
            )
        if (beta_vec <= 0).any() or not np.all(np.isfinite(beta_vec)):
            raise ValueError("All beta values must be positive and finite")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_samples)

    # Sample client proportions from Dirichlet distribution
    proportions = rng.dirichlet(beta_vec)

    # Allocate at least 1 sample to each client, distribute remainder according to proportions
    remaining = num_samples - num_clients
    split_points = (np.cumsum(proportions)[:-1] * remaining).astype(int)
    counts = np.diff(np.concatenate(([0], split_points, [remaining]))) + 1

    # Split shuffled indices by cumulative sample counts
    index_splits = np.split(shuffled_indices, np.cumsum(counts)[:-1])

    return {cid: split for cid, split in enumerate(index_splits)}
