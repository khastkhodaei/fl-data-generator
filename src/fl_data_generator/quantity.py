from typing import Dict, Optional, Sequence, Union

import numpy as np


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
    """
    num_samples = len(labels)

    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if num_samples == 0:
        raise ValueError("labels cannot be empty")
    if num_clients > num_samples:
        raise ValueError(
            f"num_clients ({num_clients}) cannot exceed number of samples ({num_samples})"
        )

    if isinstance(beta, (int, float)):
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        beta_vec = np.repeat(float(beta), num_clients)
    else:
        beta_vec = np.asarray(beta, dtype=float)
        if beta_vec.shape != (num_clients,):
            raise ValueError(
                f"beta length must match num_clients ({num_clients}), got {len(beta_vec)}"
            )
        if (beta_vec <= 0).any():
            raise ValueError("All beta values must be positive")

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
