from typing import Dict, Optional, Sequence, Union

import numpy as np

from ._validation import validate_labels, validate_partitioner_int


def iid_partition(
    labels: Union[Sequence, np.ndarray],
    num_clients: int,
    seed: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Partition dataset indices in an IID fashion across clients.

    Parameters
    ----------
    labels : Sequence or np.ndarray
        Dataset labels or target sequence.
    num_clients : int
        Number of clients to partition data among.
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
        If num_clients <= 0 or num_clients > num_samples.
    """
    labels_arr = validate_labels(labels)
    num_samples = len(labels_arr)

    num_clients = validate_partitioner_int("num_clients", num_clients, min_value=1)
    if num_clients > num_samples:
        raise ValueError(
            f"num_clients ({num_clients}) cannot exceed number of samples ({num_samples})"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    client_splits = np.array_split(indices, num_clients)

    return {cid: split for cid, split in enumerate(client_splits)}
