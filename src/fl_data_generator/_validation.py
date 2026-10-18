from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_labels(labels: Sequence | np.ndarray) -> np.ndarray:
    """Validate and cast dataset labels to a 1D int64 numpy array.

    Parameters
    ----------
    labels : Sequence or np.ndarray
        Dataset labels as an array-like or sequence.

    Returns
    -------
    np.ndarray
        A 1D numpy array with dtype np.int64.

    Raises
    ------
    ValueError
        If labels is empty, not 1-dimensional, contains NaN, Inf, or complex numbers,
        values exceed int64 bounds, or cannot be converted to integer values without
        loss of information.
    TypeError
        If labels elements cannot be converted to numeric values.
    """
    arr = np.asarray(labels)

    if arr.ndim != 1:
        raise ValueError("labels must be 1-dimensional; consider using labels.ravel()")

    if arr.size == 0:
        raise ValueError("labels cannot be empty")

    if np.iscomplexobj(arr):
        raise ValueError("labels cannot contain complex numbers")

    if not np.issubdtype(arr.dtype, np.number):
        # Could be object array with strings or other non-numeric types
        try:
            arr = arr.astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise TypeError("labels must contain numeric values") from exc

    # Check for NaN or Inf
    if np.isnan(arr).any():
        raise ValueError("labels contains NaN values")

    if np.isinf(arr).any():
        raise ValueError("labels contains infinite values")

    # Check if integer-valued and within int64 bounds
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.floor(arr) == arr):
            raise ValueError("labels must be integer-valued")
        i64_info = np.iinfo(np.int64)
        if np.any(arr < i64_info.min) or np.any(arr > i64_info.max):
            raise ValueError("labels values out of bounds for int64")
        arr = arr.astype(np.int64)
    elif np.issubdtype(arr.dtype, np.integer):
        if arr.dtype != np.int64:
            i64_info = np.iinfo(np.int64)
            if np.any(arr < i64_info.min) or np.any(arr > i64_info.max):
                raise ValueError("labels values out of bounds for int64")
            arr = arr.astype(np.int64)

    return arr
