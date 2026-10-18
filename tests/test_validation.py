from __future__ import annotations

import numpy as np
import pytest

from fl_data_generator._validation import validate_labels


def test_validate_labels_valid_integers_list() -> None:
    labels = [0, 1, 2, 1, 0]
    result = validate_labels(labels)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.ndim == 1
    np.testing.assert_array_equal(result, np.array([0, 1, 2, 1, 0], dtype=np.int64))


def test_validate_labels_valid_integers_ndarray() -> None:
    labels_int32 = np.array([0, 2, 3], dtype=np.int32)
    result = validate_labels(labels_int32)
    assert result.dtype == np.int64
    assert result.ndim == 1
    np.testing.assert_array_equal(result, np.array([0, 2, 3], dtype=np.int64))

    labels_int64 = np.array([5, 4, 3], dtype=np.int64)
    result64 = validate_labels(labels_int64)
    assert result64.dtype == np.int64
    np.testing.assert_array_equal(result64, labels_int64)


def test_validate_labels_integer_valued_floats() -> None:
    labels_float = [0.0, 1.0, 2.0]
    result = validate_labels(labels_float)
    assert result.dtype == np.int64
    assert result.ndim == 1
    np.testing.assert_array_equal(result, np.array([0, 1, 2], dtype=np.int64))

    labels_np_float = np.array([0.0, 5.0])
    result_np = validate_labels(labels_np_float)
    assert result_np.dtype == np.int64
    np.testing.assert_array_equal(result_np, np.array([0, 5], dtype=np.int64))


def test_validate_labels_empty_input() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_labels([])

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_labels(np.array([]))


def test_validate_labels_non_1d_input() -> None:
    # 2D array (N, 1)
    with pytest.raises(ValueError, match=r"labels must be 1-dimensional; consider using labels\.ravel\(\)"):
        validate_labels(np.array([[0], [1], [2]]))

    # 2D array (N, M)
    with pytest.raises(ValueError, match=r"labels must be 1-dimensional; consider using labels\.ravel\(\)"):
        validate_labels(np.array([[0, 1], [2, 3]]))

    # 0D scalar
    with pytest.raises(ValueError, match=r"labels must be 1-dimensional; consider using labels\.ravel\(\)"):
        validate_labels(np.array(1))


def test_validate_labels_non_integer_valued_floats() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        validate_labels([0.5, 1.0])

    with pytest.raises(ValueError, match="integer-valued"):
        validate_labels([1.2, 3.4])


def test_validate_labels_with_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        validate_labels([0.0, np.nan])


def test_validate_labels_with_inf() -> None:
    with pytest.raises(ValueError, match="infinite"):
        validate_labels([0.0, np.inf])

    with pytest.raises(ValueError, match="infinite"):
        validate_labels([0.0, -np.inf])


def test_validate_labels_non_numeric_or_object() -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_labels(["cat", "dog"])

    with pytest.raises((ValueError, TypeError)):
        validate_labels(np.array(["a", "b"], dtype=object))


def test_validate_labels_complex_raises() -> None:
    with pytest.raises(ValueError, match="complex"):
        validate_labels(np.array([1 + 2j, 3 + 4j]))


def test_validate_labels_overflow_raises() -> None:
    with pytest.raises(ValueError, match="overflow|out of bounds|cannot fit"):
        validate_labels([1e20, 2e20])
    with pytest.raises(ValueError, match="overflow|out of bounds|cannot fit"):
        validate_labels([2.0**63])
    with pytest.raises(ValueError, match="overflow|out of bounds|cannot fit"):
        validate_labels([-2.0**63 - 10000.0])


