import numpy as np
from tensor import Tensor
import pytest


def assert_slice_matches(tensor_view, numpy_view, index, expected_base):
    x_slice = tensor_view[index]
    y_slice = numpy_view[index]
    assert x_slice.shape == y_slice.shape
    assert x_slice.strides == y_slice.strides
    assert x_slice.ndim == y_slice.ndim
    assert x_slice.size == y_slice.size
    assert len(x_slice) == len(y_slice)
    assert x_slice.base is expected_base
    return x_slice, y_slice


BASE_SLICES = [
    slice(None, None, None),
    slice(None, None, 1),
    slice(None, None, 2),
    slice(None, None, 3),
    slice(None, None, 4),
    slice(None, None, -1),
    slice(None, None, -2),
    slice(None, None, -3),
    slice(None, None, -4),
    slice(0, None, None),
    slice(1, None, None),
    slice(2, None, None),
    slice(4, None, None),
    slice(-1, None, None),
    slice(-4, None, None),
    slice(None, 0, None),
    slice(None, 3, None),
    slice(None, 6, None),
    slice(None, 12, None),
    slice(None, -1, None),
    slice(None, -4, None),
    slice(1, 6, None),
    slice(1, 6, 2),
    slice(2, 14, 2),
    slice(14, 2, -1),
    slice(14, 2, -2),
    slice(6, 1, -1),
    slice(-10, -1, None),
    slice(-10, -1, -2),
    slice(-12, -4, None),
    slice(-12, -4, -3),
    slice(16, None, None),
    slice(16, None, -1),
    slice(32, None, None),
    slice(32, None, -1),
    slice(None, 32, None),
    slice(0, 0, None),
    slice(4, 4, None),
    slice(3, 15, 3),
    slice(15, 3, -3),
]


NESTED_SLICE_SEQUENCES = [
    [slice(None)],
    [slice(None, None, 2), slice(None, None, 2)],
    [slice(None, None, 2), slice(None, None, 2), slice(None, None, 2)],
    [slice(None, None, 3), slice(None, None, 3)],
    [slice(None, None, 3), slice(None, None, 3), slice(None, None, 3)],
    [slice(None, None, 4), slice(None, None, 4)],
    [slice(None, None, -1), slice(None, None, 2)],
    [slice(None, None, -1), slice(None, None, -2)],
    [slice(None, None, -2), slice(None, None, -2)],
    [slice(None, None, -3), slice(None, None, -3)],
    [slice(None, None, 2), slice(None, None, -1), slice(None, None, 2)],
    [slice(None, None, -2), slice(None, None, 2), slice(None, None, -2)],
    [slice(None, None, -1), slice(None, None, -1), slice(None, None, -1)],
    [slice(1, None, 2), slice(None, None, 2)],
    [slice(1, None, 2), slice(None, None, -2)],
    [slice(1, None, 3), slice(None, None, -1)],
    [slice(1, None, 3), slice(None, None, 2)],
    [slice(2, 20, 2), slice(None, None, 2)],
    [slice(2, 20, 2), slice(None, None, -2)],
    [slice(1, 25, 2), slice(None, None, 2), slice(None, None, 2)],
    [slice(1, 25, 3), slice(None, None, -3), slice(None, None, 3)],
    [slice(None, None, -2), slice(None, None, -2), slice(None, None, -2)],
    [slice(None, None, 4), slice(None, None, -2)],
    [slice(14, 2, -1), slice(None, None, 2)],
    [slice(14, 2, -1), slice(None, None, -2)],
    [slice(15, 3, -3), slice(None, None, -3)],
]


ALL_SLICE_SEQUENCES = [[slc] for slc in BASE_SLICES] + NESTED_SLICE_SEQUENCES


@pytest.mark.parametrize("sequence", ALL_SLICE_SEQUENCES)
def test_slicing(sequence):
    data = list(range(1, 33))
    x = Tensor(data)
    y = np.array(data)

    view_tensor, view_numpy = x, y
    for slicer in sequence:
        view_tensor, view_numpy = assert_slice_matches(
            view_tensor, view_numpy, slicer, x
        )
