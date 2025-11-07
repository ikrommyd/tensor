import numpy as np
import tensor
import pytest


def test_0d():
    x = tensor.Tensor(42.0)
    y = np.array(42.0)
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert x.base == y.base
    with pytest.raises(TypeError):
        len(x)
    assert x.tolist() == y.tolist()
    assert x.item() == y.item()


def test_1d():
    x = tensor.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert x.base == y.base
    len(x) == len(y)
    x = tensor.Tensor([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert x.base == y.base
    len(x) == len(y)
    assert x.tolist() == y.tolist()
    with pytest.raises(ValueError):
        x.item()


def test_tensor_tensor():
    x = tensor.tensor(42.0)
    y = np.array(42.0)
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert x.base == y.base
    with pytest.raises(TypeError):
        len(x)
    assert x.tolist() == y.tolist()
    assert x.item() == y.item()

    x = tensor.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert x.base == y.base
    len(x) == len(y)
    assert x.tolist() == y.tolist()
    with pytest.raises(ValueError):
        x.item()


def test_tensor_copy():
    x = tensor.tensor(42.0)
    y = np.array(42.0)
    x_copy = x.copy()
    y_copy = y.copy()
    assert x_copy is not x
    assert x_copy.shape == y_copy.shape
    assert x_copy.strides == y_copy.strides
    assert x_copy.ndim == y_copy.ndim
    assert x_copy.size == y_copy.size
    assert x_copy.base == y_copy.base
    with pytest.raises(TypeError):
        len(x_copy)
    assert x_copy.tolist() == y_copy.tolist()
    assert x_copy.item() == y_copy.item()

    x_copy = tensor.tensor(x, copy=True)
    y_copy = np.asarray(y, copy=True)
    assert x_copy is not x
    assert x_copy.shape == y_copy.shape
    assert x_copy.strides == y_copy.strides
    assert x_copy.ndim == y_copy.ndim
    assert x_copy.size == y_copy.size
    assert x_copy.base == y_copy.base
    with pytest.raises(TypeError):
        len(x_copy)
    assert x_copy.tolist() == y_copy.tolist()
    assert x_copy.item() == y_copy.item()

    x = tensor.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x_copy = x.copy()
    y_copy = y.copy()
    assert x_copy is not x
    assert x_copy.shape == y_copy.shape
    assert x_copy.strides == y_copy.strides
    assert x_copy.ndim == y_copy.ndim
    assert x_copy.size == y_copy.size
    assert x_copy.base == y_copy.base
    len(x_copy) == len(y_copy)
    assert x_copy.tolist() == y_copy.tolist()
    with pytest.raises(ValueError):
        x_copy.item()

    x_copy = tensor.tensor(x, copy=True)
    y_copy = np.asarray(y, copy=True)
    assert x_copy is not x
    assert x_copy.shape == y_copy.shape
    assert x_copy.strides == y_copy.strides
    assert x_copy.ndim == y_copy.ndim
    assert x_copy.size == y_copy.size
    assert x_copy.base == y_copy.base
    len(x_copy) == len(y_copy)
    assert x_copy.tolist() == y_copy.tolist()
    with pytest.raises(ValueError):
        x_copy.item()


def assert_slice_matches(tensor_view, numpy_view, index, expected_base):
    x_slice = tensor_view[index]
    y_slice = numpy_view[index]
    assert x_slice.shape == y_slice.shape
    assert x_slice.strides == y_slice.strides
    assert x_slice.ndim == y_slice.ndim
    assert x_slice.size == y_slice.size
    assert len(x_slice) == len(y_slice)
    assert x_slice.base is expected_base
    assert x_slice.tolist() == y_slice.tolist()
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
    x = tensor.Tensor(data)
    y = np.array(data, dtype=np.float64)

    view_tensor, view_numpy = x, y
    for slicer in sequence:
        view_tensor, view_numpy = assert_slice_matches(
            view_tensor, view_numpy, slicer, x
        )


@pytest.mark.parametrize("sequence", ALL_SLICE_SEQUENCES)
def test_copy_of_sliced_tensor(sequence):
    data = list(range(1, 33))
    x = tensor.Tensor(data)
    y = np.array(data, dtype=np.float64)

    view_tensor, view_numpy = x, y
    for slicer in sequence:
        view_tensor, view_numpy = view_tensor[slicer], view_numpy[slicer]

    x_copy = view_tensor.copy()
    y_copy = view_numpy.copy()

    assert x_copy.shape == y_copy.shape
    assert x_copy.strides == y_copy.strides
    assert x_copy.ndim == y_copy.ndim
    assert x_copy.size == y_copy.size
    assert len(x_copy) == len(y_copy)
    assert x_copy.base is not view_tensor.base
    assert x_copy.tolist() == y_copy.tolist()

    x_copy = tensor.tensor(view_tensor, copy=True)
    y_copy = np.asarray(view_numpy, copy=True)

    assert x_copy.shape == y_copy.shape
    assert x_copy.strides == y_copy.strides
    assert x_copy.ndim == y_copy.ndim
    assert x_copy.size == y_copy.size
    assert len(x_copy) == len(y_copy)
    assert x_copy.base is not view_tensor.base
    assert x_copy.tolist() == y_copy.tolist()
