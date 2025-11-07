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
    np.testing.assert_equal(x.to_numpy(), y)


def test_1d_size_1():
    x = tensor.Tensor([42.0])
    y = np.array([42.0])
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert x.base == y.base
    len(x) == len(y)
    assert x.tolist() == y.tolist()
    assert x.item() == y.item()
    np.testing.assert_equal(x.to_numpy(), y)


def test_1d_size_0():
    x = tensor.Tensor([])
    y = np.array([])
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert x.base == y.base
    len(x) == len(y)
    assert x.tolist() == y.tolist()
    with pytest.raises(ValueError):
        x.item()
    np.testing.assert_equal(x.to_numpy(), y)


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


def test_getitem_element():
    data = list(range(1, 33))
    x = tensor.Tensor(data)
    y = np.array(data, dtype=np.float64)

    for i in range(len(data)):
        x_elem = x[i]
        y_elem = y[i]
        assert x_elem.shape == y_elem.shape
        assert x_elem.strides == y_elem.strides
        assert x_elem.ndim == y_elem.ndim
        assert x_elem.size == y_elem.size
        assert x_elem.base is None
        assert x_elem.tolist() == y_elem.tolist()

    for i in range(-1, -len(data) - 1, -1):
        x_elem = x[i]
        y_elem = y[i]
        assert x_elem.shape == y_elem.shape
        assert x_elem.strides == y_elem.strides
        assert x_elem.ndim == y_elem.ndim
        assert x_elem.size == y_elem.size
        assert x_elem.tolist() == y_elem.tolist()

    for i in range(5):
        with pytest.raises(IndexError):
            x[len(data) + i]
        with pytest.raises(IndexError):
            x[-(len(data) + 1 + i)]


@pytest.mark.parametrize("sequence", ALL_SLICE_SEQUENCES)
def test_getitem_slice(sequence):
    data = list(range(1, 33))
    x = tensor.Tensor(data)
    y = np.array(data, dtype=np.float64)

    view_tensor, view_numpy = x, y
    original_tensor = x
    for slicer in sequence:
        x_slice = view_tensor[slicer]
        y_slice = view_numpy[slicer]
        assert x_slice.shape == y_slice.shape
        assert x_slice.strides == y_slice.strides
        assert x_slice.ndim == y_slice.ndim
        assert x_slice.size == y_slice.size
        assert len(x_slice) == len(y_slice)
        assert x_slice.base is original_tensor
        assert x_slice.tolist() == y_slice.tolist()
        view_tensor, view_numpy = x_slice, y_slice


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


def test_setitem_element_element():
    data = [42.0]
    for i in range(-5, 5):
        x = tensor.tensor(data)
        y = np.array(data)
        if i == 0 or i == -1:
            x[i] = 1.0
            y[i] = 1.0
            assert x.shape == y.shape
            assert x.strides == y.strides
            assert x.ndim == y.ndim
            assert x.size == y.size
            assert x.base is None
            assert x.tolist() == y.tolist()
        else:
            with pytest.raises(IndexError):
                x[i] = 1.0
            with pytest.raises(IndexError):
                y[i] = 1.0

    data = []
    for i in range(-5, 5):
        x = tensor.tensor(data)
        y = np.array(data)
        with pytest.raises(IndexError):
            x[i] = 1.0
        with pytest.raises(IndexError):
            y[i] = 1.0


@pytest.mark.parametrize("slice", BASE_SLICES)
def test_setitem_slice_element(slice):
    data = list(range(1, 33))

    for value in (99.0, [99.0]):
        x = tensor.tensor(data)
        y = np.array(data, dtype=np.float64)
        x[slice] = value
        y[slice] = value
        assert x.shape == y.shape
        assert x.strides == y.strides
        assert x.ndim == y.ndim
        assert x.size == y.size
        assert len(x) == len(y)
        assert x.base is None
        assert x.tolist() == y.tolist()

        x = tensor.tensor(data)
        y = np.array(data, dtype=np.float64)
        x[slice] = tensor.tensor(value)
        y[slice] = np.array(value)
        assert x.shape == y.shape
        assert x.strides == y.strides
        assert x.ndim == y.ndim
        assert x.size == y.size
        assert len(x) == len(y)
        assert x.base is None
        assert x.tolist() == y.tolist()


@pytest.mark.parametrize("slice", BASE_SLICES)
def test_setitem_slice_sequence(slice):
    data = list(range(1, 33))
    x = tensor.tensor(data)
    y = np.array(data, dtype=np.float64)
    slice_length = len(y[slice])
    values = [99.0] * slice_length
    x[slice] = values
    y[slice] = values
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert len(x) == len(y)
    assert x.base is None
    assert x.tolist() == y.tolist()

    x = tensor.tensor(data)
    y = np.array(data, dtype=np.float64)
    x[slice] = tensor.tensor(values)
    y[slice] = np.array(values)
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
    assert x.size == y.size
    assert len(x) == len(y)
    assert x.base is None
    assert x.tolist() == y.tolist()

    for i in range(-2, 3):
        x = tensor.tensor(data)
        y = np.array(data, dtype=np.float64)
        incorrect_length = slice_length + i
        values = [99.0] * incorrect_length
        # if numpy raises ValueError, tensor should too
        try:
            y[slice] = values
        except ValueError:
            with pytest.raises(ValueError):
                x[slice] = values
        else:
            x[slice] = values
            assert x.shape == y.shape
            assert x.strides == y.strides
            assert x.ndim == y.ndim
            assert x.size == y.size
            assert len(x) == len(y)
            assert x.base is None
            assert x.tolist() == y.tolist()

        x = tensor.tensor(data)
        y = np.array(data, dtype=np.float64)
        incorrect_length = slice_length + i
        values = [99.0] * incorrect_length
        # if numpy raises ValueError, tensor should too
        try:
            y[slice] = np.array(values)
        except ValueError:
            with pytest.raises(ValueError):
                x[slice] = tensor.tensor(values)
        else:
            x[slice] = tensor.tensor(values)
            assert x.shape == y.shape
            assert x.strides == y.strides
            assert x.ndim == y.ndim
            assert x.size == y.size
            assert len(x) == len(y)
            assert x.base is None
            assert x.tolist() == y.tolist()
