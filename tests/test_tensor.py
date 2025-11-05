import numpy as np
from tensor import Tensor


def test():
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert x.shape == y.shape
    assert x.strides == y.strides
    assert x.ndim == y.ndim
