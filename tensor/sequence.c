/*
 * Tensor sequence protocol
 */

#include "tensor.h"

PyObject *
Tensor_sq_item(PyObject *self, Py_ssize_t index)
{
    PyObject *key = PyLong_FromSsize_t(index);
    if (key == NULL) {
        return NULL;
    }

    PyObject *result = Tensor_subscript(self, key);
    Py_DECREF(key);
    return result;
}

int
Tensor_sq_ass_item(PyObject *self, Py_ssize_t index, PyObject *value)
{
    PyObject *key = PyLong_FromSsize_t(index);
    if (key == NULL) {
        return -1;
    }

    int result = Tensor_ass_subscript(self, key, value);
    Py_DECREF(key);
    return result;
}

int
Tensor_contains(PyObject *op, PyObject *value)
{
    TensorObject *self = (TensorObject *)op;

    if (self->nd == 0) {
        // For 0D tensor, compare directly
        double self_val = *((double *)self->data);
        PyObject *float_obj = PyNumber_Float(value);
        if (float_obj == NULL) {
            return -1;
        }
        double val = PyFloat_AsDouble(float_obj);
        Py_DECREF(float_obj);

        return (self_val == val) ? 1 : 0;
    }

    // For 1D, check each element
    for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
        char *data_ptr = self->data + (i * self->strides[0]);
        double self_val = *((double *)data_ptr);

        PyObject *float_obj = PyNumber_Float(value);
        if (float_obj == NULL) {
            return -1;
        }
        double val = PyFloat_AsDouble(float_obj);
        Py_DECREF(float_obj);

        if (self_val == val) {
            return 1;  // Found it!
        }
    }

    return 0;  // Not found
}

PyObject *
Tensor_concat(PyObject *op, PyObject *other)
{
    PyErr_SetString(PyExc_TypeError, "concatenation not supported for Tensor");
    return NULL;
}

PySequenceMethods Tensor_as_sequence = {
    .sq_length = Tensor_length,
    .sq_item = Tensor_sq_item,
    .sq_ass_item = Tensor_sq_ass_item,
    .sq_contains = Tensor_contains,
    .sq_concat = Tensor_concat,
};
