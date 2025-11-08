/*
 * Tensor getters and setters
 */

#include "tensor.h"

// Getter for the 'shape' property - returns tuple of dimension sizes
PyObject *
Tensor_getshape(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
    PyObject *shape_tuple = PyTuple_New(self->nd);
    if (shape_tuple == NULL) {
        return NULL;
    }
    // Build tuple containing size of each dimension
    for (int i = 0; i < self->nd; i++) {
        PyObject *dim_size = PyLong_FromSsize_t(self->dimensions[i]);
        if (dim_size == NULL) {
            Py_DECREF(shape_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(shape_tuple, i, dim_size);
    }
    return shape_tuple;
}

// Getter for the 'strides' property - returns tuple of stride values in bytes
PyObject *
Tensor_getstrides(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
    PyObject *strides_tuple = PyTuple_New(self->nd);
    if (strides_tuple == NULL) {
        return NULL;
    }
    // Build tuple containing stride (in bytes) for each dimension
    for (int i = 0; i < self->nd; i++) {
        PyObject *stride_size = PyLong_FromSsize_t(self->strides[i]);
        if (stride_size == NULL) {
            Py_DECREF(strides_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(strides_tuple, i, stride_size);
    }
    return strides_tuple;
}

// Getter for the 'base' property - returns the base tensor if this is a view, else None
PyObject *
Tensor_getbase(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
    if (self->base) {
        // IMPORTANT: We're returning a reference to an existing object
        // Python getters must return a NEW reference, so we INCREF
        // The caller will own this reference and DECREF it when done
        Py_INCREF(self->base);
        return self->base;
    }
    else {
        // Py_RETURN_NONE is a macro that does: Py_INCREF(Py_None); return Py_None;
        Py_RETURN_NONE;
    }
}

// Getter for the 'size' property - returns total number of elements
PyObject *
Tensor_getsize(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
    Py_ssize_t size = 1;
    // Multiply all dimension sizes together
    for (int i = 0; i < self->nd; i++) {
        size *= self->dimensions[i];
    }
    return PyLong_FromSsize_t(size);
}

// Getter for the 'data' property - returns a memoryview of the underlying buffer
PyObject *
Tensor_getdata(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
    if (self->data == NULL) {
        PyErr_SetString(PyExc_ValueError, "Tensor has no data");
        return NULL;
    }
    // Calculate buffer size based on dimensions and strides
    Py_ssize_t bufsize = 1;
    if (self->nd == 0) {
        bufsize = sizeof(double);
    }
    else {
        bufsize = self->dimensions[0] * self->strides[0];
    }
    return PyMemoryView_FromMemory(self->data, bufsize, PyBUF_READ);
}

PyGetSetDef Tensor_getseters[] = {
    {"shape", (getter)Tensor_getshape, NULL, "Shape of the tensor", NULL},
    {"strides", (getter)Tensor_getstrides, NULL, "Strides of the tensor", NULL},
    {"base", (getter)Tensor_getbase, NULL, "Base tensor if view", NULL},
    {"size", (getter)Tensor_getsize, NULL, "Total number of elements", NULL},
    {"data", (getter)Tensor_getdata, NULL, "Data buffer as memoryview", NULL},
    {NULL} /* Sentinel */
};
