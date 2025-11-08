/*
 * Tensor instance methods
 */

#define PY_ARRAY_UNIQUE_SYMBOL TENSOR_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "tensor.h"

#include <numpy/arrayobject.h>

// Method to convert tensor to a Python list
// 0D tensor returns a scalar float, 1D tensor returns a list of floats
PyObject *
Tensor_tolist(PyObject *op, PyObject *Py_UNUSED(ignored))
{
    TensorObject *self = (TensorObject *)op;

    // For 0D tensor, return scalar value directly
    if (self->nd == 0) {
        double value = *((double *)self->data);
        return PyFloat_FromDouble(value);
    }

    // For 1D tensor, create a list and populate it
    PyObject *list = PyList_New(self->dimensions[0]);
    if (list == NULL) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
        // Use stride to correctly access element (handles views with non-standard
        // strides) self->data is char* (byte pointer) for pointer arithmetic
        // self->strides[0] is bytes between consecutive elements (usually 8 for
        // doubles) For a view like x[::2], stride would be 16 (skip every other
        // element) The arithmetic: data + (i * stride) gives us the address of the i-th
        // element
        char *data_ptr = self->data + (i * self->strides[0]);
        // Cast char* to double* and dereference to get the actual value
        double value = *((double *)data_ptr);
        PyObject *float_obj = PyFloat_FromDouble(value);
        if (float_obj == NULL) {
            Py_DECREF(list);  // Clean up list on error
            return NULL;
        }
        // PyList_SET_ITEM steals the reference to float_obj (no need to DECREF)
        PyList_SET_ITEM(list, i, float_obj);
    }

    return list;
}

// Method to extract a scalar value from a size-1 tensor
// Works for 0D tensors or 1D tensors with one element
PyObject *
Tensor_item(PyObject *op, PyObject *Py_UNUSED(ignored))
{
    TensorObject *self = (TensorObject *)op;
    if (self->nd == 0) {
        // 0D tensor - return the single value
        double value = *((double *)self->data);
        return PyFloat_FromDouble(value);
    }
    else if (self->nd == 1 && self->dimensions[0] == 1) {
        // 1D tensor with one element - return that element
        char *data_ptr = self->data;
        double value = *((double *)data_ptr);
        return PyFloat_FromDouble(value);
    }
    else {
        // Error for tensors with more than one element
        PyErr_SetString(PyExc_ValueError,
                        "can only convert a tensor of size 1 to a Python scalar");
        return NULL;
    }
}

// Method to create a deep copy of the tensor
// The copy owns its own data (base is NULL) and uses contiguous strides
PyObject *
Tensor_copy(PyObject *op, PyObject *Py_UNUSED(ignored))
{
    TensorObject *self = (TensorObject *)op;

    // Create new tensor object
    TensorObject *copy = (TensorObject *)Tensor_new(Py_TYPE(self), NULL, NULL);
    if (copy == NULL) {
        return NULL;
    }

    copy->nd = self->nd;

    // Allocate dimensions array
    copy->dimensions = PyMem_Malloc(sizeof(Py_ssize_t) * copy->nd);
    if (copy->dimensions == NULL) {
        PyErr_NoMemory();
        Py_DECREF(copy);
        return NULL;
    }

    // Allocate strides array
    copy->strides = PyMem_Malloc(sizeof(Py_ssize_t) * copy->nd);
    if (copy->strides == NULL) {
        PyErr_NoMemory();
        Py_DECREF(copy);
        return NULL;
    }

    if (copy->nd == 0) {
        // Copy 0D tensor
        copy->data = PyMem_Malloc(sizeof(double));
        if (copy->data == NULL) {
            PyErr_NoMemory();
            Py_DECREF(copy);
            return NULL;
        }
        *((double *)copy->data) = *((double *)self->data);
    }
    else {
        // Copy 1D tensor
        copy->dimensions[0] = self->dimensions[0];

        if (self->dimensions[0] == 0) {
            // Empty tensor
            copy->strides[0] = 0;
            copy->data = NULL;
        }
        else {
            // Use standard contiguous stride (always sizeof(double))
            // Even if source was a view with non-standard stride, copy is contiguous
            copy->strides[0] = sizeof(double);
            Py_ssize_t data_size = self->dimensions[0] * sizeof(double);
            copy->data = PyMem_Malloc(data_size);
            if (copy->data == NULL) {
                PyErr_NoMemory();
                Py_DECREF(copy);
                return NULL;
            }

            // Copy data element by element (respects source strides)
            // This is important: source might be a view with stride != 8
            // Example: if self is x[::2], we need to read every other element
            // But we write to contiguous memory in the copy
            double *dst = (double *)copy->data;
            for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
                char *src_ptr = self->data + (i * self->strides[0]);
                dst[i] = *((double *)src_ptr);
            }
        }
    }

    copy->base = NULL;  // Copy owns its own data (not a view)
    return (PyObject *)copy;
}

// Method to convert tensor to a NumPy array
// Creates a new NumPy array and copies the data
PyObject *
Tensor_to_numpy(PyObject *op, PyObject *Py_UNUSED(ignored))
{
    TensorObject *self = (TensorObject *)op;

    npy_intp dims[1];
    int ndim;

    if (self->nd == 0) {
        // 0D tensor - create 0D NumPy array
        ndim = 0;
        dims[0] = 0;  // Not used for 0D
    }
    else {
        // 1D tensor
        ndim = 1;
        dims[0] = self->dimensions[0];
    }

    // Create a new NumPy array with contiguous memory layout
    PyObject *numpy_array = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    if (numpy_array == NULL) {
        return NULL;
    }

    // Get pointer to NumPy array data
    double *numpy_data = (double *)PyArray_DATA((PyArrayObject *)numpy_array);

    // Copy data from tensor to numpy array
    if (self->nd == 0) {
        // Copy single scalar
        numpy_data[0] = *((double *)self->data);
    }
    else {
        // Copy 1D array element by element (respects tensor strides)
        for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
            char *src_ptr = self->data + (i * self->strides[0]);
            numpy_data[i] = *((double *)src_ptr);
        }
    }

    return numpy_array;
}

struct PyMethodDef Tensor_methods[] = {
    {"tolist", (PyCFunction)Tensor_tolist, METH_NOARGS, "Convert tensor to a list"},
    {"item", (PyCFunction)Tensor_item, METH_NOARGS,
     "Get the single item from a 0D tensor as a python scalar"},
    {"copy", (PyCFunction)Tensor_copy, METH_NOARGS, "Return a copy of the tensor"},
    {"to_numpy", (PyCFunction)Tensor_to_numpy, METH_NOARGS,
     "Convert tensor to a NumPy array"},
    {NULL, NULL, 0, NULL}};
