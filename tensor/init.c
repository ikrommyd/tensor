/*
 * Tensor initialization and memory lifecycle
 */

#include "tensor.h"

// Deallocator function - called when a Tensor object's reference count reaches 0
// IMPORTANT: This is where we clean up all resources owned by the tensor
void
Tensor_dealloc(PyObject *op)
{
    TensorObject *self = (TensorObject *)op;

    // Always free dimensions and strides arrays - these are always owned by the tensor
    PyMem_Free(self->dimensions);
    PyMem_Free(self->strides);

    // Only free data if this tensor owns it (base == NULL)
    // Otherwise, decrement reference to the base tensor that owns the data
    if (self->base == NULL) {
        // This tensor owns its data - we allocated it, so we must free it
        PyMem_Free(self->data);
    }
    else {
        // This tensor is a view/slice of another tensor
        // The data belongs to self->base, so we just release our reference to it
        // When base's refcount reaches 0, it will free the data
        // Py_DECREF decrements the reference count and may trigger base's dealloc
        Py_DECREF(self->base);
    }

    // Free the tensor object itself using the type's free function
    Py_TYPE(self)->tp_free(op);
}

// Allocator function - creates a new Tensor object with default values
// This is part of Python's two-phase object creation: __new__ then __init__
PyObject *
Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    TensorObject *self;
    self = (TensorObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        // Initialize all fields to safe default values
        self->data = NULL;
        self->nd = 0;
        self->dimensions = NULL;
        self->strides = NULL;
        self->base = NULL;
    }
    return (PyObject *)self;
}

// Initializer function - populates a Tensor with data from a Python object
// Handles two cases: scalar (0D tensor) and sequence (1D tensor)
int
Tensor_init(PyObject *op, PyObject *args, PyObject *kwds)
{
    TensorObject *self = (TensorObject *)op;
    PyObject *input;

    // Include "object" as the positional argument name
    static char *kwlist[] = {"data", NULL};

    // Parse exactly one positional argument, reject any keywords
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:Tensor", kwlist, &input)) {
        return -1;
    }

    // Case 1: Input is a scalar number (creates 0D tensor)
    // PyNumber_Check returns true for numbers, PySequence_Check returns false
    // Note: strings are sequences, so we check !PySequence_Check to exclude them
    if (PyNumber_Check(input) && !PySequence_Check(input)) {
        self->nd = 0;

        // Allocate empty arrays for dimensions and strides (0D has no dimensions)
        // PyMem_Malloc(0) is allowed and returns a valid pointer (though not for
        // dereferencing) This keeps the code consistent - we can always free
        // dimensions/strides
        self->dimensions = PyMem_Malloc(0);
        self->strides = PyMem_Malloc(0);
        if (self->dimensions == NULL || self->strides == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        // Allocate space for a single double value
        self->data = PyMem_Malloc(sizeof(double));
        if (self->data == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        // Convert input to float and extract the double value
        // PyNumber_Float returns a NEW reference - we own it and must DECREF it
        PyObject *float_obj = PyNumber_Float(input);
        if (float_obj == NULL) {
            return -1;  // PyNumber_Float already set an exception
        }

        // Extract the C double value from the Python float object
        double value = PyFloat_AsDouble(float_obj);
        // We're done with float_obj - decrement its reference count
        // This will likely deallocate it since we were the only owner
        Py_DECREF(float_obj);

        // PyFloat_AsDouble returns -1.0 on error, but -1.0 is also a valid value
        // So we need to check if an exception was set to distinguish
        if (value == -1.0 && PyErr_Occurred()) {
            return -1;
        }

        // Store the scalar value by casting char* to double* and dereferencing
        *((double *)self->data) = value;
        self->base = NULL;  // We own the data
        return 0;
    }

    // Validate that input is a sequence (list, tuple, etc.)
    if (!PySequence_Check(input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a number or a sequence of numbers");
        return -1;
    }

    // Case 2: Input is a sequence (creates 1D tensor)
    Py_ssize_t length = PySequence_Length(input);
    if (length < 0) {
        return -1;
    }

    self->nd = 1;

    // Allocate dimensions array with one element
    self->dimensions = PyMem_Malloc(sizeof(Py_ssize_t) * self->nd);
    if (self->dimensions == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    // Allocate strides array with one element
    self->strides = PyMem_Malloc(sizeof(Py_ssize_t) * self->nd);
    if (self->strides == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    self->dimensions[0] = length;

    // Handle empty sequence special case
    if (length == 0) {
        self->strides[0] = 0;
        self->data = NULL;
        self->base = NULL;
        return 0;
    }

    // Stride is the size of one double
    self->strides[0] = sizeof(double);

    // Allocate data buffer for all elements
    self->data = PyMem_Malloc(length * self->strides[0]);
    if (self->data == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    double *data_ptr = (double *)self->data;

    // Iterate through the sequence and convert each element to double
    for (Py_ssize_t i = 0; i < length; i++) {
        // PySequence_GetItem returns a NEW reference to the item
        PyObject *item = PySequence_GetItem(input, i);

        if (item == NULL) {
            PyErr_SetString(PyExc_TypeError, "Failed to get item from sequence");
            return -1;
        }

        // Ensure each item is a number
        if (!PyNumber_Check(item)) {
            Py_DECREF(item);  // Clean up the reference we own before returning error
            PyErr_SetString(PyExc_TypeError, "All elements must be numbers");
            return -1;
        }

        // Convert to float - returns NEW reference
        PyObject *float_item = PyNumber_Float(item);
        Py_DECREF(item);  // Done with original item

        if (float_item == NULL) {
            return -1;
        }

        // Extract double value
        double value = PyFloat_AsDouble(float_item);
        Py_DECREF(float_item);  // Done with float object

        if (value == -1.0 && PyErr_Occurred()) {
            return -1;
        }

        // Store in data buffer at index i
        // data_ptr is double*, so data_ptr[i] is the i-th double
        data_ptr[i] = value;
    }

    self->base = NULL;  // This tensor owns its data
    return 0;
}
