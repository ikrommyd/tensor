/*
 * Tensor mapping protocol (indexing and slicing)
 */

#include "tensor.h"

// Returns the length of the tensor (for len() builtin)
// Only works for 1D tensors (0D tensors have no length)
Py_ssize_t
Tensor_length(PyObject *op)
{
    TensorObject *self = (TensorObject *)op;
    if (self->nd == 0) {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object");
        return -1;
    }
    return self->dimensions[0];
}

// Subscript getter - handles indexing (x[i]) and slicing (x[start:stop:step])
PyObject *
Tensor_subscript(PyObject *op, PyObject *key)
{
    TensorObject *self = (TensorObject *)op;

    // 0D tensors cannot be indexed
    if (self->nd == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Tensor is 0-dimensional and cannot be indexed");
        return NULL;
    }

    // Case 1: Integer indexing (e.g., x[3])
    if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        Py_ssize_t new_index;
        if (index == -1 && PyErr_Occurred()) {
            return NULL;
        }

        // Handle negative indices (e.g., x[-1] is last element)
        if (index < 0) {
            new_index = self->dimensions[0] + index;
        }
        else {
            new_index = index;
        }

        // Check bounds
        if (new_index < 0 || new_index >= self->dimensions[0]) {
            PyErr_Format(PyExc_IndexError,
                         "Index %zd out of range for tensor of size %zd", index,
                         self->dimensions[0]);
            return NULL;
        }

        // Create a 0D tensor containing the indexed element
        // Note: We COPY the value rather than creating a view
        // This is consistent with NumPy behavior - indexing reduces dimensionality
        TensorObject *scalar = (TensorObject *)Tensor_new(Py_TYPE(self), NULL, NULL);
        if (scalar == NULL) {
            return NULL;
        }

        scalar->nd = 0;

        // Allocate empty dimension/stride arrays for 0D tensor
        scalar->dimensions = PyMem_Malloc(0);
        scalar->strides = PyMem_Malloc(0);
        if (scalar->dimensions == NULL || scalar->strides == NULL) {
            PyErr_NoMemory();
            Py_DECREF(scalar);
            return NULL;
        }

        // Allocate memory for the single value and copy it
        scalar->data = PyMem_Malloc(sizeof(double));
        if (scalar->data == NULL) {
            PyErr_NoMemory();
            Py_DECREF(scalar);
            return NULL;
        }

        // Calculate address of the indexed element using stride arithmetic
        // Then copy the value to the new scalar tensor
        char *data_ptr = self->data + (new_index * self->strides[0]);
        *((double *)scalar->data) = *((double *)data_ptr);

        scalar->base = NULL;  // Scalar owns its own copy of the data
        return (PyObject *)scalar;
    }
    // Case 2: Slice indexing (e.g., x[1:5:2])
    else if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength;

        // Parse the slice object to get start, stop, step
        // PySlice_Unpack handles None values and returns -1 on error
        // Example: x[:5] becomes start=0 (from None), stop=5, step=1 (from None)
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) {
            return NULL;
        }

        // Adjust indices to handle negative values, clamp to bounds, and compute length
        // Example: x[-3:] with length 5 becomes start=2, stop=5
        // Also handles step < 0 for reverse slicing (though rare)
        // Returns the number of elements in the slice
        slicelength = PySlice_AdjustIndices(self->dimensions[0], &start, &stop, step);

        // Create a view tensor that shares data with the original
        TensorObject *view = (TensorObject *)Tensor_new(Py_TYPE(self), NULL, NULL);
        if (view == NULL) {
            return NULL;
        }

        view->nd = 1;

        view->dimensions = PyMem_Malloc(sizeof(Py_ssize_t) * view->nd);
        if (view->dimensions == NULL) {
            PyErr_NoMemory();
            Py_DECREF(view);
            return NULL;
        }

        view->strides = PyMem_Malloc(sizeof(Py_ssize_t) * view->nd);
        if (view->strides == NULL) {
            PyErr_NoMemory();
            Py_DECREF(view);
            return NULL;
        }

        view->dimensions[0] = slicelength;

        if (slicelength == 0) {
            // Empty slice - still points to parent data but with zero length
            view->strides[0] = self->strides[0];
            view->data = self->data;
        }
        else {
            // Non-empty slice - adjust stride by step and offset data pointer
            // Example: x = [0,1,2,3,4,5], x[1:5:2] selects [1,3]
            // Original stride: 8 bytes, step: 2, so new stride: 16 bytes
            // This makes view skip the right number of elements
            view->strides[0] = self->strides[0] * step;

            // Offset the data pointer to start at the first element of the slice
            // start * stride gives us byte offset to the start element
            view->data = self->data + (start * self->strides[0]);
        }

        // Set base to track which tensor owns the data
        // This is critical for memory management!
        if (self->base != NULL) {
            // self is already a view, so point to the ultimate owner
            view->base = self->base;
        }
        else {
            // self owns its data, so it becomes the base for this view
            view->base = (PyObject *)self;
        }
        // Increment reference count - view now holds a reference to base
        // This prevents base from being deallocated while view exists
        Py_INCREF(view->base);

        return (PyObject *)view;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
        return NULL;
    }
}

// Subscript setter - handles assignment to indices (x[i] = val) and slices (x[1:5] =
// val)
int
Tensor_ass_subscript(PyObject *op, PyObject *key, PyObject *value)
{
    TensorObject *self = (TensorObject *)op;

    // Deletion not supported
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot delete tensor elements");
        return -1;
    }

    // 0D tensors cannot be indexed
    if (self->nd == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Tensor is 0-dimensional and cannot be indexed");
        return -1;
    }

    // Convert value to a tensor if it isn't already
    // This allows x[0] = 5.0 and x[0] = Tensor(5.0) to both work
    TensorObject *value_tensor;

    if (PyObject_TypeCheck(value, &TensorType)) {
        // Value is already a Tensor - just use it
        value_tensor = (TensorObject *)value;
        // IMPORTANT: We need to increment the reference count because we'll DECREF
        // later Without this, we might deallocate an object the caller still needs
        Py_INCREF(value_tensor);
    }
    else {
        // Value is a number or sequence - convert to Tensor
        // Create a new tensor from the value
        value_tensor = (TensorObject *)Tensor_new(&TensorType, NULL, NULL);
        if (value_tensor == NULL) {
            return -1;
        }

        // Pack value into a tuple for Tensor_init (it expects args tuple)
        PyObject *init_args = PyTuple_Pack(1, value);
        if (init_args == NULL) {
            Py_DECREF(value_tensor);
            return -1;
        }

        // Initialize the tensor with the value
        if (Tensor_init((PyObject *)value_tensor, init_args, NULL) < 0) {
            Py_DECREF(init_args);
            Py_DECREF(value_tensor);
            return -1;
        }

        Py_DECREF(init_args);
    }

    // Case 1: Integer indexing assignment (x[i] = value)
    if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        Py_ssize_t new_index;
        if (index == -1 && PyErr_Occurred()) {
            Py_DECREF(value_tensor);
            return -1;
        }

        // Handle negative indices
        if (index < 0) {
            new_index = self->dimensions[0] + index;
        }
        else {
            new_index = index;
        }

        // Check bounds
        if (new_index < 0 || new_index >= self->dimensions[0]) {
            PyErr_Format(PyExc_IndexError,
                         "Index %zd out of range for tensor of size %zd", index,
                         self->dimensions[0]);
            Py_DECREF(value_tensor);
            return -1;
        }

        // Value must be a scalar (0D tensor)
        if (value_tensor->nd != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "setting a tensor element with a sequence");
            Py_DECREF(value_tensor);
            return -1;
        }

        // Extract scalar value and assign it
        double scalar_value;
        scalar_value = *((double *)value_tensor->data);
        char *data_ptr = self->data + (new_index * self->strides[0]);
        *((double *)data_ptr) = scalar_value;

        Py_DECREF(value_tensor);
        return 0;
    }
    // Case 2: Slice assignment (x[1:5] = value)
    else if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength;

        // Parse slice
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) {
            Py_DECREF(value_tensor);
            return -1;
        }

        slicelength = PySlice_AdjustIndices(self->dimensions[0], &start, &stop, step);

        // Case 2a: Scalar broadcast - assign single value to all elements in slice
        // This handles x[1:5] = 3.0 or x[1:5] = Tensor([3.0])
        if (value_tensor->nd == 0 ||
            (value_tensor->nd == 1 && value_tensor->dimensions[0] == 1)) {
            double scalar_value = *((double *)value_tensor->data);

            // Broadcast scalar to all elements in the slice
            // Example: x[1:4] = 5.0 sets x[1], x[2], x[3] all to 5.0
            for (Py_ssize_t i = 0; i < slicelength; i++) {
                // Calculate actual index in the original tensor
                // For x[start:stop:step], the i-th element is at index (start + i*step)
                Py_ssize_t idx = start + i * step;
                // Use stride to get the memory address
                char *data_ptr = self->data + (idx * self->strides[0]);
                *((double *)data_ptr) = scalar_value;
            }
        }
        // Case 2b: Element-wise assignment - shapes must match
        // This handles x[1:5] = Tensor([a, b, c, d]) where source has same length
        else if (value_tensor->nd == 1 && value_tensor->dimensions[0] == slicelength) {
            for (Py_ssize_t i = 0; i < slicelength; i++) {
                Py_ssize_t idx = start + i * step;
                char *data_ptr = self->data + (idx * self->strides[0]);
                // Get i-th value from value_tensor (also using its stride)
                char *value_ptr = value_tensor->data + (i * value_tensor->strides[0]);
                double val = *((double *)value_ptr);
                *((double *)data_ptr) = val;
            }
        }
        // Case 2c: Shape mismatch - error
        else {
            PyErr_Format(
                PyExc_ValueError,
                "could not broadcast input tensor from shape (%zd,) into shape (%zd,)",
                value_tensor->dimensions[0], slicelength);
            Py_DECREF(value_tensor);
            return -1;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
        Py_DECREF(value_tensor);
        return -1;
    }

    Py_DECREF(value_tensor);
    return 0;
}

// Mapping protocol - enables x[i], x[1:5], len(x), etc.
PyMappingMethods Tensor_as_mapping = {
    .mp_length = Tensor_length,        // Called for len(tensor)
    .mp_subscript = Tensor_subscript,  // Called for tensor[key] (reading)
    .mp_ass_subscript =
        Tensor_ass_subscript,  // Called for tensor[key] = value (writing)
};
