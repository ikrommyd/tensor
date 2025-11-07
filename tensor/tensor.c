/*
This is a simple C extension module that defines a Tensor type for Python.
It's only a 0D or 1D tensor of double precision floats for now but we keep things
open for more dimensions in the future.
The goal is to have the most minimal implementation possible so that I can
learn the C API for Python extensions.
*/
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <stddef.h>/* for offsetof() */

#include <numpy/arrayobject.h>

// Define the Tensor object struct
typedef struct {
    PyObject_HEAD  // Required macro for all Python objects (contains refcount, type)
    char *data;  // Pointer to the start of the data buffer (char* for byte arithmetic)
    int nd;      // Number of dimensions (will be 0 or 1 for now)
    Py_ssize_t
        *dimensions;  // Array of dimension sizes (e.g., [5] for a 5-element tensor)
    Py_ssize_t *
        strides;  // Bytes to step in memory to reach next element in each dimension
                  // Example: stride[0]=16 means elements are 16 bytes apart
                  // This allows for views with custom strides (e.g., every 2nd element)
    PyObject
        *base;  // For views/slices - points to the tensor that owns the actual data
                // NULL means this tensor owns its data and must free it on dealloc
                // Non-NULL means this is a view - don't free data, just DECREF base
} TensorObject;

// Deallocator function - called when a Tensor object's reference count reaches 0
// IMPORTANT: This is where we clean up all resources owned by the tensor
static void
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
static PyObject *
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
static int
Tensor_init(PyObject *op, PyObject *args, PyObject *kwds)
{
    TensorObject *self = (TensorObject *)op;
    PyObject *input;

    // Parse the input argument (a number or sequence)
    if (!PyArg_ParseTuple(args, "O", &input)) {
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

// Direct member access - exposes struct fields as Python attributes
// offsetof() calculates the byte offset of a field within the struct
static PyMemberDef Tensor_members[] = {
    {"ndim", Py_T_INT, offsetof(TensorObject, nd), Py_READONLY, "number of dimensions"},
    {NULL}  // Sentinel - marks the end of the array (common C pattern)
};

// Getter for the 'shape' property - returns tuple of dimension sizes
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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

static PyGetSetDef Tensor_getseters[] = {
    {"shape", (getter)Tensor_getshape, NULL, "Shape of the tensor", NULL},
    {"strides", (getter)Tensor_getstrides, NULL, "Strides of the tensor", NULL},
    {"base", (getter)Tensor_getbase, NULL, "Base tensor if view", NULL},
    {"size", (getter)Tensor_getsize, NULL, "Total number of elements", NULL},
    {"data", (getter)Tensor_getdata, NULL, "Data buffer as memoryview", NULL},
    {NULL} /* Sentinel */
};

// Method to convert tensor to a Python list
// 0D tensor returns a scalar float, 1D tensor returns a list of floats
static PyObject *
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
static PyObject *
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
static PyObject *
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
static PyObject *
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

static struct PyMethodDef Tensor_methods[] = {
    {"tolist", (PyCFunction)Tensor_tolist, METH_NOARGS, "Convert tensor to a list"},
    {"item", (PyCFunction)Tensor_item, METH_NOARGS,
     "Get the single item from a 0D tensor as a python scalar"},
    {"copy", (PyCFunction)Tensor_copy, METH_NOARGS, "Return a copy of the tensor"},
    {"to_numpy", (PyCFunction)Tensor_to_numpy, METH_NOARGS,
     "Convert tensor to a NumPy array"},
    {NULL, NULL, 0, NULL}};

// String representation function - returns string like "Tensor([1.0, 2.0], shape=(2,))"
static PyObject *
Tensor_repr(PyObject *op)
{
    TensorObject *self = (TensorObject *)op;

    // Handle 0D tensor: "Tensor(5.0, shape=())"
    if (self->nd == 0) {
        double value = *((double *)self->data);
        PyObject *float_obj = PyFloat_FromDouble(value);
        if (float_obj == NULL) {
            return NULL;
        }
        PyObject *repr_str = PyUnicode_FromFormat("Tensor(%R, shape=())", float_obj);
        Py_DECREF(float_obj);
        return repr_str;
    }

    // Handle empty 1D tensor: "Tensor([], shape=(0,))"
    if (self->dimensions[0] == 0) {
        return PyUnicode_FromFormat("Tensor([], shape=(0,))");
    }

    // Build representation for non-empty 1D tensor
    PyObject *repr_str = PyUnicode_FromString("Tensor([");
    if (repr_str == NULL) {
        return NULL;
    }

    // Iterate through elements and build comma-separated list
    for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
        // Get element value (respects strides)
        char *data_ptr = self->data + (i * self->strides[0]);
        double value = *((double *)data_ptr);
        PyObject *float_obj = PyFloat_FromDouble(value);

        if (float_obj == NULL) {
            Py_DECREF(repr_str);
            return NULL;
        }

        // Convert to string representation
        PyObject *num_str = PyObject_Repr(float_obj);
        Py_DECREF(float_obj);
        if (num_str == NULL) {
            Py_DECREF(repr_str);
            return NULL;
        }

        // Append element to result string
        PyUnicode_Append(&repr_str, num_str);
        Py_DECREF(num_str);
        if (repr_str == NULL) {
            return NULL;
        }

        // Add comma separator between elements (but not after last element)
        if (i < self->dimensions[0] - 1) {
            PyObject *comma_str = PyUnicode_FromString(", ");
            if (comma_str == NULL) {
                Py_DECREF(repr_str);
                return NULL;
            }
            PyUnicode_Append(&repr_str, comma_str);
            Py_DECREF(comma_str);
            if (repr_str == NULL) {
                return NULL;
            }
        }
    }

    // Append "], shape=("
    PyObject *bracket_str = PyUnicode_FromString("], shape=(");
    if (bracket_str == NULL) {
        Py_DECREF(repr_str);
        return NULL;
    }
    PyUnicode_Append(&repr_str, bracket_str);
    Py_DECREF(bracket_str);
    if (repr_str == NULL) {
        return NULL;
    }

    // Append dimension size
    PyObject *shape_str = PyUnicode_FromFormat("%zd", self->dimensions[0]);
    if (shape_str == NULL) {
        Py_DECREF(repr_str);
        return NULL;
    }
    PyUnicode_Append(&repr_str, shape_str);
    Py_DECREF(shape_str);
    if (repr_str == NULL) {
        return NULL;
    }

    // Append ",))" to close
    PyObject *end_str = PyUnicode_FromString(",))");
    if (end_str == NULL) {
        Py_DECREF(repr_str);
        return NULL;
    }
    PyUnicode_Append(&repr_str, end_str);
    Py_DECREF(end_str);

    return repr_str;
}

// Returns the length of the tensor (for len() builtin)
// Only works for 1D tensors (0D tensors have no length)
static Py_ssize_t
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
static PyObject *
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

static PyTypeObject TensorType;

// Subscript setter - handles assignment to indices (x[i] = val) and slices (x[1:5] =
// val)
static int
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
static PyMappingMethods Tensor_as_mapping = {
    .mp_length = Tensor_length,        // Called for len(tensor)
    .mp_subscript = Tensor_subscript,  // Called for tensor[key] (reading)
    .mp_ass_subscript =
        Tensor_ass_subscript,  // Called for tensor[key] = value (writing)
};

// Type object definition - this is the "class" definition in C
// Think of this as the equivalent of "class Tensor:" in Python
static PyTypeObject TensorType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "tensor.Tensor",
    .tp_doc = PyDoc_STR("Tensor object"),
    .tp_basicsize = sizeof(TensorObject),  // Memory size of each instance
    .tp_itemsize = 0,  // For variable-size objects (we don't use this)
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // Can be subclassed
    .tp_new = Tensor_new,                                  // Called by Tensor.__new__()
    .tp_init = (initproc)Tensor_init,     // Called by Tensor.__init__()
    .tp_dealloc = Tensor_dealloc,         // Called when refcount reaches 0
    .tp_members = Tensor_members,         // Direct member access (like .ndim)
    .tp_methods = Tensor_methods,         // Methods like .tolist(), .copy()
    .tp_getset = Tensor_getseters,        // Properties like .shape, .strides
    .tp_repr = (reprfunc)Tensor_repr,     // Called by repr() and in REPL
    .tp_as_mapping = &Tensor_as_mapping,  // Enables indexing/slicing behavior
};

// Module-level tensor() function - alternative constructor with copy parameter
// Usage: tensor.tensor([1, 2, 3]) or tensor.tensor(existing_tensor, copy=True)
static PyObject *
tensor_tensor(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int copy = 0;

    static char *kwlist[] = {"object", "copy", NULL};

    // Parse arguments: required object, optional copy keyword argument
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$p", kwlist, &obj, &copy)) {
        return NULL;
    }

    // If input is already a Tensor, either copy it or return it as-is
    if (PyObject_TypeCheck(obj, &TensorType)) {
        if (copy) {
            return Tensor_copy(obj, NULL);
        }
        else {
            Py_INCREF(obj);
            return obj;
        }
    }

    // Create new tensor from other object (number or sequence)
    PyObject *tensor_obj = Tensor_new(&TensorType, NULL, NULL);
    if (tensor_obj == NULL) {
        return NULL;
    }

    PyObject *init_args = PyTuple_Pack(1, obj);
    if (init_args == NULL) {
        Py_DECREF(tensor_obj);
        return NULL;
    }

    if (Tensor_init(tensor_obj, init_args, NULL) < 0) {
        Py_DECREF(init_args);
        Py_DECREF(tensor_obj);
        return NULL;
    }

    Py_DECREF(init_args);
    return tensor_obj;
}

static PyMethodDef tensor_module_methods[] = {
    {"tensor", (PyCFunction)tensor_tensor, METH_VARARGS | METH_KEYWORDS,
     "Create a Tensor from a number or sequence"},
    {NULL, NULL, 0, NULL}};

// Module execution function - called when the module is first imported
// Initializes the TensorType and adds it to the module namespace
static int
tensor_module_exec(PyObject *m)
{
    // Finalize the TensorType (computes method resolution order, etc.)
    if (PyType_Ready(&TensorType) < 0) {
        return -1;
    }

    // Add the Tensor class to the module (accessible as tensor.Tensor)
    if (PyModule_AddObjectRef(m, "Tensor", (PyObject *)&TensorType) < 0) {
        return -1;
    }

    return 0;
}

// Module slots - define how the module is initialized
// This is the modern Python 3.5+ multi-phase initialization approach
static PyModuleDef_Slot tensor_module_slots[] = {
    {Py_mod_exec, tensor_module_exec},  // Function to execute during module creation
    {Py_mod_multiple_interpreters,
     Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},  // We don't support sub-interpreters
    {0, NULL}                                      // Sentinel
};

// Module definition - describes the module to Python
static PyModuleDef tensor_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "tensor",                  // Module name (what you import)
    .m_doc = "A simple Tensor module",   // Module docstring
    .m_size = 0,                         // Module state size (0 = no per-module state)
    .m_methods = tensor_module_methods,  // Module-level functions
    .m_slots = tensor_module_slots,      // Initialization slots
};

// Module initialization function - entry point when Python imports the module
// This is the function that Python calls when you do "import tensor"
// PyMODINIT_FUNC is a macro that expands to the right return type and linkage
PyMODINIT_FUNC
PyInit_tensor(void)
{
    // Import NumPy C API (required for to_numpy() method)
    // This populates function pointers for PyArray_SimpleNew, PyArray_DATA, etc.
    // Must be called before using any NumPy C API functions
    // Returns -1 on error (e.g., if NumPy is not installed)
    if (PyArray_ImportNumPyAPI() < 0) {
        return NULL;  // Python exception is already set
    }

    // Initialize and return the module using the multi-phase init mechanism
    // This calls tensor_module_exec (via slots) which finalizes TensorType
    return PyModuleDef_Init(&tensor_module);
}
