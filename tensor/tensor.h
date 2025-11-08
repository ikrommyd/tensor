/*
 * Tensor C Extension - Shared Header
 * Internal declarations for the tensor module implementation
 */

#ifndef TENSOR_H
#define TENSOR_H

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <structmember.h>

#include <stddef.h>

/* TensorObject struct definition */
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

/* External type reference - defined in tensor.c */
extern PyTypeObject TensorType;

/* Memory lifecycle functions - init.c */
PyObject *
Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int
Tensor_init(PyObject *op, PyObject *args, PyObject *kwds);
void
Tensor_dealloc(PyObject *op);
extern PyMemberDef Tensor_members[];

/* Getters and setters - getset.c */
PyObject *
Tensor_getshape(PyObject *op, void *closure);
PyObject *
Tensor_getstrides(PyObject *op, void *closure);
PyObject *
Tensor_getbase(PyObject *op, void *closure);
PyObject *
Tensor_getsize(PyObject *op, void *closure);
PyObject *
Tensor_getdata(PyObject *op, void *closure);
extern PyGetSetDef Tensor_getseters[];

/* Instance methods - methods.c */
PyObject *
Tensor_tolist(PyObject *op, PyObject *Py_UNUSED(ignored));
PyObject *
Tensor_item(PyObject *op, PyObject *Py_UNUSED(ignored));
PyObject *
Tensor_copy(PyObject *op, PyObject *Py_UNUSED(ignored));
PyObject *
Tensor_to_numpy(PyObject *op, PyObject *Py_UNUSED(ignored));
extern struct PyMethodDef Tensor_methods[];

/* String representations - repr.c */
PyObject *
Tensor_str(PyObject *op);
PyObject *
Tensor_repr(PyObject *op);

/* Mapping protocol - mapping.c */
Py_ssize_t
Tensor_length(PyObject *op);
PyObject *
Tensor_subscript(PyObject *op, PyObject *key);
int
Tensor_ass_subscript(PyObject *op, PyObject *key, PyObject *value);
extern PyMappingMethods Tensor_as_mapping;

/* Sequence protocol - sequence.c */
PyObject *
Tensor_sq_item(PyObject *self, Py_ssize_t index);
int
Tensor_sq_ass_item(PyObject *self, Py_ssize_t index, PyObject *value);
int
Tensor_contains(PyObject *op, PyObject *value);
PyObject *
Tensor_concat(PyObject *op, PyObject *other);
extern PySequenceMethods Tensor_as_sequence;

#endif /* TENSOR_H */
