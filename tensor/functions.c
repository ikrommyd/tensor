/*
 * Module-level Functions
 * Contains module-level constructor functions like tensor.tensor()
 */

#include "tensor.h"

// Module-level tensor() function - alternative constructor with copy parameter
// Usage: tensor.tensor([1, 2, 3]) or tensor.tensor(existing_tensor, copy=True)
PyObject *
tensor_tensor(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int copy = 0;

    static char *kwlist[] = {"data", "copy", NULL};

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
