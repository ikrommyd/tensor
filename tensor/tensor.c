/*
 * Tensor C Extension Module
 * Main type definition and module initialization
 */

#define PY_ARRAY_UNIQUE_SYMBOL TENSOR_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "tensor.h"

#include <numpy/arrayobject.h>

// Type object definition - this is the "class" definition in C
// Think of this as the equivalent of "class Tensor:" in Python
PyTypeObject TensorType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "tensor.Tensor",
    .tp_doc = PyDoc_STR("Tensor object"),
    .tp_basicsize = sizeof(TensorObject),  // Memory size of each instance
    .tp_itemsize = 0,  // For variable-size objects (we don't use this)
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // Can be subclassed
    .tp_new = Tensor_new,                                  // Called by Tensor.__new__()
    .tp_init = (initproc)Tensor_init,       // Called by Tensor.__init__()
    .tp_dealloc = Tensor_dealloc,           // Called when refcount reaches 0
    .tp_members = Tensor_members,           // Direct member access (like .ndim)
    .tp_methods = Tensor_methods,           // Methods like .tolist(), .copy()
    .tp_getset = Tensor_getseters,          // Properties like .shape, .strides
    .tp_repr = (reprfunc)Tensor_repr,       // Called by repr() and in REPL
    .tp_str = (reprfunc)Tensor_str,         // Called by str() and print()
    .tp_as_mapping = &Tensor_as_mapping,    // Enables indexing/slicing behavior
    .tp_as_sequence = &Tensor_as_sequence,  // Enables sequence behavior
};

// Module-level tensor() function - alternative constructor with copy parameter
// Usage: tensor.tensor([1, 2, 3]) or tensor.tensor(existing_tensor, copy=True)
static PyObject *
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
#if PY_VERSION_HEX >= 0x030c00f0        // Python 3.12+
    {Py_mod_multiple_interpreters,
     Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},  // We don't support sub-interpreters
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}  // Sentinel
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
    // Returns NULL on error (e.g., if NumPy is not installed)
    import_array();
    if (PyErr_Occurred()) {
        return NULL;  // Python exception is already set
    }

    // Initialize and return the module using the multi-phase init mechanism
    // This calls tensor_module_exec (via slots) which finalizes TensorType
    return PyModuleDef_Init(&tensor_module);
}
