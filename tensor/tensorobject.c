/*
 * Tensor Type Definition
 * Defines the TensorType PyTypeObject ("class" in C)
 */

#include "tensor.h"

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
