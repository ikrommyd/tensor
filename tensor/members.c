/*
 * Tensor members (direct field access)
 */

#include "tensor.h"

// Direct member access - exposes struct fields as Python attributes
// offsetof() calculates the byte offset of a field within the struct
PyMemberDef Tensor_members[] = {
    {"ndim", Py_T_INT, offsetof(TensorObject, nd), Py_READONLY, "number of dimensions"},
    {NULL}  // Sentinel - marks the end of the array (common C pattern)
};
