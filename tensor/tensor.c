#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef_Slot tensor_slots[] = {
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "tensor",
    .m_size = 0,
    .m_methods = methods,
    .m_slots = tensor_slots,
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit_tensor(void) {
    return PyModuleDef_Init(&moduledef);
}
