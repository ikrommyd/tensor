/*
This is a simple C extension module that defines a Tensor type for Python.
It's only a 1D tensor of single precision floats for now but we keep things
open for more dimensions in the future.
The goal is to have the most minimal implementation possbile so that I can
learn the C API for Python extensions.
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h> /* for offsetof() */


// Define the Tensor object struct
typedef struct {
    PyObject_HEAD
    char *data;              // Pointer to the start of the data buffer
    int nd;                  // Number of dimensions (will be 1 for now)
    Py_ssize_t *dimensions;  // Array of dimension sizes
    Py_ssize_t *strides;     // Bytes to next element in each dimension
    PyObject *base;          // For views/slices - NULL if owner
} TensorObject;

static int
Tensor_traverse(PyObject *op, visitproc visit, void *arg)
{
    TensorObject *self = (TensorObject *) op;
    Py_VISIT(self->base);
    return 0;
}

static int
Tensor_clear(PyObject *op)
{
    TensorObject *self = (TensorObject *) op;
    Py_CLEAR(self->base);
    return 0;
}

static void
Tensor_dealloc(PyObject *op)
{
    TensorObject *self = (TensorObject *) op;
    PyMem_Free(self->dimensions);
    PyMem_Free(self->strides);
    if (self->base == NULL) {
        PyMem_Free(self->data);
    }
    PyObject_GC_UnTrack(op);
    (void)Tensor_clear(op);
    Py_TYPE(op)->tp_free(op);
}

static PyObject *
Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    TensorObject *self;
    self = (TensorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->data = NULL;
        self->nd = 0;
        self->dimensions = NULL;
        self->strides = NULL;
        self->base = NULL;
    }
    return (PyObject *) self;
}

static int
Tensor_init(PyObject *op, PyObject *args, PyObject *kwds)
{
    TensorObject *self = (TensorObject *) op;
    PyObject *input_list;

    if (!PyArg_ParseTuple(args, "O", &input_list)) {
        return -1;
    }

    if (!PySequence_Check(input_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence (list or tuple)");
        return -1;
    }

    Py_ssize_t length = PySequence_Length(input_list);
    if (length < 0) {
        return -1;
    }

    self->nd = 1;

    self->dimensions = PyMem_Malloc(sizeof(Py_ssize_t) * self->nd);
    if (self->dimensions == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    self->strides = PyMem_Malloc(sizeof(Py_ssize_t) * self->nd);
    if (self->strides == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    self->dimensions[0] = length;

    if (length == 0) {
        self->strides[0] = 0;
        self->data = NULL;
        self->base = NULL;
        return 0;
    }

    self->strides[0] = sizeof(double);

    self->data = PyMem_Malloc(length * self->strides[0]);
    if (self->data == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    double *data_ptr = (double *)self->data;

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *item = PySequence_GetItem(input_list, i);

        if (item == NULL) {
            PyErr_SetString(PyExc_TypeError, "Failed to get item from sequence");
            return -1;
        }

        if (!PyNumber_Check(item)) {
            Py_DECREF(item);
            PyErr_SetString(PyExc_TypeError, "All elements must be numbers");
            return -1;
        }

        PyObject *float_item = PyNumber_Float(item);
        Py_DECREF(item);

        if (float_item == NULL) {
            return -1;
        }

        double value = PyFloat_AsDouble(float_item);
        Py_DECREF(float_item);

        if (value == -1.0 && PyErr_Occurred()) {
            return -1;
        }

        data_ptr[i] = value;
    }

    self->base = NULL;
    return 0;
}

static PyMemberDef Tensor_members[] = {
    {"ndim", Py_T_INT, offsetof(TensorObject, nd), Py_READONLY, "number of dimensions"},
    {NULL}  /* Sentinel */
};

static PyObject *
Tensor_getdata(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *) op;
    if (self->data) {
        return PyMemoryView_FromMemory(self->data,
                                       self->dimensions[0] * self->strides[0],
                                       PyBUF_READ);
    } else {
        Py_RETURN_NONE;
    }
}

static PyObject *
Tensor_getshape(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *) op;
    PyObject *shape_tuple = PyTuple_New(self->nd);
    if (shape_tuple == NULL) {
        return NULL;
    }
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

static PyObject *
Tensor_getstrides(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *) op;
    PyObject *strides_tuple = PyTuple_New(self->nd);
    if (strides_tuple == NULL) {
        return NULL;
    }
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

static PyObject *
Tensor_getbase(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *) op;
    if (self->base) {
        Py_INCREF(self->base);
        return self->base;
    } else {
        Py_RETURN_NONE;
    }
}

static PyObject *
Tensor_getsize(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *) op;
    Py_ssize_t size = 1;
    for (int i = 0; i < self->nd; i++) {
        size *= self->dimensions[i];
    }
    return PyLong_FromSsize_t(size);
}

static PyGetSetDef Tensor_getseters[] = {
    {"data", (getter)Tensor_getdata, NULL, "Data buffer as memoryview", NULL},
    {"shape", (getter)Tensor_getshape, NULL, "Shape of the tensor", NULL},
    {"strides", (getter)Tensor_getstrides, NULL, "Strides of the tensor", NULL},
    {"base", (getter)Tensor_getbase, NULL, "Base tensor if view", NULL},
    {"size", (getter)Tensor_getsize, NULL, "Total number of elements", NULL},
    {NULL}  /* Sentinel */
};

static struct PyMethodDef Tensor_methods[] = {
    {NULL, NULL, 0, NULL}
};

static PyObject *
Tensor_repr(PyObject *op)
{
    TensorObject *self = (TensorObject *) op;

    if (self->dimensions[0] == 0) {
        return PyUnicode_FromFormat("Tensor([], shape=(0,))");
    }

    PyObject *repr_str = PyUnicode_FromString("Tensor([");
    if (repr_str == NULL) {
        return NULL;
    }
    
    double *data_ptr = (double *)self->data;
    for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
        // Convert double to Python float, then to string
        PyObject *float_obj = PyFloat_FromDouble(data_ptr[i]);
        if (float_obj == NULL) {
            Py_DECREF(repr_str);
            return NULL;
        }
        
        PyObject *num_str = PyObject_Repr(float_obj);
        Py_DECREF(float_obj);
        if (num_str == NULL) {
            Py_DECREF(repr_str);
            return NULL;
        }
        
        PyUnicode_Append(&repr_str, num_str);
        Py_DECREF(num_str);
        if (repr_str == NULL) {
            return NULL;
        }

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
    
    PyObject *end_str = PyUnicode_FromString(",))");
    if (end_str == NULL) {
        Py_DECREF(repr_str);
        return NULL;
    }
    PyUnicode_Append(&repr_str, end_str);
    Py_DECREF(end_str);
    
    return repr_str;
}

static PyTypeObject TensorType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensor.Tensor",
    .tp_doc = PyDoc_STR("Tensor object"),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new = Tensor_new,
    .tp_init = (initproc)Tensor_init,
    .tp_dealloc = Tensor_dealloc,
    .tp_traverse = Tensor_traverse,
    .tp_clear = Tensor_clear,
    .tp_members = Tensor_members,
    .tp_methods = Tensor_methods,
    .tp_getset = Tensor_getseters,
    .tp_repr = Tensor_repr,
};

static int
tensor_module_exec(PyObject *m)
{
    if (PyType_Ready(&TensorType) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(m, "Tensor", (PyObject *) &TensorType) < 0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot tensor_module_slots[] = {
    {Py_mod_exec, tensor_module_exec},
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
    {0, NULL}
};

static PyModuleDef tensor_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "tensor",
    .m_doc = "A simple Tensor module",
    .m_size = 0,
    .m_slots = tensor_module_slots,
};

PyMODINIT_FUNC
PyInit_tensor(void)
{
    return PyModuleDef_Init(&tensor_module);
}
