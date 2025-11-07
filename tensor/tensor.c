/*
This is a simple C extension module that defines a Tensor type for Python.
It's only a 0D or 1D tensor of double precision floats for now but we keep things
open for more dimensions in the future.
The goal is to have the most minimal implementation possible so that I can
learn the C API for Python extensions.
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h> /* for offsetof() */

// Define the Tensor object struct
typedef struct {
    PyObject_HEAD
    char *data;              // Pointer to the start of the data buffer
    int nd;                  // Number of dimensions (will be 0 or 1 for now)
    Py_ssize_t *dimensions;  // Array of dimension sizes
    Py_ssize_t *strides;     // Bytes to next element in each dimension
    PyObject *base;          // For views/slices - NULL if owner
} TensorObject;

static void
Tensor_dealloc(PyObject *op)
{
    TensorObject *self = (TensorObject *)op;

    PyMem_Free(self->dimensions);
    PyMem_Free(self->strides);

    if (self->base == NULL) {
        PyMem_Free(self->data);
    }
    else {
        Py_DECREF(self->base);
    }

    Py_TYPE(self)->tp_free(op);
}

static PyObject *
Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    TensorObject *self;
    self = (TensorObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->data = NULL;
        self->nd = 0;
        self->dimensions = NULL;
        self->strides = NULL;
        self->base = NULL;
    }
    return (PyObject *)self;
}

static int
Tensor_init(PyObject *op, PyObject *args, PyObject *kwds)
{
    TensorObject *self = (TensorObject *)op;
    PyObject *input;

    if (!PyArg_ParseTuple(args, "O", &input)) {
        return -1;
    }

    if (PyNumber_Check(input) && !PySequence_Check(input)) {
        self->nd = 0;

        self->dimensions = PyMem_Malloc(0);
        self->strides = PyMem_Malloc(0);
        if (self->dimensions == NULL || self->strides == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        self->data = PyMem_Malloc(sizeof(double));
        if (self->data == NULL) {
            PyErr_NoMemory();
            return -1;
        }

        PyObject *float_obj = PyNumber_Float(input);
        if (float_obj == NULL) {
            return -1;
        }

        double value = PyFloat_AsDouble(float_obj);
        Py_DECREF(float_obj);

        if (value == -1.0 && PyErr_Occurred()) {
            return -1;
        }

        *((double *)self->data) = value;
        self->base = NULL;
        return 0;
    }

    if (!PySequence_Check(input)) {
        PyErr_SetString(PyExc_TypeError, "Expected a number or a sequence of numbers");
        return -1;
    }

    Py_ssize_t length = PySequence_Length(input);
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
        PyObject *item = PySequence_GetItem(input, i);

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
    {NULL} /* Sentinel */
};

static PyObject *
Tensor_getshape(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
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
    TensorObject *self = (TensorObject *)op;
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
    TensorObject *self = (TensorObject *)op;
    if (self->base) {
        Py_INCREF(self->base);
        return self->base;
    }
    else {
        Py_RETURN_NONE;
    }
}

static PyObject *
Tensor_getsize(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
    Py_ssize_t size = 1;
    for (int i = 0; i < self->nd; i++) {
        size *= self->dimensions[i];
    }
    return PyLong_FromSsize_t(size);
}

static PyObject *
Tensor_getdata(PyObject *op, void *closure)
{
    TensorObject *self = (TensorObject *)op;
    if (self->data == NULL) {
        PyErr_SetString(PyExc_ValueError, "Tensor has no data");
        return NULL;
    }
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

static PyObject *
Tensor_tolist(PyObject *op, PyObject *Py_UNUSED(ignored))
{
    TensorObject *self = (TensorObject *)op;

    if (self->nd == 0) {
        double value = *((double *)self->data);
        return PyFloat_FromDouble(value);
    }

    PyObject *list = PyList_New(self->dimensions[0]);
    if (list == NULL) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
        char *data_ptr = self->data + (i * self->strides[0]);
        double value = *((double *)data_ptr);
        PyObject *float_obj = PyFloat_FromDouble(value);
        if (float_obj == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, float_obj);
    }

    return list;
}

static PyObject *
Tensor_item(PyObject *op, PyObject *Py_UNUSED(ignored))
{
    TensorObject *self = (TensorObject *)op;
    if (self->nd != 0) {
        PyErr_SetString(PyExc_ValueError, "item() only valid for 0D tensors");
        return NULL;
    }
    double value = *((double *)self->data);
    return PyFloat_FromDouble(value);
}

static PyObject *
Tensor_copy(PyObject *op, PyObject *Py_UNUSED(ignored))
{
    TensorObject *self = (TensorObject *)op;

    TensorObject *copy = (TensorObject *)Tensor_new(Py_TYPE(self), NULL, NULL);
    if (copy == NULL) {
        return NULL;
    }

    copy->nd = self->nd;

    copy->dimensions = PyMem_Malloc(sizeof(Py_ssize_t) * copy->nd);
    if (copy->dimensions == NULL) {
        PyErr_NoMemory();
        Py_DECREF(copy);
        return NULL;
    }

    copy->strides = PyMem_Malloc(sizeof(Py_ssize_t) * copy->nd);
    if (copy->strides == NULL) {
        PyErr_NoMemory();
        Py_DECREF(copy);
        return NULL;
    }

    if (copy->nd == 0) {
        copy->data = PyMem_Malloc(sizeof(double));
        if (copy->data == NULL) {
            PyErr_NoMemory();
            Py_DECREF(copy);
            return NULL;
        }
        *((double *)copy->data) = *((double *)self->data);
    }
    else {
        copy->dimensions[0] = self->dimensions[0];

        if (self->dimensions[0] == 0) {
            copy->strides[0] = 0;
            copy->data = NULL;
        }
        else {
            copy->strides[0] = sizeof(double);
            Py_ssize_t data_size = self->dimensions[0] * sizeof(double);
            copy->data = PyMem_Malloc(data_size);
            if (copy->data == NULL) {
                PyErr_NoMemory();
                Py_DECREF(copy);
                return NULL;
            }

            double *dst = (double *)copy->data;
            for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
                char *src_ptr = self->data + (i * self->strides[0]);
                dst[i] = *((double *)src_ptr);
            }
        }
    }

    copy->base = NULL;
    return (PyObject *)copy;
}

static struct PyMethodDef Tensor_methods[] = {
    {"tolist", (PyCFunction)Tensor_tolist, METH_NOARGS, "Convert tensor to a list"},
    {"item", (PyCFunction)Tensor_item, METH_NOARGS,
     "Get the single item from a 0D tensor as a python scalar"},
    {"copy", (PyCFunction)Tensor_copy, METH_NOARGS, "Return a copy of the tensor"},
    {NULL, NULL, 0, NULL}};

static PyObject *
Tensor_repr(PyObject *op)
{
    TensorObject *self = (TensorObject *)op;

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

    if (self->dimensions[0] == 0) {
        return PyUnicode_FromFormat("Tensor([], shape=(0,))");
    }

    PyObject *repr_str = PyUnicode_FromString("Tensor([");
    if (repr_str == NULL) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < self->dimensions[0]; i++) {
        // Convert double to Python float, then to string
        char *data_ptr = self->data + (i * self->strides[0]);
        double value = *((double *)data_ptr);
        PyObject *float_obj = PyFloat_FromDouble(value);

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

static PyObject *
Tensor_subscript(PyObject *op, PyObject *key)
{
    TensorObject *self = (TensorObject *)op;

    if (self->nd == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Tensor is 0-dimensional and cannot be indexed");
        return NULL;
    }

    if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        Py_ssize_t new_index;
        if (index == -1 && PyErr_Occurred()) {
            return NULL;
        }

        if (index < 0) {
            new_index = self->dimensions[0] + index;
        }
        else {
            new_index = index;
        }

        if (new_index < 0 || new_index >= self->dimensions[0]) {
            PyErr_Format(PyExc_IndexError,
                         "Index %zd out of range for tensor of size %zd", index,
                         self->dimensions[0]);
            return NULL;
        }

        TensorObject *scalar = (TensorObject *)Tensor_new(Py_TYPE(self), NULL, NULL);
        if (scalar == NULL) {
            return NULL;
        }

        scalar->nd = 0;

        scalar->dimensions = PyMem_Malloc(0);
        scalar->strides = PyMem_Malloc(0);
        if (scalar->dimensions == NULL || scalar->strides == NULL) {
            PyErr_NoMemory();
            Py_DECREF(scalar);
            return NULL;
        }

        scalar->data = PyMem_Malloc(sizeof(double));
        if (scalar->data == NULL) {
            PyErr_NoMemory();
            Py_DECREF(scalar);
            return NULL;
        }

        char *data_ptr = self->data + (new_index * self->strides[0]);
        *((double *)scalar->data) = *((double *)data_ptr);

        scalar->base = NULL;
        return (PyObject *)scalar;
    }
    else if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength;

        if (PySlice_Unpack(key, &start, &stop, &step) < 0) {
            return NULL;
        }

        slicelength = PySlice_AdjustIndices(self->dimensions[0], &start, &stop, step);

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
            view->strides[0] = self->strides[0];
            view->data = self->data;
        }
        else {
            view->strides[0] = self->strides[0] * step;
            view->data = self->data + (start * self->strides[0]);
        }

        if (self->base != NULL) {
            view->base = self->base;
        }
        else {
            view->base = (PyObject *)self;
        }
        Py_INCREF(view->base);

        return (PyObject *)view;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
        return NULL;
    }
}

static PyTypeObject TensorType;

static int
Tensor_ass_subscript(PyObject *op, PyObject *key, PyObject *value)
{
    TensorObject *self = (TensorObject *)op;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot delete tensor elements");
        return -1;
    }

    if (self->nd == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Tensor is 0-dimensional and cannot be indexed");
        return -1;
    }
    TensorObject *value_tensor;

    if (PyObject_TypeCheck(value, &TensorType)) {
        value_tensor = (TensorObject *)value;
        Py_INCREF(value_tensor);
    }
    else {
        value_tensor = (TensorObject *)Tensor_new(&TensorType, NULL, NULL);
        if (value_tensor == NULL) {
            return -1;
        }

        PyObject *init_args = PyTuple_Pack(1, value);
        if (init_args == NULL) {
            Py_DECREF(value_tensor);
            return -1;
        }

        if (Tensor_init((PyObject *)value_tensor, init_args, NULL) < 0) {
            Py_DECREF(init_args);
            Py_DECREF(value_tensor);
            return -1;
        }

        Py_DECREF(init_args);
    }

    // Handle integer indexing: x[i] = value
    if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        Py_ssize_t new_index;
        if (index == -1 && PyErr_Occurred()) {
            Py_DECREF(value_tensor);
            return -1;
        }

        if (index < 0) {
            new_index = self->dimensions[0] + index;
        }
        else {
            new_index = index;
        }

        if (new_index < 0 || new_index >= self->dimensions[0]) {
            PyErr_Format(PyExc_IndexError,
                         "Index %zd out of range for tensor of size %zd", index,
                         self->dimensions[0]);
            Py_DECREF(value_tensor);
            return -1;
        }

        if (value_tensor->nd != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "setting a tensor element with a sequence");
            Py_DECREF(value_tensor);
            return -1;
        }

        double scalar_value;
        scalar_value = *((double *)value_tensor->data);
        char *data_ptr = self->data + (new_index * self->strides[0]);
        *((double *)data_ptr) = scalar_value;

        Py_DECREF(value_tensor);
        return 0;
    }
    else if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength;

        if (PySlice_Unpack(key, &start, &stop, &step) < 0) {
            Py_DECREF(value_tensor);
            return -1;
        }

        slicelength = PySlice_AdjustIndices(self->dimensions[0], &start, &stop, step);

        // Case 1: Scalar broadcast (0D or size 1)
        if (value_tensor->nd == 0 ||
            (value_tensor->nd == 1 && value_tensor->dimensions[0] == 1)) {
            double scalar_value = *((double *)value_tensor->data);

            // Broadcast to all elements in the slice
            for (Py_ssize_t i = 0; i < slicelength; i++) {
                Py_ssize_t idx = start + i * step;
                char *data_ptr = self->data + (idx * self->strides[0]);
                *((double *)data_ptr) = scalar_value;
            }
        }
        // Case 2: Element-wise assignment
        else if (value_tensor->nd == 1 && value_tensor->dimensions[0] == slicelength) {
            for (Py_ssize_t i = 0; i < slicelength; i++) {
                Py_ssize_t idx = start + i * step;
                char *data_ptr = self->data + (idx * self->strides[0]);
                char *value_ptr = value_tensor->data + (i * value_tensor->strides[0]);
                double val = *((double *)value_ptr);
                *((double *)data_ptr) = val;
            }
        }
        // Case 3: Shape mismatch
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

static PyMappingMethods Tensor_as_mapping = {
    .mp_length = Tensor_length,
    .mp_subscript = Tensor_subscript,
    .mp_ass_subscript = Tensor_ass_subscript,
};

static PyTypeObject TensorType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "tensor.Tensor",
    .tp_doc = PyDoc_STR("Tensor object"),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Tensor_new,
    .tp_init = (initproc)Tensor_init,
    .tp_dealloc = Tensor_dealloc,
    .tp_members = Tensor_members,
    .tp_methods = Tensor_methods,
    .tp_getset = Tensor_getseters,
    .tp_repr = (reprfunc)Tensor_repr,
    .tp_as_mapping = &Tensor_as_mapping,
};

static PyObject *
tensor_tensor(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int copy = 0;

    static char *kwlist[] = {"object", "copy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$p", kwlist, &obj, &copy)) {
        return NULL;
    }

    if (PyObject_TypeCheck(obj, &TensorType)) {
        if (copy) {
            return Tensor_copy(obj, NULL);
        }
        else {
            Py_INCREF(obj);
            return obj;
        }
    }

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

static int
tensor_module_exec(PyObject *m)
{
    if (PyType_Ready(&TensorType) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(m, "Tensor", (PyObject *)&TensorType) < 0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot tensor_module_slots[] = {
    {Py_mod_exec, tensor_module_exec},
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
    {0, NULL}};

static PyModuleDef tensor_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "tensor",
    .m_doc = "A simple Tensor module",
    .m_size = 0,
    .m_methods = tensor_module_methods,
    .m_slots = tensor_module_slots,
};

PyMODINIT_FUNC
PyInit_tensor(void)
{
    return PyModuleDef_Init(&tensor_module);
}
