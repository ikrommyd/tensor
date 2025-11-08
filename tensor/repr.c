/*
 * Tensor string representations (__str__ and __repr__)
 */

#include "tensor.h"

// String representation function (__str__)
PyObject *
Tensor_str(PyObject *op)
{
    TensorObject *self = (TensorObject *)op;

    // Handle uninitialized tensor
    if (self->data == NULL) {
        return PyUnicode_FromString("<uninitialized>");
    }

    // Handle 0D tensor: just the value
    if (self->nd == 0) {
        double value = *((double *)self->data);
        PyObject *float_obj = PyFloat_FromDouble(value);
        if (float_obj == NULL) {
            return NULL;
        }
        PyObject *str = PyObject_Str(float_obj);
        Py_DECREF(float_obj);
        return str;
    }

    // Handle empty 1D tensor
    if (self->dimensions[0] == 0) {
        return PyUnicode_FromString("[]");
    }

    Py_ssize_t len = self->dimensions[0];
    Py_ssize_t max_display = 200;
    Py_ssize_t edge_items = 3;
    Py_ssize_t linebreak_threshold = 17;
    Py_ssize_t items_per_line = 20;

    int show_ellipsis = (len > max_display);
    int use_multiline = (len > linebreak_threshold) && !show_ellipsis;

    PyObject *str = PyUnicode_FromString("[");
    if (str == NULL) {
        return NULL;
    }

    // Determine which elements to show
    Py_ssize_t elements_to_show[max_display];
    Py_ssize_t num_elements_to_show;

    if (len <= max_display) {
        // Show all elements
        num_elements_to_show = len;
        for (Py_ssize_t i = 0; i < len; i++) {
            elements_to_show[i] = i;
        }
    }
    else {
        // Show first edge_items, ..., last edge_items
        num_elements_to_show = edge_items * 2;
        for (Py_ssize_t i = 0; i < edge_items; i++) {
            elements_to_show[i] = i;
        }
        for (Py_ssize_t i = 0; i < edge_items; i++) {
            elements_to_show[edge_items + i] = len - edge_items + i;
        }
    }

    // Print elements
    for (Py_ssize_t idx = 0; idx < num_elements_to_show; idx++) {
        Py_ssize_t i = elements_to_show[idx];

        // Add ellipsis between first and last edge items
        if (show_ellipsis && idx == edge_items) {
            PyObject *ellipsis_str;
            if (use_multiline) {
                ellipsis_str = PyUnicode_FromString("...,\n ");
            }
            else {
                ellipsis_str = PyUnicode_FromString("..., ");
            }
            if (ellipsis_str == NULL) {
                Py_DECREF(str);
                return NULL;
            }
            PyUnicode_Append(&str, ellipsis_str);
            Py_DECREF(ellipsis_str);
            if (str == NULL) {
                return NULL;
            }
        }

        char *data_ptr = self->data + (i * self->strides[0]);
        double value = *((double *)data_ptr);
        PyObject *float_obj = PyFloat_FromDouble(value);

        if (float_obj == NULL) {
            Py_DECREF(str);
            return NULL;
        }

        PyObject *num_str = PyObject_Repr(float_obj);
        Py_DECREF(float_obj);
        if (num_str == NULL) {
            Py_DECREF(str);
            return NULL;
        }

        PyUnicode_Append(&str, num_str);
        Py_DECREF(num_str);
        if (str == NULL) {
            return NULL;
        }

        // Add comma and potential line break
        if (idx < num_elements_to_show - 1) {
            PyObject *separator;
            if (use_multiline && (idx + 1) % items_per_line == 0) {
                // Line break after every items_per_line elements
                separator = PyUnicode_FromString(",\n ");
            }
            else {
                separator = PyUnicode_FromString(", ");
            }
            if (separator == NULL) {
                Py_DECREF(str);
                return NULL;
            }
            PyUnicode_Append(&str, separator);
            Py_DECREF(separator);
            if (str == NULL) {
                return NULL;
            }
        }
    }

    // Close with "]"
    PyObject *end_str = PyUnicode_FromString("]");
    if (end_str == NULL) {
        Py_DECREF(str);
        return NULL;
    }
    PyUnicode_Append(&str, end_str);
    Py_DECREF(end_str);

    return str;
}

// String representation function (__repr__)
PyObject *
Tensor_repr(PyObject *op)
{
    TensorObject *self = (TensorObject *)op;

    // Handle uninitialized tensor
    if (self->data == NULL) {
        return PyUnicode_FromString("Tensor(<uninitialized>)");
    }

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

    Py_ssize_t len = self->dimensions[0];
    Py_ssize_t max_display = 200;  // Maximum elements before truncation
    Py_ssize_t edge_items = 3;     // Items to show at start/end when truncated
    Py_ssize_t linebreak_threshold =
        17;                          // Break into multiple lines after this many items
    Py_ssize_t items_per_line = 20;  // Elements per line in multiline mode

    // Determine if we should use multi-line format
    int show_ellipsis = (len > max_display);
    int use_multiline = (len > linebreak_threshold) && !show_ellipsis;

    // Start building the string
    PyObject *repr_str = PyUnicode_FromString("Tensor([");
    if (repr_str == NULL) {
        return NULL;
    }

    // Determine which elements to show
    Py_ssize_t elements_to_show[max_display];
    Py_ssize_t num_elements_to_show;

    if (len <= max_display) {
        // Show all elements
        num_elements_to_show = len;
        for (Py_ssize_t i = 0; i < len; i++) {
            elements_to_show[i] = i;
        }
    }
    else {
        // Show first edge_items, ..., last edge_items
        num_elements_to_show = edge_items * 2;
        for (Py_ssize_t i = 0; i < edge_items; i++) {
            elements_to_show[i] = i;
        }
        for (Py_ssize_t i = 0; i < edge_items; i++) {
            elements_to_show[edge_items + i] = len - edge_items + i;
        }
    }

    // Print elements
    for (Py_ssize_t idx = 0; idx < num_elements_to_show; idx++) {
        Py_ssize_t i = elements_to_show[idx];

        // Add ellipsis between first and last edge items
        if (show_ellipsis && idx == edge_items) {
            PyObject *ellipsis_str;
            if (use_multiline) {
                ellipsis_str = PyUnicode_FromString("...,\n       ");
            }
            else {
                ellipsis_str = PyUnicode_FromString("..., ");
            }
            if (ellipsis_str == NULL) {
                Py_DECREF(repr_str);
                return NULL;
            }
            PyUnicode_Append(&repr_str, ellipsis_str);
            Py_DECREF(ellipsis_str);
            if (repr_str == NULL) {
                return NULL;
            }
        }

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

        // Add comma and potential line break
        if (idx < num_elements_to_show - 1) {
            PyObject *separator;
            if (use_multiline && (idx + 1) % items_per_line == 0) {
                // Line break after every items_per_line elements
                separator = PyUnicode_FromString(",\n       ");
            }
            else {
                separator = PyUnicode_FromString(", ");
            }
            if (separator == NULL) {
                Py_DECREF(repr_str);
                return NULL;
            }
            PyUnicode_Append(&repr_str, separator);
            Py_DECREF(separator);
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
