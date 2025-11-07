from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            "tensor",
            sources=["tensor/tensor.c"],
            include_dirs=[numpy.get_include()],
        )
    ]
)
