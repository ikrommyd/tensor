from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            "tensor",
            sources=[
                "tensor/tensor.c",
                "tensor/init.c",
                "tensor/getset.c",
                "tensor/methods.c",
                "tensor/repr.c",
                "tensor/mapping.c",
                "tensor/sequence.c",
            ],
            include_dirs=[numpy.get_include()],
        )
    ]
)
