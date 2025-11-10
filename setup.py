from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            "tensor",
            sources=[
                "tensor/tensorobject.c",
                "tensor/init.c",
                "tensor/getset.c",
                "tensor/members.c",
                "tensor/methods.c",
                "tensor/repr.c",
                "tensor/mapping.c",
                "tensor/sequence.c",
                "tensor/functions.c",
                "tensor/module.c",
            ],
            include_dirs=[numpy.get_include()],
        )
    ]
)
