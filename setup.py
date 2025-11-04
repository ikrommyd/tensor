from setuptools import Extension, setup


setup(ext_modules=[Extension("tensor", ["tensor/tensor.c"])])
