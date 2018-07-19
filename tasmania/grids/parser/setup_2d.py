"""
Cython setup file for the two-dimensional parser.
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "parser2d",
	ext_modules = cythonize('parser_2d.pyx'),
	)
