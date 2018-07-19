## @package gt4ess
#  Cython setup file.

from distutils.core import setup
from Cython.Build import cythonize

setup(name = "parser1d",
	  ext_modules = cythonize("parser_1d.pyx"),
	 )
