# distutils: language = c++
# distutils: sources = parser_2d_cpp.cpp

"""
Wrap C++ class parser_2d_cpp.
"""
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

# Expose C++ class parser_2d_cpp to Python
cdef extern from "parser_2d_cpp.hpp":
	cdef cppclass parser_2d_cpp:
		parser_2d_cpp(char *, vector[double] &, vector[double] &) except +
		vector[vector[double]] evaluate()

cdef class Parser2d:
	"""
	Cython wrapper for the C++ class :obj:`parser_2d_cpp`.

	Attributes:
		parser (obj): Pointer to a :obj:`parser_2d_cpp` object.
	"""
	cdef parser_2d_cpp * parser

	def __cinit__(self, char * expr, vector[double] x, vector[double] y):
		"""
		Constructor.

		Args:
			expr (bytes): The (encoded) string to parse.
			x (array_like): :class:`numpy.ndarray` storing the :math:`x`-coordinates of
				the evaluation points.
			y (array_like): :class:`numpy.ndarray` storing the :math:`y`-coordinates of
				the evaluation points.
		"""
		self.parser = new parser_2d_cpp(expr, x, y)

	def __dealloc(self):
		"""
		Deallocator.
		"""
		del self.parser

	def evaluate(self):
		"""
		Evaluate the expression.

		Return:
			Two-dimensional :class:`numpy.ndarray` of the evaluations.
		"""
		values = self.parser.evaluate()
		return np.asarray(values)

