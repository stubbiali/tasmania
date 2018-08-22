# distutils: language = c++
# distutils: sources = parser_1d_cpp.cpp

"""
Wrap C++ class parser_1d_cpp.
"""
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

# Expose C++ class parser_1d_cpp to Python
cdef extern from "parser_1d_cpp.hpp":
	cdef cppclass parser_1d_cpp:
		parser_1d_cpp(char *, vector[double] &) except +
		vector[double] evaluate()

cdef class Parser1d:
	"""
	Cython wrapper for the C++ class :obj:`parser_1d_cpp`.

	Attributes:
		parser (obj): Pointer to a :obj:`parser_1d_cpp` object.
	"""
	cdef parser_1d_cpp * parser

	def __cinit__(self, char * expr, vector[double] x):
		"""
		Constructor.

		Args:
			expr (bytes): The (encoded) string to parse.
			x (array_like): :class:`numpy.ndarray` storing the evaluation points.
		"""
		self.parser = new parser_1d_cpp(expr, x)

	def __dealloc(self):
		"""
		Deallocator.
		"""
		del self.parser

	def evaluate(self):
		"""
		Evaluate the expression.

		Return:
			:class:`numpy.ndarray` of the evaluations.
		"""
		values = self.parser.evaluate()
		return np.asarray(values)

