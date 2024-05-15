# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
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
