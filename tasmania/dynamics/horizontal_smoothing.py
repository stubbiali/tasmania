# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
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
"""
Classes:
	HorizontalSmoothing
"""
import abc
import math
import numpy as np

import gridtools as gt
try:
	from tasmania.namelist import datatype
except ImportError:
	from numpy import float32 as datatype


class HorizontalSmoothing:
	"""
	Abstract base class whose derived classes apply horizontal
	numerical smoothing to a generic (prognostic) field by means
	of a GT4Py stencil.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, dims, grid, smooth_damp_depth, smooth_coeff,
				 smooth_coeff_max, backend, dtype):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Shape of the (three-dimensional) arrays on which
			to apply numerical smoothing.
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		smooth_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing.
		dtype : obj
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
		"""
		# Store input arguments
		self._dims              = dims
		self._grid              = grid
		self._smooth_damp_depth = smooth_damp_depth
		self._smooth_coeff      = smooth_coeff
		self._smooth_coeff_max  = smooth_coeff_max
		self._backend			= backend

		# Initialize the smoothing matrix
		self._gamma = self._smooth_coeff * np.ones(self._dims, dtype=dtype)

		# The diffusivity is monotically increased towards the top of the model,
		# so to mimic the effect of a short length wave absorber
		n = self._smooth_damp_depth
		if n > 0:
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
			pert = np.tile(pert[np.newaxis, np.newaxis, :], (dims[0], dims[1], 1))
			self._gamma[:, :, :n] += (self._smooth_coeff_max - self._smooth_coeff) * pert

		# Initialize the pointer to the underlying stencil
		# It will be properly re-directed the first time the call
		# operator in invoked
		self._stencil = None

	@abc.abstractmethod
	def __call__(self, phi, phi_out):
		"""
		Apply horizontal smoothing to a prognostic field.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		phi : array_like
			3-D :class:`numpy.ndarray` representing the field to filter.
		phi_out : array_like
			3-D :class:`numpy.ndarray` into which the filtered field
			is written.
		"""

	@staticmethod
	def factory(smooth_type, dims, grid, smooth_damp_depth, smooth_coeff,
				smooth_coeff_max, backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Static method returning an instance of the derived class
		implementing the smoothing technique specified by :data:`smooth_type`.

		Parameters
		----------
		smooth_type : string
			String specifying the smoothing technique to implement. Either:

			* 'first_order', for first-order numerical smoothing;
			* 'second_order', for second-order numerical smoothing;
			* 'third_order', for third-order numerical smoothing.

		dims : tuple
			Shape of the (three-dimensional) arrays on which
			to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		smooth_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.

		Return
		------
		obj :
			Instance of the suitable derived class.
		"""
		arg_list = [dims, grid, smooth_damp_depth, smooth_coeff,
					smooth_coeff_max, backend, dtype]
		import tasmania.dynamics._horizontal_smoothing as module

		if smooth_type == 'first_order':
			if dims[1] == 1:
				return module._FirstOrderXZ(*arg_list)
			elif dims[0] == 1:
				return module._FirstOrderYZ(*arg_list)
			else:
				return module._FirstOrder(*arg_list)
		elif smooth_type == 'second_order':
			if dims[1] == 1:
				return module._SecondOrderXZ(*arg_list)
			elif dims[0] == 1:
				return module._SecondOrderYZ(*arg_list)
			else:
				return module._SecondOrder(*arg_list)
		elif smooth_type == 'third_order':
			if dims[1] == 1:
				return module._ThirdOrderXZ(*arg_list)
			elif dims[0] == 1:
				return module._ThirdOrderYZ(*arg_list)
			else:
				return module._ThirdOrder(*arg_list)
