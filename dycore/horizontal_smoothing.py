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
Classes applying numerical smoothing.
"""
import abc
import math
import numpy as np

import gridtools as gt
from tasmania.namelist import datatype

class HorizontalSmoothing:
	"""
	Abstract base class whose derived classes apply horizontal numerical smoothing to a generic (prognostic) 
	field by means of a GT4Py stencil.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : int
			Depth of the damping region, i.e., number of vertical layers in the damping region.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical smoothing.
		"""
		# Store useful input arguments
		self._dims              = dims
		self._grid              = grid
		self._smooth_damp_depth = smooth_damp_depth
		self._smooth_coeff      = smooth_coeff
		self._smooth_coeff_max  = smooth_coeff_max
		self._backend           = backend

		# Initialize the smoothing matrix
		self._gamma = self._smooth_coeff * np.ones(self._dims, dtype = datatype)

		# The filterivity is monotically increased towards the top of the model, 
		# so to mimic the effect of a short length wave absorber
		n = self._smooth_damp_depth
		if n > 0:
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype = datatype)) / n) ** 2
			pert = np.tile(pert[np.newaxis, np.newaxis, :], (dims[0], dims[1], 1))
			self._gamma[:, :, 0:n] += (self._smooth_coeff_max - self._smooth_coeff) * pert

		# Allocate the Numpy array which will represent the stencil's input field
		self._in_phi = np.zeros(dims, dtype = datatype)

		# Allocate memory for the stencil's output field
		self._out_phi = np.zeros(dims, dtype = datatype)

		# Initialize pointer to stencil; this will be properly re-directed the first time 
		# the entry-point method apply is invoked
		self._stencil = None

	@abc.abstractmethod
	def apply(self, phi):
		"""
		Apply horizontal smoothing to a prognostic field.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the filtered field.
		"""

	@staticmethod
	def factory(smooth_type, dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend):
		"""
		Static method returning an instance of the derived class implementing the smoothing technique
		specified by :data:`smooth_type`.

		Parameters
		----------
		smooth_type : string
			String specifying the smoothing technique to implement. Either:

			* 'first_order', for first-order numerical smoothing;
			* 'second_order', for second-order numerical smoothing.

		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : int
			Depth of the damping region, i.e., number of vertical layers in the damping region.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical 
			smoothing. Default is :class:`gridtools.mode.NUMPY`.

		Return
		------
		obj :
			Instance of the suitable derived class.
		"""
		if smooth_type == 'first_order':
			if dims[1] == 1:
				return HorizontalSmoothingFirstOrderXZ(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)
			elif dims[0] == 1:
				return HorizontalSmoothingFirstOrderYZ(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)
			else:
				return HorizontalSmoothingFirstOrderXYZ(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)
		elif smooth_type == 'second_order':
			if dims[1] == 1:
				return HorizontalSmoothingSecondOrderXZ(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)
			elif dims[0] == 1:
				return HorizontalSmoothingSecondOrderYZ(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)
			else:
				return HorizontalSmoothingSecondOrderXYZ(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)


class HorizontalSmoothingFirstOrderXYZ(HorizontalSmoothing):
	"""
	This class inherits :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` to apply first-order numerical smoothing to 
	three-dimensional fields with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(self, dims, grid, smooth_damp_depth = 10, smooth_coeff = .03, smooth_coeff_max = .24, backend = gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : `int`, optional
			Depth of the damping region, i.e., number of vertical layers in the damping region. Default is 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary. Default is 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. For the sake of numerical stability, it should not 
			exceed 0.25. Default is 0.24.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical 
			smoothing. Default is :class:`gridtools.mode.NUMPY`.

		Note
		----
		To instantiate the class, please prefer the static method :meth:`~dycore.horizontal_smoothing.HorizontalSmoothing.factor`
		of :class:`~dycore.horizontal_smoothing.HorizontalSmoothing`.
		"""
		super().__init__(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)

	def apply(self, phi):
		"""
		Apply first-order horizontal smoothing to a prognostic field.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the filtered field.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[:,:,:] = phi[:,:,:]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layer of the output field, not affected by the stencil
		self._out_phi[ 0, :, :]    = self._in_phi[ 0, :, :]
		self._out_phi[-1, :, :]    = self._in_phi[-1, :, :]
		self._out_phi[1:-1,  0, :] = self._in_phi[1:-1,  0, :]
		self._out_phi[1:-1, -1, :] = self._in_phi[1:-1, -1, :]
		
		return self._out_phi

	def _stencil_initialize(self, phi):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.
		"""
		# Set the computational domain
		ni, nj, nk = self._dims
		_domain = gt.domain.Rectangle((1, 1, 0), (ni - 2, nj - 2, nk - 1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs = {'out_phi': self._out_phi},
			domain = _domain, 
			mode = self._backend)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil applying first-order horizontal smoothing. A centered 5-points formula is used.

		Parameters
		----------
		in_phi : obj 
			:class:`gridtools.Equation` representing the input field to filter.
		gamma : obj 
			:class:`gridtools.Equation` representing the smoothing coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the filtered output field.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j, k] = (1. - 4. * gamma[i, j, k]) * in_phi[i, j, k] + \
						   gamma[i, j, k] * (in_phi[i-1, j, k] + in_phi[i+1, j, k] + in_phi[i, j-1, k] + in_phi[i, j+1, k])

		return out_phi


class HorizontalSmoothingFirstOrderXZ(HorizontalSmoothing):
	"""
	This class inherits :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` to apply first-order numerical smoothing to 
	three-dimensional fields with only one element in the :math:`y`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(self, dims, grid, smooth_damp_depth = 10, smooth_coeff = .03, smooth_coeff_max = .49, backend = gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : `int`, optional
			Depth of the damping region, i.e., number of vertical layers in the damping region. Default is 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary. Default is 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. For the sake of numerical stability, it should not 
			exceed 0.5. Default is 0.49.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical 
			smoothing. Default is :class:`gridtools.mode.NUMPY`.
		"""
		super().__init__(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)

	def apply(self, phi):
		"""
		Apply first-order horizontal smoothing to a prognostic field.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the filtered field.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[:,:,:] = phi[:,:,:]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field, not affected by the stencil
		self._out_phi[ 0, :, :] = self._in_phi[ 0, :, :]
		self._out_phi[-1, :, :] = self._in_phi[-1, :, :]
		
		return self._out_phi

	def _stencil_initialize(self, phi):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.
		"""
		# Set the computational domain
		ni, _, nk = self._dims
		_domain = gt.domain.Rectangle((1, 0, 0), (ni - 2, 0, nk - 1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs = {'out_phi': self._out_phi},
			domain = _domain, 
			mode = self._backend)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil applying horizontal smoothing. A standard centered 3-points formula is used.

		Parameters
		----------
		in_phi : obj 
			:class:`gridtools.Equation` representing the input field to filter.
		gamma : obj 
			:class:`gridtools.Equation` representing the smoothing coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the filtered output field.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j, k] = (1. - 2. * gamma[i, j, k]) * in_phi[i, j, k] + \
						   gamma[i, j, k] * (in_phi[i-1, j, k] + in_phi[i+1, j, k])

		return out_phi


class HorizontalSmoothingFirstOrderYZ(HorizontalSmoothing):
	"""
	This class inherits :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` to apply first-order numerical smoothing to 
	three-dimensional fields with only one element in the :math:`x`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(self, dims, grid, smooth_damp_depth = 10, smooth_coeff = .03, smooth_coeff_max = .49, backend = gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : int
			Depth of the damping region, i.e., number of vertical layers in the damping region. Default is 10.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary. Default is 0.03.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient. For the sake of numerical stability, it should not 
			exceed 0.5. Default is 0.49.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical 
			smoothing. Default is :class:`gridtools.mode.NUMPY`.
		"""
		super().__init__(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)

	def apply(self, phi):
		"""
		Apply first-order horizontal smoothing to a prognostic field.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the filtered field.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[:,:,:] = phi[:,:,:]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layer of the output field, not affected by the stencil
		self._out_phi[:,  0, :] = self._in_phi[:,  0, :]
		self._out_phi[:, -1, :] = self._in_phi[:, -1, :]
		
		return self._out_phi

	def _stencil_initialize(self, phi):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.
		"""
		# Set the computational domain
		_, nj, nk = self._dims
		_domain = gt.domain.Rectangle((0, 1, 0), (0, nj - 2, nk - 1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs = {'out_phi': self._out_phi},
			domain = _domain, 
			mode = self._backend)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil applying horizontal smoothing. A standard centered 3-points formula is used.

		Parameters
		----------
		in_phi : obj 
			:class:`gridtools.Equation` representing the input field to filter.
		gamma : obj 
			:class:`gridtools.Equation` representing the smoothing coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the filtered output field.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j, k] = (1. - 2. * gamma[i, j, k]) * in_phi[i, j, k] + \
						   gamma[i, j, k] * (in_phi[i, j-1, k] + in_phi[i, j+1, k])

		return out_phi


class HorizontalSmoothingSecondOrderXYZ(HorizontalSmoothing):
	"""
	This class inherits :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` to apply second-order numerical smoothing to 
	three-dimensional fields.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(self, dims, grid, smooth_damp_depth = 10, smooth_coeff = .03, smooth_coeff_max = .24, backend = gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : `int`, optional
			Depth of the damping region, i.e., number of vertical layers in the damping region. Default is 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary. Default is 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. For the sake of numerical stability, it should not 
			exceed 0.25. Default is 0.24.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical 
			smoothing. Default is :class:`gridtools.mode.NUMPY`.
		"""
		super().__init__(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)

	def apply(self, phi):
		"""
		Apply second-order horizontal smoothing to a prognostic field.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the filtered field.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[:,:,:] = phi[:,:,:]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field, not affected by the stencil
		self._out_phi[ 0, :, :]    = self._in_phi[ 0, :, :]
		self._out_phi[ 1, :, :]    = self._in_phi[ 1, :, :]
		self._out_phi[-2, :, :]    = self._in_phi[-2, :, :]
		self._out_phi[-1, :, :]    = self._in_phi[-1, :, :]
		self._out_phi[2:-2,  0, :] = self._in_phi[2:-2,  0, :]
		self._out_phi[2:-2,  1, :] = self._in_phi[2:-2,  1, :]
		self._out_phi[2:-2, -2, :] = self._in_phi[2:-2, -2, :]
		self._out_phi[2:-2, -1, :] = self._in_phi[2:-2, -1, :]
		
		return self._out_phi

	def _stencil_initialize(self, phi):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.
		"""
		# Set the computational domain
		ni, nj, nk = self._dims
		_domain = gt.domain.Rectangle((2, 2, 0), (ni - 3, nj - 3, nk - 1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs = {'out_phi': self._out_phi},
			domain = _domain, 
			mode = self._backend)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil applying horizontal smoothing. A standard centered 5-points formula is used.

		Parameters
		----------
		in_phi : obj 
			:class:`gridtools.Equation` representing the input field to filter.
		gamma : obj 
			:class:`gridtools.Equation` representing the smoothing coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the filtered output field.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j, k] = (1 - 12. * gamma[i, j, k]) * in_phi[i, j, k] + \
						   gamma[i, j, k] * (- (in_phi[i-2, j, k] + in_phi[i+2, j, k]) 
						   					 + 4. * (in_phi[i-1, j, k] + in_phi[i+1, j, k])
											 + 4. * (in_phi[i, j-1, k] + in_phi[i, j+1, k])
											 - (in_phi[i, j-2, k] + in_phi[i, j+2, k]))

		return out_phi
		
		
class HorizontalSmoothingSecondOrderXZ(HorizontalSmoothing):
	"""
	This class inherits :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` to apply second-order numerical smoothing to 
	three-dimensional fields with only one element in the :math:`y`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(self, dims, grid, smooth_damp_depth = 10, smooth_coeff = .03, smooth_coeff_max = .49, backend = gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : `int`, optional
			Depth of the damping region, i.e., number of vertical layers in the damping region. Default is 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary. Default is 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. For the sake of numerical stability, it should not 
			exceed 0.5. Default is 0.49.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical 
			smoothing. Default is :class:`gridtools.mode.NUMPY`.
		"""
		super().__init__(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)

	def apply(self, phi):
		"""
		Apply second-order horizontal smoothing to a prognostic field.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the filtered field.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[:,:,:] = phi[:,:,:]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field, not affected by the stencil
		self._out_phi[ 0, :, :]    = self._in_phi[ 0, :, :]
		self._out_phi[ 1, :, :]    = self._in_phi[ 1, :, :]
		self._out_phi[-2, :, :]    = self._in_phi[-2, :, :]
		self._out_phi[-1, :, :]    = self._in_phi[-1, :, :]
		
		return self._out_phi

	def _stencil_initialize(self, phi):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.
		"""
		# Set the computational domain
		ni, _, nk = self._dims
		_domain = gt.domain.Rectangle((2, 0, 0), (ni - 3, 0, nk - 1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs = {'out_phi': self._out_phi},
			domain = _domain, 
			mode = self._backend)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil applying horizontal smoothing. A standard centered 5-points formula is used.

		Parameters
		----------
		in_phi : obj 
			:class:`gridtools.Equation` representing the input field to filter.
		gamma : obj 
			:class:`gridtools.Equation` representing the smoothing coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the filtered output field.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j, k] = (1 - 6. * gamma[i, j, k]) * in_phi[i, j, k] + \
						   gamma[i, j, k] * (- (in_phi[i-2, j, k] + in_phi[i+2, j, k]) 
						   					 + 4. * (in_phi[i-1, j, k] + in_phi[i+1, j, k]))

		return out_phi


class HorizontalSmoothingSecondOrderYZ(HorizontalSmoothing):
	"""
	This class inherits :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` to apply second-order numerical smoothing to 
	three-dimensional fields with only one element in the :math:`x`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(self, dims, grid, smooth_damp_depth = 10, smooth_coeff = .03, smooth_coeff_max = .49, backend = gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the arrays on which to apply numerical smoothing.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		smooth_damp_depth : `int`, optional
			Depth of the damping region, i.e., number of vertical layers in the damping region. Default is 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary. Default is 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. For the sake of numerical stability, it should not 
			exceed 0.5. Default is 0.49.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil implementing numerical 
			smoothing. Default is :class:`gridtools.mode.NUMPY`.
		"""
		super().__init__(dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend)

	def apply(self, phi):
		"""
		Apply second-order horizontal smoothing to a prognostic field.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the filtered field.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[:,:,:] = phi[:,:,:]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field, not affected by the stencil
		self._out_phi[:,  0, :]    = self._in_phi[:,  0, :]
		self._out_phi[:,  1, :]    = self._in_phi[:,  1, :]
		self._out_phi[:, -2, :]    = self._in_phi[:, -2, :]
		self._out_phi[:, -1, :]    = self._in_phi[:, -1, :]
		
		return self._out_phi

	def _stencil_initialize(self, phi):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the field to filter.
		"""
		# Set the computational domain
		_, nj, nk = self._dims
		_domain = gt.domain.Rectangle((0, 2, 0), (0, nj - 3, nk - 1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs = {'out_phi': self._out_phi},
			domain = _domain, 
			mode = self._backend)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil applying horizontal smoothing. A standard centered 5-points formula is used.

		Parameters
		----------
		in_phi : obj 
			:class:`gridtools.Equation` representing the input field to filter.
		gamma : obj 
			:class:`gridtools.Equation` representing the smoothing coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the filtered output field.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j, k] = (1 - 6. * gamma[i, j, k]) * in_phi[i, j, k] + \
						   gamma[i, j, k] * (- (in_phi[i, j-2, k] + in_phi[i, j+2, k]) 
						   					 + 4. * (in_phi[i, j+1, k] + in_phi[i, j+1, k]))

		return out_phi
