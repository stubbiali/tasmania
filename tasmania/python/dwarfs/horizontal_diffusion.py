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
This module contains:
	HorizontalDiffusion
	_FirstOrder(HorizontalDiffusion)
	_FirstOrder{XZ, YZ}(HorizontalDiffusion)
	_SecondOrder(HorizontalDiffusion)
	_SecondOrder{XZ, YZ}(HorizontalDiffusion)
	_ThirdOrder(HorizontalDiffusion)
	_ThirdOrder{XZ, YZ}(HorizontalDiffusion)
"""
import abc
import math
import numpy as np

import gridtools as gt
try:
	from tasmania.namelist import datatype
except ImportError:
	from numpy import float32 as datatype


def stage_laplacian(i, j, dx, dy, in_phi, out_phi):
	out_phi[i, j] = \
		(in_phi[i-1, j] - 2*in_phi[i, j] + in_phi[i+1, j]) / (dx*dx) + \
		(in_phi[i, j-1] - 2*in_phi[i, j] + in_phi[i, j+1]) / (dy*dy)
	return out_phi


class HorizontalDiffusion:
	"""
	Abstract base class whose derived classes calculates the
	tendency due to horizontal diffusion for any prognostic field.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, dims, grid, smooth_damp_depth, smooth_coeff,
		smooth_coeff_max, xaxis_units, yaxis_units, backend, dtype
	):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Shape of the (three-dimensional) arrays for which
			tendencies should be computed.
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.python.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		smooth_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient.
		xaxis_units : str
			TODO
		yaxis_units : str
			TODO
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
		self._xunits			= xaxis_units
		self._yunits			= yaxis_units
		self._backend			= backend

		# Initialize the smoothing matrix
		self._gamma = self._smooth_coeff * np.ones(self._dims, dtype=dtype)
		for i in range(10):
			self._gamma[i, :, :] *= (10-i)
			self._gamma[-(i+1), :, :] *= (10-i)
			self._gamma[:, i, :] *= (10-i)
			self._gamma[:, -(i+1), :] *= (10-i)

		# The diffusivity is monotonically increased towards the top of the model,
		# so to mimic the effect of a short length wave absorber
		n = self._smooth_damp_depth
		if n > 0:
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
			pert = np.tile(pert[np.newaxis, np.newaxis, :], (dims[0], dims[1], 1))
			self._gamma[:, :, :n] += (self._smooth_coeff_max - self._smooth_coeff) * pert

		# Initialize the pointer to the underlying stencil
		# It will be properly re-directed the first time the call
		# operator is invoked
		self._stencil = None

	@abc.abstractmethod
	def __call__(self, phi, phi_tnd):
		"""
		Calculate the tendency.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		phi : :class:`numpy.ndarray`
			The 3-D prognostic field.
		phi_tnd : :class:`numpy.ndarray`
			Buffer into which the calculated tendency is written.
		"""

	@staticmethod
	def factory(
		smooth_type, dims, grid, smooth_damp_depth, smooth_coeff,
		smooth_coeff_max, xaxis_units, yaxis_units,
		backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Static method returning an instance of the derived class
		calculating the tendency due to horizontal diffusion of type
		:data:`smooth_type`.

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
		xaxis_units : str
			TODO
		yaxis_units : str
			TODO
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
		arg_list = [
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		]

		if smooth_type == 'first_order':
			if dims[1] == 1:
				return _FirstOrderXZ(*arg_list)
			elif dims[0] == 1:
				return _FirstOrderYZ(*arg_list)
			else:
				return _FirstOrder(*arg_list)
		elif smooth_type == 'second_order':
			if dims[1] == 1:
				return _SecondOrderXZ(*arg_list)
			elif dims[0] == 1:
				return _SecondOrderYZ(*arg_list)
			else:
				return _SecondOrder(*arg_list)
		elif smooth_type == 'third_order':
			if dims[1] == 1:
				return _ThirdOrderXZ(*arg_list)
			elif dims[0] == 1:
				return _ThirdOrderYZ(*arg_list)
			else:
				return _ThirdOrder(*arg_list)
		else:
			raise ValueError(
				"Supported smoothing operators are ''first_order'', "
				"''second_order'', and ''third_order''."
			)


class _FirstOrder(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a first-order horizontal digital filter to three-dimensional fields
	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions
	match those specified at instantiation time. Hence, one should use (at least)
	one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.24, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.24.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def __call__(self, phi, phi_tnd):
		"""
		Apply first-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Write the output field into the provided array
		phi_tnd[...] = self._tnd_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((1, 1, 0), (ni-2, nj-2, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'tnd_phi': self._tnd_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil implementing first-order filtering.

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
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output field
		lap = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		tnd_phi[i, j] = 0.25 * gamma[i, j] * \
			stage_laplacian(i, j, dx, dy, in_phi, lap)[i, j]

		return tnd_phi


class _FirstOrderXZ(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a first-order horizontal digital filter to three-dimensional fields
	with only one element in the :math:`y`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.49, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend, dtype
		)

	def __call__(self, phi, phi_out):
		"""
		Apply first-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		self._out_phi[ 0, :, :] = self._in_phi[ 0, :, :]
		self._out_phi[-1, :, :] = self._in_phi[-1, :, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((1, 0, 0), (ni-2, 0, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'out_phi': self._out_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	@staticmethod
	def _stencil_defs(in_phi, gamma):
		"""
		The GT4Py stencil implementing first-order filtering.

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
		# Index
		i = gt.Index(axis=0)

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i] = (1. - 0.5 * gamma[i]) * in_phi[i] + \
					 0.25 * gamma[i] * (in_phi[i-1] + in_phi[i+1])

		return out_phi


class _FirstOrderYZ(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a first-order horizontal digital filter to three-dimensional fields
	with only one element in the :math:`x`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.49, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend, dtype
		)

	def __call__(self, phi, phi_out):
		"""
		Apply first-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layer of the output field,
		# not affected by the stencil
		self._out_phi[:,  0, :] = self._in_phi[:,  0, :]
		self._out_phi[:, -1, :] = self._in_phi[:, -1, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 1, 0), (0, nj-2, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'out_phi': self._out_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	@staticmethod
	def _stencil_defs(in_phi, gamma):
		"""
		The GT4Py stencil implementing first-order filtering.

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
		# Index
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[j] = (1. - 0.5 * gamma[j]) * in_phi[j] + \
					 0.25 * gamma[j] * (in_phi[j-1] + in_phi[j+1])

		return out_phi


class _SecondOrder(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a second-order horizontal digital filter to three-dimensional fields
	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.24, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def __call__(self, phi, phi_tnd):
		"""
		Apply second-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Write the output field into the provided array
		phi_tnd[...] = self._tnd_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((4, 4, 0), (ni-5, nj-5, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'tnd_phi': self._tnd_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		"""
		The GT4Py stencil implementing second-order filtering.

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
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Computations
		d3phi_dx3 = gt.Equation()
		d3phi_dx3[i, j] = (
			- in_phi[i-2, j] + 2*in_phi[i-1, j] - 2*in_phi[i+1, j] + in_phi[i+2, j]
		) / (2*dx**3)
		d4phi_dx4 = gt.Equation()
		d4phi_dx4[i, j] = (d3phi_dx3[i+1, j] - d3phi_dx3[i-1, j]) / (2*dx)

		d3phi_dy3 = gt.Equation()
		d3phi_dy3[i, j] = (
			- in_phi[i, j-2] + 2*in_phi[i, j-1] - 2*in_phi[i, j+1] + in_phi[i, j+2]
		) / (2*dy**3)
		d4phi_dy4 = gt.Equation()
		d4phi_dy4[i, j] = (d3phi_dy3[i, j+1] - d3phi_dy3[i, j-1]) / (2*dy)

		tnd_phi = gt.Equation()
		tnd_phi[i, j] = gamma[i, j] * (d4phi_dx4[i, j] + d4phi_dy4[i, j])

		return tnd_phi


class _SecondOrderXZ(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a second-order horizontal digital filter to three-dimensional fields
	with only one element in the :math:`y`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.49, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend, dtype
		)

	def __call__(self, phi, phi_out):
		"""
		Apply second-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		self._out_phi[ 0, :, :]    = self._in_phi[ 0, :, :]
		self._out_phi[ 1, :, :]    = self._in_phi[ 1, :, :]
		self._out_phi[-2, :, :]    = self._in_phi[-2, :, :]
		self._out_phi[-1, :, :]    = self._in_phi[-1, :, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((2, 0, 0), (ni-3, 0, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'out_phi': self._out_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	@staticmethod
	def _stencil_defs(in_phi, gamma):
		"""
		The GT4Py stencil implementing second-order filtering.

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
		# Index
		i = gt.Index(axis=0)

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i] = (1. - 0.375 * gamma[i]) * in_phi[i] + \
					 0.0625 * gamma[i] * (4. * (in_phi[i-1] + in_phi[i+1])
										  - in_phi[i-2] - in_phi[i+2])

		return out_phi


class _SecondOrderYZ(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a second-order horizontal digital filter to three-dimensional fields
	with only one element in the :math:`x`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.49, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend, dtype\
		)

	def __call__(self, phi, phi_out):
		"""
		Apply second-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		self._out_phi[:,  0, :] = self._in_phi[:,  0, :]
		self._out_phi[:,  1, :] = self._in_phi[:,  1, :]
		self._out_phi[:, -2, :] = self._in_phi[:, -2, :]
		self._out_phi[:, -1, :] = self._in_phi[:, -1, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 2, 0), (0, nj-3, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'out_phi': self._out_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	@staticmethod
	def _stencil_defs(in_phi, gamma):
		"""
		The GT4Py stencil implementing second-order filtering.

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
		# Index
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[j] = (1. - 0.375 * gamma[j]) * in_phi[j] + \
					 0.0625 * gamma[j] * (- (in_phi[j-2] + in_phi[j+2])
										  + 4. * (in_phi[j-1] + in_phi[j+1]))

		return out_phi


class _ThirdOrder(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a third-order horizontal digital filter to three-dimensional fields
	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.24, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend, dtype
		)

	def __call__(self, phi, phi_out):
		"""
		Apply second-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		self._out_phi[ 0, :, :]    = self._in_phi[ 0, :, :]
		self._out_phi[ 1, :, :]    = self._in_phi[ 1, :, :]
		self._out_phi[ 2, :, :]    = self._in_phi[ 2, :, :]
		self._out_phi[-3, :, :]    = self._in_phi[-3, :, :]
		self._out_phi[-2, :, :]    = self._in_phi[-2, :, :]
		self._out_phi[-1, :, :]    = self._in_phi[-1, :, :]
		self._out_phi[3:-3,  0, :] = self._in_phi[3:-3,  0, :]
		self._out_phi[3:-3,  1, :] = self._in_phi[3:-3,  1, :]
		self._out_phi[3:-3,  2, :] = self._in_phi[3:-3,  2, :]
		self._out_phi[3:-3, -3, :] = self._in_phi[3:-3, -3, :]
		self._out_phi[3:-3, -2, :] = self._in_phi[3:-3, -2, :]
		self._out_phi[3:-3, -1, :] = self._in_phi[3:-3, -1, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((3, 3, 0), (ni-4, nj-4, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'out_phi': self._out_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	@staticmethod
	def _stencil_defs(in_phi, gamma):
		"""
		The GT4Py stencil implementing third-order filtering.

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
		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j] = (1. - 0.625 * gamma[i, j]) * in_phi[i, j] + \
						0.015625 * gamma[i, j] * \
						(+ in_phi[i-3, j] + in_phi[i+3]
						 - 6. * (in_phi[i-2, j] + in_phi[i+2, j])
						 + 15. * (in_phi[i-1, j] + in_phi[i+1, j])
						 + 15. * (in_phi[i, j-1] + in_phi[i, j+1])
						 - 6. * (in_phi[i, j-2] + in_phi[i, j+2])
						 + in_phi[i, j-3] + in_phi[i, j+3])

		return out_phi


class _ThirdOrderXZ(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a third-order horizontal digital filter to three-dimensional fields
	with only one element in the :math:`y`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.49, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend, dtype
		)

	def __call__(self, phi, phi_out):
		"""
		Apply second-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		self._out_phi[ 0, :, :]    = self._in_phi[ 0, :, :]
		self._out_phi[ 1, :, :]    = self._in_phi[ 1, :, :]
		self._out_phi[ 2, :, :]    = self._in_phi[ 2, :, :]
		self._out_phi[-3, :, :]    = self._in_phi[-3, :, :]
		self._out_phi[-2, :, :]    = self._in_phi[-2, :, :]
		self._out_phi[-1, :, :]    = self._in_phi[-1, :, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((3, 0, 0), (ni-4, 0, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'out_phi': self._out_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	@staticmethod
	def _stencil_defs(in_phi, gamma):
		"""
		The GT4Py stencil implementing third-order filtering.

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
		# Index
		i = gt.Index(axis=0)

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i] = (1. - 0.3125 * gamma[i]) * in_phi[i] + \
					 0.015625 * gamma[i] * (+ in_phi[i-3] + in_phi[i+3]
											- 6. * (in_phi[i-2] + in_phi[i+2])
											+ 15. * (in_phi[i-1] + in_phi[i+1]))

		return out_phi


class _ThirdOrderYZ(HorizontalDiffusion):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalDiffusion`
	to apply a third-order horizontal digital filter to three-dimensional fields
	with only one element in the :math:`x`-direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, smooth_damp_depth=10, smooth_coeff=.03,
		smooth_coeff_max=.49, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
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
		smooth_damp_depth : `int`, optional
			Depth of, i.e., number of vertical regions in the damping region.
			Defaults to 10.
		smooth_coeff : `float`, optional
			Value for the smoothing coefficient far from the top boundary.
			Defaults to 0.03
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient.
			Defaults to 0.49.
		xaxis_units : `str`, optional
			TODO
		yaxis_units : `str`, optional
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical smoothing. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		super().__init__(
			dims, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, backend, dtype
		)

	def __call__(self, phi, phi_out):
		"""
		Apply second-order horizontal smoothing to a prognostic field.
		"""
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		self._out_phi[:,  0, :] = self._in_phi[:,  0, :]
		self._out_phi[:,  1, :] = self._in_phi[:,  1, :]
		self._out_phi[:,  2, :] = self._in_phi[:,  2, :]
		self._out_phi[:, -3, :] = self._in_phi[:, -3, :]
		self._out_phi[:, -2, :] = self._in_phi[:, -2, :]
		self._out_phi[:, -1, :] = self._in_phi[:, -1, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil applying horizontal smoothing.
		"""
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 3, 0), (0, nj-4, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs			 = {'out_phi': self._out_phi},
			domain			 = _domain,
			mode			 = self._backend
		)

	@staticmethod
	def _stencil_defs(in_phi, gamma):
		"""
		The GT4Py stencil implementing third-order filtering.

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
		# Index
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[j] = (1. - 0.3125 * gamma[j]) * in_phi[j] + \
					 0.015625 * gamma[j] * (+ in_phi[j-3] + in_phi[j+3]
											- 6. * (in_phi[j-2] + in_phi[j+2])
											+ 15. * (in_phi[j-1] + in_phi[j+1]))

		return out_phi
