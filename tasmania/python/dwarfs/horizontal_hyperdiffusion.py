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
	HorizontalHyperDiffusion
	_FirstOrder(HorizontalHyperDiffusion)
	_FirstOrder{XZ, YZ}(HorizontalHyperDiffusion)
	_SecondOrder(HorizontalHyperDiffusion)
	_SecondOrder{XZ, YZ}(HorizontalHyperDiffusion)
	_ThirdOrder(HorizontalHyperDiffusion)
	_ThirdOrder{XZ, YZ}(HorizontalHyperDiffusion)
"""
import abc
import math
import numpy as np

import gridtools as gt

try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


def stage_laplacian(i, j, dx, dy, in_phi, tnd_phi):
	tnd_phi[i, j] = \
		(in_phi[i-1, j] - 2*in_phi[i, j] + in_phi[i+1, j]) / (dx*dx) + \
		(in_phi[i, j-1] - 2*in_phi[i, j] + in_phi[i, j+1]) / (dy*dy)
	return tnd_phi


def stage_laplacian_1d(i, dx, in_phi, tnd_phi):
	tnd_phi[i] = (in_phi[i-1] - 2*in_phi[i] + in_phi[i+1]) / (dx*dx)
	return tnd_phi


class HorizontalHyperDiffusion:
	"""
	Abstract base class whose derived classes calculates the
	tendency due to horizontal hyper-diffusion.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, dims, grid, diffusion_damp_depth, diffusion_coeff,
		diffusion_coeff_max, xaxis_units, yaxis_units, backend, dtype
	):
		"""
		Parameters
		----------
		dims : tuple
			Shape of the (three-dimensional) arrays for which
			tendencies should be computed.
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.python.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		diffusion_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		diffusion_coeff : float
			Value for the diffusion coefficient far from the top boundary.
		diffusion_coeff_max : float
			Maximum value for the diffusion coefficient.
		xaxis_units : str
			TODO
		yaxis_units : str
			TODO
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical diffusion.
		dtype : obj
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
		"""
		# Store input arguments
		self._dims              = dims
		self._grid              = grid
		self._diff_damp_depth   = diffusion_damp_depth
		self._diff_coeff        = diffusion_coeff
		self._diff_coeff_max    = diffusion_coeff_max
		self._xunits			= xaxis_units
		self._yunits			= yaxis_units
		self._backend			= backend

		# Initialize the diffusion matrix
		self._gamma = self._diff_coeff * np.ones(self._dims, dtype=dtype)

		# The diffusivity is monotonically increased towards the top of the model,
		# so to mimic the effect of a short length wave absorber
		n = self._diff_damp_depth
		if n > 0:
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
			pert = np.tile(pert[np.newaxis, np.newaxis, :], (dims[0], dims[1], 1))
			self._gamma[:, :, :n] += (self._diff_coeff_max - self._diff_coeff) * pert

		# Initialize the pointer to the underlying stencil
		# It will be properly re-directed the first time the call
		# operator is invoked
		self._stencil = None

	def __call__(self, phi, phi_tnd):
		"""
		Calculate the tendency.

		Parameters
		----------
		phi : :class:`numpy.ndarray`
			The 3-D prognostic field.
		phi_tnd : :class:`numpy.ndarray`
			Buffer into which the calculated tendency is written.
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

	@abc.abstractmethod
	def _stencil_initialize(self, dtype):
		"""
		Initialize the underlying GT4Py stencil.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.
		"""

	@staticmethod
	def factory(
		diffusion_type, dims, grid, diffusion_damp_depth, diffusion_coeff,
		diffusion_coeff_max, xaxis_units, yaxis_units,
		backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Static method returning an instance of the derived class
		calculating the tendency due to horizontal hyper-diffusion of type
		:data:`diffusion_type`.

		Parameters
		----------
		diffusion_type : string
			String specifying the diffusion technique to implement. Either:

			* 'first_order', for first-order numerical hyper-diffusion;
			* 'second_order', for second-order numerical hyper-diffusion;
			* 'third_order', for third-order numerical hyper-diffusion.

		dims : tuple
			Shape of the (three-dimensional) arrays on which
			to apply numerical diffusion.
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		diffusion_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		diffusion_coeff : float
			Value for the diffusion coefficient far from the top boundary.
		diffusion_coeff_max : float
			Maximum value for the diffusion coefficient.
		xaxis_units : str
			TODO
		yaxis_units : str
			TODO
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencil
			implementing numerical diffusion. Defaults to :class:`gridtools.mode.NUMPY`.
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
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		]

		if diffusion_type == 'first_order':
			assert not(dims[0] < 3 and dims[1] < 3)

			if dims[1] < 3:
				return _FirstOrderXZ(*arg_list)
			elif dims[0] < 3:
				return _FirstOrderYZ(*arg_list)
			else:
				return _FirstOrder(*arg_list)
		elif diffusion_type == 'second_order':
			assert not(dims[0] < 5 and dims[1] < 5)

			if dims[1] < 5:
				return _SecondOrderXZ(*arg_list)
			elif dims[0] < 5:
				return _SecondOrderYZ(*arg_list)
			else:
				return _SecondOrder(*arg_list)
		elif diffusion_type == 'third_order':
			assert not(dims[0] < 7 and dims[1] < 7)

			if dims[1] < 7:
				return _ThirdOrderXZ(*arg_list)
			elif dims[0] < 7:
				return _ThirdOrderYZ(*arg_list)
			else:
				return _ThirdOrder(*arg_list)
		else:
			raise ValueError(
				"Supported diffusion operators are ''first_order'', "
				"''second_order'', and ''third_order''."
			)


class _FirstOrder(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to first-order horizontal hyper-diffusion for any
	three-dimensional field	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions
	match those specified at instantiation time. Hence, one should use (at least)
	one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
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
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Temporary and output fields
		lap = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian(i, j, dx, dy, in_phi, lap)
		tnd_phi[i, j] = gamma[i, j] * lap[i, j]

		return tnd_phi


class _FirstOrderXZ(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to first-order horizontal hyper-diffusion for any
	three-dimensional field	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((1, 0, 0), (ni-2, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()

		# Index
		i = gt.Index(axis=0)

		# Temporary and output fields
		lap = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian_1d(i, dx, in_phi, lap)
		tnd_phi[i] = gamma[i] * lap[i]

		return tnd_phi


class _FirstOrderYZ(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to first-order horizontal hyper-diffusion for any
	three-dimensional field	with only one element along the first dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 1, 0), (ni-1, nj-2, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Index
		j = gt.Index(axis=1)

		# Temporary and output fields
		lap = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian_1d(j, dy, in_phi, lap)
		tnd_phi[j] = gamma[j] * lap[j]

		return tnd_phi


class _SecondOrder(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to second-order horizontal hyper-diffusion for any
	three-dimensional field	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((2, 2, 0), (ni-3, nj-3, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Temporary and output fields
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian(i, j, dx, dy, in_phi, lap0)
		stage_laplacian(i, j, dx, dy, lap0, lap1)
		tnd_phi[i, j] = gamma[i, j] * lap1[i, j]

		return tnd_phi


class _SecondOrderXZ(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to second-order horizontal hyper-diffusion for any
	three-dimensional field	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((2, 0, 0), (ni-3, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()

		# Index
		i = gt.Index(axis=0)

		# Temporary and output fields
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian_1d(i, dx, in_phi, lap0)
		stage_laplacian_1d(i, dx, lap0, lap1)
		tnd_phi[i] = gamma[i] * lap1[i]

		return tnd_phi


class _SecondOrderYZ(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to second-order horizontal hyper-diffusion for any
	three-dimensional field	with only one element along the first dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 2, 0), (ni-1, nj-3, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Index
		j = gt.Index(axis=1)

		# Temporary and output fields
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian_1d(j, dy, in_phi, lap0)
		stage_laplacian_1d(j, dy, lap0, lap1)
		tnd_phi[j] = gamma[j] * lap1[j]

		return tnd_phi


class _ThirdOrder(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to third-order horizontal hyper-diffusion for any
	three-dimensional field	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((3, 3, 0), (ni-4, nj-4, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Temporary and output field
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		lap2 = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian(i, j, dx, dy, in_phi, lap0)
		stage_laplacian(i, j, dx, dy, lap0, lap1)
		stage_laplacian(i, j, dx, dy, lap1, lap2)
		tnd_phi[i, j] = gamma[i, j] * lap2[i, j]

		return tnd_phi


class _ThirdOrderXZ(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to third-order horizontal hyper-diffusion for any
	three-dimensional field	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((3, 0, 0), (ni-4, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dx = self._grid.dx.to_units(self._xunits).values.item()

		# Indices
		i = gt.Index(axis=0)

		# Temporary and output field
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		lap2 = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian_1d(i, dx, in_phi, lap0)
		stage_laplacian_1d(i, dx, lap0, lap1)
		stage_laplacian_1d(i, dx, lap1, lap2)
		tnd_phi[i] = gamma[i] * lap2[i]

		return tnd_phi


class _ThirdOrderYZ(HorizontalHyperDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalHyperDiffusion`
	to calculate the tendency due to third-order horizontal hyper-diffusion for any
	three-dimensional field	with only one element along the first dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, dims, grid, diffusion_damp_depth=10, diffusion_coeff=1.0,
		diffusion_coeff_max=1.0, xaxis_units='m', yaxis_units='m',
		backend=gt.mode.NUMPY, dtype=datatype
	):
		super().__init__(
			dims, grid, diffusion_damp_depth, diffusion_coeff, diffusion_coeff_max,
			xaxis_units, yaxis_units, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._dims

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, 3, 0), (ni-1, nj-4, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_phi': self._in_phi, 'gamma': self._gamma},
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma):
		# Shortcuts
		dy = self._grid.dy.to_units(self._yunits).values.item()

		# Indices
		j = gt.Index(axis=1)

		# Temporary and output field
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		lap2 = gt.Equation()
		tnd_phi = gt.Equation()

		# Computations
		stage_laplacian_1d(j, dy, in_phi, lap0)
		stage_laplacian_1d(j, dy, lap0, lap1)
		stage_laplacian_1d(j, dy, lap1, lap2)
		tnd_phi[j] = gamma[j] * lap2[j]

		return tnd_phi
