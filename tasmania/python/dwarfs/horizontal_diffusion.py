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
	SecondOrder(HorizontalDiffusion)
	SecondOrder1D{X, Y}(HorizontalDiffusion)
	FourthOrder(HorizontalDiffusion)
	FourthOrder1D{X, Y}(HorizontalDiffusion)
"""
import abc
import math
import numpy as np

import gridtools as gt

try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


class HorizontalDiffusion:
	"""
	Abstract base class whose derived classes calculates the
	tendency due to horizontal diffusion.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
		diffusion_damp_depth, nb, backend, dtype
	):
		"""
		Parameters
		----------
		shape : tuple
			Shape of the 3-D arrays for which tendencies should be computed.
		dx : float
			The grid spacing along the first horizontal dimension.
		dy : float
			The grid spacing along the second horizontal dimension.
		diffusion_coeff : float
			Value for the diffusion coefficient far from the top boundary.
		diffusion_coeff_max : float
			Maximum value for the diffusion coefficient.
		diffusion_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		nb : int
			Number of boundary layers.
		backend : obj
			TODO
		dtype : numpy.dtype
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
		# store input arguments
		self._shape           = shape
		self._dx              = dx
		self._dy              = dy
		self._diff_coeff      = diffusion_coeff
		self._diff_coeff_max  = diffusion_coeff_max
		self._diff_damp_depth = diffusion_damp_depth
		self._nb              = nb
		self._backend		  = backend

		# initialize the diffusion coefficient
		self._gamma = self._diff_coeff

		# the diffusivity is monotonically increased towards the top of the model,
		# so to mimic the effect of a short length wave absorber
		n = self._diff_damp_depth
		if n > 0:
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
			pert = np.tile(pert[np.newaxis, np.newaxis, :], (shape[0], shape[1], 1))
			self._gamma = self._diff_coeff * np.ones(shape, dtype=dtype)
			self._gamma[:, :, :n] += (self._diff_coeff_max - self._diff_coeff) * pert

		# initialize the pointer to the underlying stencil
		# it will be properly re-directed the first time the call
		# operator is invoked
		self._stencil = None

	def __call__(self, phi, phi_tnd):
		"""
		Calculate the tendency.

		Parameters
		----------
		phi : numpy.ndarray
			The 3-D prognostic field.
		phi_tnd : numpy.ndarray
			Buffer where the calculated tendency will be written.
		"""
		# initialize the underlying GT4Py stencil
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
		diffusion_type, shape, dx, dy, diffusion_coeff,	diffusion_coeff_max,
		diffusion_damp_depth, nb=None, backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Parameters
		----------
		diffusion_type : string
			String specifying the diffusion technique to implement. Either:

			* 'second_order', for second-order computational diffusion;
			* 'fourth_order', for fourth-order computational diffusion.

		shape : tuple
			Shape of the 3-D arrays for which tendencies should be computed.
		dx : float
			The grid spacing along the first horizontal dimension.
		dy : float
			The grid spacing along the second horizontal dimension.
		diffusion_coeff : float
			Value for the diffusion coefficient far from the top boundary.
		diffusion_coeff_max : float
			Maximum value for the diffusion coefficient.
		diffusion_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		nb : `int`, optional
			Number of boundary layers. If not specified, this is derived
			from the extent of the underlying stencil.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated within
			this class.

		Return
		------
		obj :
			Instance of the appropriate derived class.
		"""
		arg_list = [
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		]

		if diffusion_type == 'second_order':
			assert not(shape[0] < 3 and shape[1] < 3)

			if shape[1] < 3:
				return SecondOrder1DX(*arg_list)
			elif shape[0] < 3:
				return SecondOrder1DY(*arg_list)
			else:
				return SecondOrder(*arg_list)
		elif diffusion_type == 'fourth_order':
			assert not(shape[0] < 5 and shape[1] < 5)

			if shape[1] < 5:
				return FourthOrder1DX(*arg_list)
			elif shape[0] < 5:
				return FourthOrder1DY(*arg_list)
			else:
				return FourthOrder(*arg_list)
		else:
			raise ValueError(
				"Supported diffusion operators are ''second_order'' "
				"and ''fourth_order''."
			)


class SecondOrder(HorizontalDiffusion):
	"""
	This class inherits	:class:`tasmania.HorizontalDiffusion`
	to calculate the tendency due to second-order horizontal diffusion for any
	three-dimensional field	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions
	match those specified at instantiation time. Hence, one should use (at least)
	one instance per field shape.
	"""
	def __init__(
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Shortcuts
		dx = self._dx
		dy = self._dy

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Temporary and output fields
		tnd_phi = gt.Equation()

		# Computations
		tnd_phi[i, j] = (self._gamma if gamma is None else gamma[i, j]) * (
			(in_phi[i-1, j] - 2.0*in_phi[i, j] + in_phi[i+1, j]) / (dx*dx) +
			(in_phi[i, j-1] - 2.0*in_phi[i, j] + in_phi[i, j+1]) / (dy*dy)
		)

		return tnd_phi


class SecondOrder1DX(HorizontalDiffusion):
	"""
	This class inherits	:class:`tasmania.HorizontalDiffusion`
	to calculate the tendency due to second-order horizontal diffusion for any
	three-dimensional field	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Shortcuts
		dx = self._dx

		# Index
		i = gt.Index(axis=0)

		# Temporary and output fields
		tnd_phi = gt.Equation()

		# Computations
		tnd_phi[i] = (self._gamma if gamma is None else gamma[i]) * \
			(in_phi[i-1] - 2.0*in_phi[i] + in_phi[i+1]) / (dx*dx)

		return tnd_phi


class SecondOrder1DY(HorizontalDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalDiffusion`
	to calculate the tendency due to second-order horizontal diffusion for any
	three-dimensional field	with only one element along the first dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Shortcuts
		dy = self._dy

		# Index
		j = gt.Index(axis=1)

		# Temporary and output fields
		tnd_phi = gt.Equation()

		# Computations
		tnd_phi[j] = (self._gamma if gamma is None else gamma[j]) * \
			(in_phi[j-1] - 2.0*in_phi[j] + in_phi[j+1]) / (dy*dy)

		return tnd_phi


class FourthOrder(HorizontalDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalDiffusion`
	to calculate the tendency due to fourth-order horizontal diffusion for any
	three-dimensional field	with at least three elements in each direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Shortcuts
		dx = self._dx
		dy = self._dy

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Temporary and output fields
		tnd_phi = gt.Equation()

		# Computations
		tnd_phi[i, j] = (self._gamma if gamma is None else gamma[i, j]) * \
			(
				(
					- in_phi[i-2, j] + 16.0*in_phi[i-1, j]
					- 30.0*in_phi[i, j]
					+ 16.0*in_phi[i+1, j] - in_phi[i+2, j]
				) / (12.0 * dx * dx)
				+
				(
					- in_phi[i, j-2] + 16.0*in_phi[i, j-1]
					- 30.0*in_phi[i, j]
					+ 16.0*in_phi[i, j+1] - in_phi[i, j+2]
				) / (12.0 * dy * dy)
			)

		return tnd_phi


class FourthOrder1DX(HorizontalDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalDiffusion`
	to calculate the tendency due to fourth-order horizontal diffusion for any
	three-dimensional field	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Shortcuts
		dx = self._dx

		# Index
		i = gt.Index(axis=0)

		# Temporary and output fields
		tnd_phi = gt.Equation()

		# Computations
		tnd_phi[i] = (self._gamma if gamma is None else gamma[i]) * \
			(
				- in_phi[i-2] + 16.0*in_phi[i-1]
				- 30.0*in_phi[i]
				+ 16.0*in_phi[i+1] - in_phi[i+2]
			) / (12.0 * dx * dx)

		return tnd_phi


class FourthOrder1DY(HorizontalDiffusion):
	"""
	This class inherits	:class:`~tasmania.HorizontalDiffusion`
	to calculate the tendency due to fourth-order horizontal diffusion for any
	three-dimensional field	with only one element along the first dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Shortcuts
		dy = self._dy

		# Index
		j = gt.Index(axis=1)

		# Temporary and output fields
		tnd_phi = gt.Equation()

		# Computations
		tnd_phi[j] = (self._gamma if gamma is None else gamma[j]) * \
			(
				- in_phi[j-2] + 16.0*in_phi[j-1]
				- 30.0*in_phi[j]
				+ 16.0*in_phi[j+1] - in_phi[j+2]
			) / (12.0 * dy * dy)

		return tnd_phi
