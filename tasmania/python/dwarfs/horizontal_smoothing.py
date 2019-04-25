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
	HorizontalSmoothing
	FirstOrder(HorizontalSmoothing)
	FirstOrder{1DX, 1DY}(HorizontalSmoothing)
	SecondOrder(HorizontalSmoothing)
	SecondOrder{1DX, 1DY}(HorizontalSmoothing)
	ThirdOrder(HorizontalSmoothing)
	ThirdOrder{1DX, 1DY}(HorizontalSmoothing)
"""
import abc
import math
import numpy as np

import gridtools as gt
try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


class HorizontalSmoothing:
	"""
	Abstract base class whose derived classes apply horizontal
	numerical smoothing to a generic (prognostic) field.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
		nb, backend, dtype
	):
		"""
		Parameters
		----------
		shape : tuple
			Shape of the 3-D arrays which should be filtered.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient.
		smooth_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		nb : int
			Number of boundary layers.
		backend : obj
			TODO
		dtype : numpy.dtype
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
		# Store input arguments
		self._shape             = shape
		self._smooth_damp_depth = smooth_damp_depth
		self._smooth_coeff      = smooth_coeff
		self._smooth_coeff_max  = smooth_coeff_max
		self._nb				= nb
		self._backend			= backend

		# initialize the diffusion coefficient
		self._gamma = self._smooth_coeff

		# the diffusivity is monotonically increased towards the top of the model,
		# so to mimic the effect of a short length wave absorber
		n = self._smooth_damp_depth
		if n > 0:
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
			pert = np.tile(pert[np.newaxis, np.newaxis, :], (shape[0], shape[1], 1))
			self._gamma = self._smooth_coeff * np.ones(shape, dtype=dtype)
			self._gamma[:, :, :n] += (self._smooth_coeff_max - self._smooth_coeff) * pert

		# initialize the pointer to the underlying stencil
		# it will be properly re-directed the first time the call
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
	def factory(
		smooth_type, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
		nb=None, backend=gt.mode.NUMPY, dtype=datatype
	):
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

		shape : tuple
			Shape of the 3-D arrays which should be filtered.
		smooth_coeff : float
			Value for the smoothing coefficient far from the top boundary.
		smooth_coeff_max : float
			Maximum value for the smoothing coefficient.
		smooth_damp_depth : int
			Depth of, i.e., number of vertical regions in the damping region.
		nb : int
			Number of boundary layers.
		backend : obj
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.

		Return
		------
		obj :
			Instance of the suitable derived class.
		"""
		arg_list = [
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		]

		if smooth_type == 'first_order':
			assert not(shape[0] < 3 and shape[1] < 3)

			if shape[1] < 3:
				return FirstOrder1DX(*arg_list)
			elif shape[0] < 3:
				return FirstOrder1DY(*arg_list)
			else:
				return FirstOrder(*arg_list)
		elif smooth_type == 'second_order':
			assert not(shape[0] < 5 and shape[1] < 5)

			if shape[1] < 5:
				return SecondOrder1DX(*arg_list)
			elif shape[0] < 5:
				return SecondOrder1DY(*arg_list)
			else:
				return SecondOrder(*arg_list)
		elif smooth_type == 'third_order':
			assert not(shape[0] < 7 and shape[1] < 7)

			if shape[1] < 7:
				return ThirdOrder1DX(*arg_list)
			elif shape[0] < 7:
				return ThirdOrder1DY(*arg_list)
			else:
				return ThirdOrder(*arg_list)
		else:
			raise ValueError(
				"Supported smoothing operators are ''first_order'', "
				"''second_order'', and ''third_order''."
			)


class FirstOrder(HorizontalSmoothing):
	"""
	This class inherits :class:`~tasmania.HorizontalSmoothing`
	to apply a first-order horizontal digital filter to three-dimensional fields
	with at least three elements along each dimension.

	Note
	----
	An instance of this class should only be applied to fields whose dimensions
	match those specified at instantiation time. Hence, one should use (at least)
	one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layer of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:nb, :]       = self._in_phi[:nb, :]
		self._out_phi[-nb:, :]      = self._in_phi[-nb:, :]
		self._out_phi[nb:-nb, :nb]  = self._in_phi[nb:-nb,  :nb]
		self._out_phi[nb:-nb, -nb:] = self._in_phi[nb:-nb, -nb:]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[i, j]
		out_phi[i, j] = (1 - g) * in_phi[i, j] + \
			0.25 * g * (
				in_phi[i-1, j] + in_phi[i+1, j] +
				in_phi[i, j-1] + in_phi[i, j+1]
			)

		return out_phi


class FirstOrder1DX(HorizontalSmoothing):
	"""
	This class inherits	:class:`~tasmania.HorizontalSmoothing`
	to apply a first-order horizontal digital filter to three-dimensional fields
	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:nb, :]  = self._in_phi[:nb, :]
		self._out_phi[-nb:, :] = self._in_phi[-nb:, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Index
		i = gt.Index(axis=0)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[i]
		out_phi[i] = (1. - 0.5 * g) * in_phi[i] + \
			0.25 * g * (in_phi[i-1] + in_phi[i+1])

		return out_phi


class FirstOrder1DY(HorizontalSmoothing):
	"""
	This class inherits :class:`~tasmania.HorizontalSmoothing`
	to apply a first-order horizontal digital filter to three-dimensional fields
	with only one element along the first direction.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layer of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:, :nb]  = self._in_phi[:, :nb]
		self._out_phi[:, -nb:] = self._in_phi[:, -nb:]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Index
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[j]
		out_phi[j] = (1. - 0.5 * g) * in_phi[j] + \
			0.25 * g * (in_phi[j-1] + in_phi[j+1])

		return out_phi


class SecondOrder(HorizontalSmoothing):
	"""
	This class inherits	:class:`~tasmania.HorizontalSmoothing`
	to apply a second-order horizontal digital filter to three-dimensional fields
	with at least three elements along each dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:nb, :]       = self._in_phi[:nb, :]
		self._out_phi[-nb:, :]      = self._in_phi[-nb:, :]
		self._out_phi[nb:-nb, :nb]  = self._in_phi[nb:-nb,  :nb]
		self._out_phi[nb:-nb, -nb:] = self._in_phi[nb:-nb, -nb:]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[i, j]
		out_phi[i, j] = (1. - 0.75 * g) * in_phi[i, j] + \
			0.0625 * g * (
				- in_phi[i-2, j] + 4. * in_phi[i-1, j]
				- in_phi[i+2, j] + 4. * in_phi[i+1, j]
				- in_phi[i, j-2] + 4. * in_phi[i, j-1]
				- in_phi[i, j+2] + 4. * in_phi[i, j+1]
			)

		return out_phi


class SecondOrder1DX(HorizontalSmoothing):
	"""
	This class inherits	:class:`~tasmania.HorizontalSmoothing`
	to apply a second-order horizontal digital filter to three-dimensional fields
	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:nb, :]  = self._in_phi[:nb, :]
		self._out_phi[-nb:, :] = self._in_phi[-nb:, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Index
		i = gt.Index(axis=0)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[i]
		out_phi[i] = (1. - 0.375 * g) * in_phi[i] + \
			0.0625 * g * (
				- in_phi[i-2] + 4. * in_phi[i-1]
				- in_phi[i+2] + 4. * in_phi[i+1]
			)

		return out_phi


class SecondOrder1DY(HorizontalSmoothing):
	"""
	This class inherits	:class:`~tasmania.HorizontalSmoothing`
	to apply a second-order horizontal digital filter to three-dimensional fields
	with only one element along the first dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:, :nb]  = self._in_phi[:, :nb]
		self._out_phi[:, -nb:] = self._in_phi[:, -nb:]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Index
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[j]
		out_phi[j] = (1. - 0.375 * g) * in_phi[j] + \
			0.0625 * g * (
				 - in_phi[j-2] + 4. * in_phi[j-1]
				 - in_phi[j+2] + 4. * in_phi[j+1]
			)

		return out_phi


class ThirdOrder(HorizontalSmoothing):
	"""
	This class inherits :class:`~tasmania.HorizontalSmoothing`
	to apply a third-order horizontal digital filter to three-dimensional fields
	with at least three elements along each dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=3, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:nb, :]       = self._in_phi[:nb, :]
		self._out_phi[-nb:, :]      = self._in_phi[-nb:, :]
		self._out_phi[nb:-nb, :nb]  = self._in_phi[nb:-nb,  :nb]
		self._out_phi[nb:-nb, -nb:] = self._in_phi[nb:-nb, -nb:]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[i, j]
		out_phi[i, j] = (1. - 0.625 * g) * in_phi[i, j] + \
			0.015625 * g * (
				in_phi[i-3, j] - 6*in_phi[i-2, j] + 15*in_phi[i-1, j] +
				in_phi[i+3, j] - 6*in_phi[i+2, j] + 15*in_phi[i+1, j] +
				in_phi[i, j-3] - 6*in_phi[i, j-2] + 15*in_phi[i, j-1] +
				in_phi[i, j+3] - 6*in_phi[i, j+2] + 15*in_phi[i, j+1]
			)

		return out_phi


class ThirdOrder1DX(HorizontalSmoothing):
	"""
	This class inherits :class:`~tasmania.HorizontalSmoothing`
	to apply a third-order horizontal digital filter to three-dimensional fields
	with only one element along the second dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=3, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:nb, :]  = self._in_phi[:nb, :]
		self._out_phi[-nb:, :] = self._in_phi[-nb:, :]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Index
		i = gt.Index(axis=0)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[i]
		out_phi[i] = (1. - 0.3125 * g) * in_phi[i] + \
			0.015625 * g * (
				in_phi[i-3] - 6*in_phi[i-2] + 15*in_phi[i-1] +
				in_phi[i+3] - 6*in_phi[i+2] + 15*in_phi[i+1]
			)

		return out_phi


class ThirdOrder1DY(HorizontalSmoothing):
	"""
	This class inherits	:class:`~tasmania.HorizontalSmoothing`
	to apply a third-order horizontal digital filter to three-dimensional fields
	with only one element along the first dimension.

	Note
	----
	An instance of this class should only be applied to fields whose
	dimensions match those specified at instantiation time.
	Hence, one should use (at least) one instance per field shape.
	"""
	def __init__(
		self, shape, smooth_coeff=1.0, smooth_coeff_max=1.0, smooth_damp_depth=10,
		nb=3, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
			nb, backend, dtype
		)

	def __call__(self, phi, phi_out):
		# Initialize the underlying GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(phi.dtype)

		# Update the Numpy array representing the stencil's input field
		self._in_phi[...] = phi[...]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layers of the output field,
		# not affected by the stencil
		nb = self._nb
		self._out_phi[:, :nb]  = self._in_phi[:, :nb]
		self._out_phi[:, -nb:] = self._in_phi[:, -nb:]

		# Write the output field into the provided array
		phi_out[...] = self._out_phi[...]

	def _stencil_initialize(self, dtype):
		# Shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._out_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._smooth_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# Set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'out_phi': self._out_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# Index
		j = gt.Index(axis=1)

		# Output field
		out_phi = gt.Equation()

		# Computations
		g = self._gamma if gamma is None else gamma[j]
		out_phi[j] = (1. - 0.3125 * g) * in_phi[j] + \
			0.015625 * g * (
				in_phi[j-3] - 6*in_phi[j-2] + 15*in_phi[j-1] +
				in_phi[j+3] - 6*in_phi[j+2] + 15*in_phi[j+1]
			)

		return out_phi
