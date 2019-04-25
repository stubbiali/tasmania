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
	FirstOrder(HorizontalHyperDiffusion)
	FirstOrder{1DX, 1DY}(HorizontalHyperDiffusion)
	SecondOrder(HorizontalHyperDiffusion)
	SecondOrder{1DX, 1DY}(HorizontalHyperDiffusion)
	ThirdOrder(HorizontalHyperDiffusion)
	ThirdOrder{1DX, 1DY}(HorizontalHyperDiffusion)
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
		diffusion_type, shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
		diffusion_damp_depth, nb=None, backend=gt.mode.NUMPY, dtype=datatype
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
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.

		Return
		------
		obj :
			Instance of the appropriate derived class.
		"""
		arg_list = [
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		]

		if diffusion_type == 'first_order':
			assert not(shape[0] < 3 and shape[1] < 3)

			if shape[1] < 3:
				return FirstOrder1DX(*arg_list)
			elif shape[0] < 3:
				return FirstOrder1DY(*arg_list)
			else:
				return FirstOrder(*arg_list)
		elif diffusion_type == 'second_order':
			assert not(shape[0] < 5 and shape[1] < 5)

			if shape[1] < 5:
				return SecondOrder1DX(*arg_list)
			elif shape[0] < 5:
				return SecondOrder1DY(*arg_list)
			else:
				return SecondOrder(*arg_list)
		elif diffusion_type == 'third_order':
			assert not(shape[0] < 7 and shape[1] < 7)

			if shape[1] < 7:
				return ThirdOrder1DX(*arg_list)
			elif shape[0] < 7:
				return ThirdOrder1DY(*arg_list)
			else:
				return ThirdOrder(*arg_list)
		else:
			raise ValueError(
				"Supported diffusion operators are ''first_order'', "
				"''second_order'', and ''third_order''."
			)


class FirstOrder(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dx, dy = self._dx, self._dy

		# indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		lap = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian(i, j, dx, dy, in_phi, lap)
		tnd_phi[i, j] = (self._gamma if gamma is None else gamma[i, j]) * lap[i, j]

		return tnd_phi


class FirstOrder1DX(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dx = self._dx

		# index
		i = gt.Index(axis=0)

		# temporary and output fields
		lap = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian_1d(i, dx, in_phi, lap)
		tnd_phi[i] = (self._gamma if gamma is None else gamma[i]) * lap[i]

		return tnd_phi


class FirstOrder1DY(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=1, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dy = self._dy

		# index
		j = gt.Index(axis=1)

		# temporary and output fields
		lap = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian_1d(j, dy, in_phi, lap)
		tnd_phi[j] = (self._gamma if gamma is None else gamma[j]) * lap[j]

		return tnd_phi


class SecondOrder(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dx, dy = self._dx, self._dy

		# indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian(i, j, dx, dy, in_phi, lap0)
		stage_laplacian(i, j, dx, dy, lap0, lap1)
		tnd_phi[i, j] = (self._gamma if gamma is None else gamma[i, j]) * lap1[i, j]

		return tnd_phi


class SecondOrder1DX(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dx = self._dx

		# index
		i = gt.Index(axis=0)

		# temporary and output fields
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian_1d(i, dx, in_phi, lap0)
		stage_laplacian_1d(i, dx, lap0, lap1)
		tnd_phi[i] = (self._gamma if gamma is None else gamma[i]) * lap1[i]

		return tnd_phi


class SecondOrder1DY(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=2, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dy = self._dy

		# index
		j = gt.Index(axis=1)

		# temporary and output fields
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian_1d(j, dy, in_phi, lap0)
		stage_laplacian_1d(j, dy, lap0, lap1)
		tnd_phi[j] = (self._gamma if gamma is None else gamma[j]) * lap1[j]

		return tnd_phi


class ThirdOrder(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=3, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((nb, nb, 0), (ni-nb-1, nj-nb-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dx, dy = self._dx, self._dy

		# indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output field
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		lap2 = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian(i, j, dx, dy, in_phi, lap0)
		stage_laplacian(i, j, dx, dy, lap0, lap1)
		stage_laplacian(i, j, dx, dy, lap1, lap2)
		tnd_phi[i, j] = (self._gamma if gamma is None else gamma[i, j]) * lap2[i, j]

		return tnd_phi


class ThirdOrder1DX(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=3, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((nb, 0, 0), (ni-nb-1, nj-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dx = self._dx

		# indices
		i = gt.Index(axis=0)

		# temporary and output field
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		lap2 = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian_1d(i, dx, in_phi, lap0)
		stage_laplacian_1d(i, dx, lap0, lap1)
		stage_laplacian_1d(i, dx, lap1, lap2)
		tnd_phi[i] = (self._gamma if gamma is None else gamma[i]) * lap2[i]

		return tnd_phi


class ThirdOrder1DY(HorizontalHyperDiffusion):
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
		self, shape, dx, dy, diffusion_coeff=1.0, diffusion_coeff_max=1.0,
		diffusion_damp_depth=10, nb=3, backend=gt.mode.NUMPY, dtype=datatype
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, dx, dy, diffusion_coeff, diffusion_coeff_max,
			diffusion_damp_depth, nb, backend, dtype
		)

	def _stencil_initialize(self, dtype):
		# shortcuts
		ni, nj, nk = self._shape
		nb = self._nb

		# allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		self._in_phi = np.zeros((ni, nj, nk), dtype=dtype)
		self._tnd_phi = np.zeros((ni, nj, nk), dtype=dtype)

		# set stencil's inputs
		_inputs = {'in_phi': self._in_phi}
		if self._diff_damp_depth > 0:
			_inputs['gamma'] = self._gamma

		# set the computational domain
		_domain = gt.domain.Rectangle((0, nb, 0), (ni-1, nj-nb-1, nk-1))

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs={'tnd_phi': self._tnd_phi},
			domain=_domain,
			mode=self._backend
		)

	def _stencil_defs(self, in_phi, gamma=None):
		# shortcuts
		dy = self._dy

		# indices
		j = gt.Index(axis=1)

		# temporary and output field
		lap0 = gt.Equation()
		lap1 = gt.Equation()
		lap2 = gt.Equation()
		tnd_phi = gt.Equation()

		# computations
		stage_laplacian_1d(j, dy, in_phi, lap0)
		stage_laplacian_1d(j, dy, lap0, lap1)
		stage_laplacian_1d(j, dy, lap1, lap2)
		tnd_phi[j] = (self._gamma if gamma is None else gamma[j]) * lap2[j]

		return tnd_phi
