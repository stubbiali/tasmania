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
from tasmania.python.utils.storage_utils import get_storage_descriptor

try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


class HorizontalSmoothing(abc.ABC):
	"""
	Abstract base class whose derived classes apply horizontal
	numerical smoothing to a generic (prognostic) field.
	"""
	def __init__(
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
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
		backend : str
			TODO
		backend_opts : dict
			TODO
		build_info : dict
			TODO
		dtype : numpy.dtype
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		exec_info : dict
			TODO
		halo : tuple
			TODO
		rebuild : bool
			TODO
		"""
		# store input arguments needed at run-time
		self._shape = shape
		self._nb = nb
		self._exec_info = exec_info

		# initialize the diffusivity
		gamma = smooth_coeff * np.ones((1, 1, shape[2]), dtype=dtype)

		# the diffusivity is monotonically increased towards the top of the model,
		# so to mimic the effect of a short-length wave absorber
		n = smooth_damp_depth
		if n > 0:
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
			gamma[:, :, :n] += (smooth_coeff_max - smooth_coeff) * pert

		# convert diffusivity to gt4py storage
		descriptor = get_storage_descriptor(shape, dtype, halo, mask=(True, True, True))  # mask=(False, False, True)
		self._gamma = gt.storage.from_array(gamma, descriptor, backend=backend)

		# initialize the underlying stencil
		decorator = gt.stencil(
			backend, backend_opts=backend_opts, build_info=build_info, rebuild=rebuild
		)
		self._stencil = decorator(self._stencil_defs)

	@abc.abstractmethod
	def __call__(self, phi, phi_out):
		"""
		Apply horizontal smoothing to a prognostic field.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		phi : gridtools.storage.Storage
			The 3-D field to filter.
		phi_out : array_like
			The 3-D buffer into which the filtered field is written.
		"""
		pass

	@staticmethod
	def factory(
		smooth_type, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,
		nb=None, *, backend="numpy", backend_opts=None, build_info=None,
		dtype=datatype, exec_info=None, halo=None, rebuild=False
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
		nb : `int`, optional
			Number of boundary layers.
		backend : `str`, optional
			TODO
		backend_opts : `dict`, optional
			TODO
		build_info : `dict`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		exec_info : `dict`, optional
			TODO
		halo : `tuple`, optional
			TODO
		rebuild : `bool`, optional
			TODO

		Return
		------
		obj :
			Instance of the suitable derived class.
		"""
		args = [
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		]

		if smooth_type == 'first_order':
			assert not(shape[0] < 3 and shape[1] < 3)

			if shape[1] < 3:
				return FirstOrder1DX(*args)
			elif shape[0] < 3:
				return FirstOrder1DY(*args)
			else:
				return FirstOrder(*args)
		elif smooth_type == 'second_order':
			assert not(shape[0] < 5 and shape[1] < 5)

			if shape[1] < 5:
				return SecondOrder1DX(*args)
			elif shape[0] < 5:
				return SecondOrder1DY(*args)
			else:
				return SecondOrder(*args)
		elif smooth_type == 'third_order':
			assert not(shape[0] < 7 and shape[1] < 7)

			if shape[1] < 7:
				return ThirdOrder1DX(*args)
			elif shape[0] < 7:
				return ThirdOrder1DY(*args)
			else:
				return ThirdOrder(*args)
		else:
			raise ValueError(
				"Supported smoothing operators are ''first_order'', "
				"''second_order'', and ''third_order''."
			)

	@staticmethod
	@abc.abstractmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		pass


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (nb, nb, 0)}, domain=(nx-2*nb, ny-2*nb, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:nb, :]       = phi.data[:nb, :]
		phi_out.data[-nb:, :]      = phi.data[-nb:, :]
		phi_out.data[nb:-nb, :nb]  = phi.data[nb:-nb,  :nb]
		phi_out.data[nb:-nb, -nb:] = phi.data[nb:-nb, -nb:]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.25 * in_gamma[0, 0, 0] * (
				in_phi[-1, 0, 0] + in_phi[1, 0, 0] +
				in_phi[0, -1, 0] + in_phi[0, 1, 0]
			)


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (nb, 0, 0)}, domain=(nx-2*nb, ny, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:nb, :]  = phi.data[:nb, :]
		phi_out.data[-nb:, :] = phi.data[-nb:, :]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.5 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.25 * in_gamma[0, 0, 0] * (in_phi[-1, 0, 0] + in_phi[1, 0, 0])


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 1 if (nb is None or nb < 1) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"in_phi": (0, nb, 0), "in_gamma": (0, nb, 0), "out_phi": (0, nb, 0)},
			domain=(nx, ny-2*nb, nz), exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:, :nb]  = phi.data[:, :nb]
		phi_out.data[:, -nb:] = phi.data[:, -nb:]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.5 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.25 * in_gamma[0, 0, 0] * (in_phi[0, -1, 0] + in_phi[0, 1, 0])


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (nb, nb, 0)}, domain=(nx-2*nb, ny-2*nb, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:nb, :]       = phi.data[:nb, :]
		phi_out.data[-nb:, :]      = phi.data[-nb:, :]
		phi_out.data[nb:-nb, :nb]  = phi.data[nb:-nb,  :nb]
		phi_out.data[nb:-nb, -nb:] = phi.data[nb:-nb, -nb:]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.75 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.0625 * in_gamma[0, 0, 0] * (
				- in_phi[-2, 0, 0] + 4.0 * in_phi[-1, 0, 0]
				- in_phi[+2, 0, 0] + 4.0 * in_phi[+1, 0, 0]
				- in_phi[0, -2, 0] + 4.0 * in_phi[0, -1, 0]
				- in_phi[0, +2, 0] + 4.0 * in_phi[0, +1, 0]
			)


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (nb, 0, 0)}, domain=(nx-2*nb, ny, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:nb, :]  = phi.data[:nb, :]
		phi_out.data[-nb:, :] = phi.data[-nb:, :]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.375 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.0625 * in_gamma[0, 0, 0] * (
				- in_phi[-2, 0, 0] + 4.0 * in_phi[-1, 0, 0]
				- in_phi[+2, 0, 0] + 4.0 * in_phi[+1, 0, 0]
			)


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 2 if (nb is None or nb < 2) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (0, nb, 0)}, domain=(nx, ny-2*nb, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:, :nb]  = phi.data[:, :nb]
		phi_out.data[:, -nb:] = phi.data[:, -nb:]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.375 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.0625 * in_gamma[0, 0, 0] * (
				- in_phi[0, -2, 0] + 4.0 * in_phi[0, -1, 0]
				- in_phi[0, +2, 0] + 4.0 * in_phi[0, +1, 0]
			)


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (nb, nb, 0)}, domain=(nx-2*nb, ny-2*nb, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:nb, :]       = phi.data[:nb, :]
		phi_out.data[-nb:, :]      = phi.data[-nb:, :]
		phi_out.data[nb:-nb, :nb]  = phi.data[nb:-nb,  :nb]
		phi_out.data[nb:-nb, -nb:] = phi.data[nb:-nb, -nb:]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.625 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.015625 * in_gamma[0, 0, 0] * (
				in_phi[-3, 0, 0] - 6.0*in_phi[-2, 0, 0] + 15.0*in_phi[-1, 0, 0] +
				in_phi[+3, 0, 0] - 6.0*in_phi[+2, 0, 0] + 15.0*in_phi[+1, 0, 0] +
				in_phi[0, -3, 0] - 6.0*in_phi[0, -2, 0] + 15.0*in_phi[0, -1, 0] +
				in_phi[0, +3, 0] - 6.0*in_phi[0, +2, 0] + 15.0*in_phi[0, +1, 0]
			)


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (nb, 0, 0)}, domain=(nx-2*nb, ny, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:nb, :]  = phi.data[:nb, :]
		phi_out.data[-nb:, :] = phi.data[-nb:, :]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.3125 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.015625 * in_gamma[0, 0, 0] * (
				in_phi[-3, 0, 0] - 6.0*in_phi[-2, 0, 0] + 15.0*in_phi[-1, 0, 0] +
				in_phi[+3, 0, 0] - 6.0*in_phi[+2, 0, 0] + 15.0*in_phi[+1, 0, 0]
			)


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
		self, shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth,	nb,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		nb = 3 if (nb is None or nb < 3) else nb
		super().__init__(
			shape, smooth_coeff, smooth_coeff_max, smooth_damp_depth, nb,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, phi, phi_out):
		# shortcuts
		nb = self._nb
		nx, ny, nz = self._shape

		# run the stencil
		self._stencil(
			in_phi=phi, in_gamma=self._gamma, out_phi=phi_out,
			origin={"_all_": (0, nb, 0)}, domain=(nx, ny-2*nb, nz),
			exec_info=self._exec_info
		)

		# set the outermost lateral layers of the output field,
		# not affected by the stencil
		phi_out.data[:, :nb]  = phi.data[:, :nb]
		phi_out.data[:, -nb:] = phi.data[:, -nb:]

	@staticmethod
	def _stencil_defs(
		in_phi: gt.storage.f64_ijk_sd,
		in_gamma: gt.storage.f64_k_sd,
		out_phi: gt.storage.f64_ijk_sd
	):
		out_phi = (1.0 - 0.3125 * in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + \
			0.015625 * in_gamma[0, 0, 0] * (
				in_phi[0, -3, 0] - 6.0*in_phi[0, -2, 0] + 15.0*in_phi[0, -1, 0] +
				in_phi[0, +3, 0] - 6.0*in_phi[0, +2, 0] + 15.0*in_phi[0, +1, 0]
			)

