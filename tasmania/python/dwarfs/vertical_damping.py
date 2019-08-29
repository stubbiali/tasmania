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
	VerticalDamping
	Rayleigh(VerticalDamping)
"""
import abc
import math
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.utils.storage_utils import get_storage_descriptor
from tasmania.python.utils.utils import greater_or_equal_than as ge

try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


class VerticalDamping(abc.ABC):
	"""
	Abstract base class whose derived classes implement different
	vertical damping, i.e., wave absorbing, techniques.
	"""
	def __init__(
		self, grid, shape, damp_depth, damp_coeff_max, time_units,
		backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
	):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		shape : tuple
			TODO
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_coeff_max : float
			Maximum value for the damping coefficient.
		time_units : str
			Time units to be used throughout the class.
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
		# safety-guard checks
		assert damp_depth <= grid.nz, \
			"The depth of the damping region ({}) should be smaller or equal than " \
			"the number of main vertical levels ({}).".format(damp_depth, grid.nz)

		# store input arguments needed at run-time
		self._shape = shape
		self._damp_depth = damp_depth
		self._tunits = time_units
		self._exec_info = exec_info

		# compute lower-bound of damping region
		lb = grid.z.values[damp_depth-1]

		# compute the damping matrix
		z = grid.z.values if shape[2] == grid.nz else grid.z_on_interface_levels.values
		za, zt = z[damp_depth-1], z[-1]
		r = ge(z, za) * damp_coeff_max * (1 - np.cos(math.pi * (z - za) / (zt - za)))
		rmat = r[np.newaxis, np.newaxis, :]

		# promote rmat to a gt4py storage
		descriptor = get_storage_descriptor(shape, dtype, halo, mask=(True, True, True))  # mask=(False, False, True)
		self._rmat = gt.storage.from_array(rmat, descriptor, backend=backend)

		# instantiate the underlying stencil
		decorator = gt.stencil(
			backend, backend_opts=backend_opts, build_info=build_info,
			rebuild=rebuild
		)
		self._stencil = decorator(self._stencil_defs)

	@abc.abstractmethod
	def __call__(
		self, dt, field_now, field_new, field_ref, field_out
	):
		"""
		Apply vertical damping to a generic field.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		dt : timedelta
			The time step.
		field_now : gridtools.storage.Storage
			The field at the current time level.
		field_new : gridtools.storage.Storage
			The field at the next time level, on which the absorber will be applied.
		field_ref : gridtools.storage.Storage
			A reference value for the field.
		field_out : gridtools.storage.Storage
			Buffer into which writing the output, vertically damped field.
		"""
		pass

	@staticmethod
	def factory(
		damp_type, grid, shape, damp_depth, damp_coeff_max,	time_units='s', *,
		backend="numpy", backend_opts=None, build_info=None, dtype=datatype,
		exec_info=None, halo=None, rebuild=False
	):
		"""
		Static method which returns an instance of the derived class
		implementing the damping method specified by :data:`damp_type`.

		Parameters
		----------
		damp_type : str
			String specifying the damper to implement. Either:

				* 'rayleigh', for a Rayleigh damper.

		shape : tuple
			Shape of the 3-D arrays on which applying the absorber.
		grid : tasmania.Grid
			The underlying grid.
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_coeff_max : float
			Maximum value for the damping coefficient.
		time_units : `str`, optional
			Time units to be used throughout the class. Defaults to 's'.
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
			An instance of the appropriate derived class.
		"""
		args = [
			grid, shape, damp_depth, damp_coeff_max, time_units,
			backend, backend_opts, build_info, dtype, exec_info, halo, rebuild
		]
		if damp_type == 'rayleigh':
			return Rayleigh(*args)
		else:
			raise ValueError('Unknown damping scheme. Available options: ''rayleigh''.')

	@staticmethod
	@abc.abstractmethod
	def _stencil_defs(
		in_phi_now: gt.storage.f64_ijk_sd,
		in_phi_new: gt.storage.f64_ijk_sd,
		in_phi_ref: gt.storage.f64_ijk_sd,
		in_rmat: gt.storage.f64_ijk_sd,
		out_phi: gt.storage.f64_ijk_sd,
		*,
		dt: float
	):
		pass


class Rayleigh(VerticalDamping):
	"""
	This class inherits	:class:`~tasmania.VerticalDamping`
	to implement a Rayleigh absorber.
	"""
	def __init__(
		self, grid, shape, damp_depth=15, damp_coeff_max=0.0002, time_units='s',
		backend='numpy', backend_opts=None, build_info=None, dtype=datatype,
		exec_info=None, halo=None, rebuild=False
	):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		shape : tuple
			Shape of the 3-D arrays on which applying the absorber.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_coeff_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		time_units : `str`, optional
			Time units to be used throughout the class. Defaults to 's'.
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
		"""
		super().__init__(
			grid, shape, damp_depth, damp_coeff_max, time_units, backend,
			backend_opts, build_info, dtype, exec_info, halo, rebuild
		)

	def __call__(self, dt, field_now, field_new, field_ref, field_out):
		# shortcuts
		ni, nj, nk = self._shape
		dnk = self._damp_depth

		# convert the timestep to seconds
		dt_da = DataArray(dt.total_seconds(), attrs={'units': 's'})
		dt_raw = dt_da.to_units(self._tunits).values.item()

		if dnk > 0:
			# run the stencil
			self._stencil(
				in_phi_now=field_now, in_phi_new=field_new, in_phi_ref=field_ref,
				in_rmat=self._rmat, out_phi=field_out, dt=dt_raw,
				origin={"_all_": (0, 0, 0)}, domain=(ni, nj, dnk),
				exec_info=self._exec_info
			)

		# set the lowermost layers, outside of the damping region
		field_out.data[:, :, dnk:] = field_new.data[:, :, dnk:]

	@staticmethod
	def _stencil_defs(
		in_phi_now: gt.storage.f64_ijk_sd,
		in_phi_new: gt.storage.f64_ijk_sd,
		in_phi_ref: gt.storage.f64_ijk_sd,
		in_rmat: gt.storage.f64_ijk_sd,
		out_phi: gt.storage.f64_ijk_sd,
		*,
		dt: float
	):
		out_phi = in_phi_new[0, 0, 0] - \
			dt * in_rmat[0, 0, 0] * (in_phi_now[0, 0, 0] - in_phi_ref[0, 0, 0])

