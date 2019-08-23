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
	IsentropicDiagnostics
"""
import numpy as np
from sympl import DataArray

import gridtools as gt
from gridtools.storage import StorageDescriptor
from tasmania.python.utils.data_utils import get_physical_constants

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class IsentropicDiagnostics:
	"""
	Class implementing the diagnostic steps of the three-dimensional
	isentropic dynamical core using GT4Py stencils.
	"""
	# Default values for the physical constants used in the class
	_d_physical_constants = {
		'air_pressure_at_sea_level':
			DataArray(1e5, attrs={'units': 'Pa'}),
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gravitational_acceleration':
			DataArray(9.80665, attrs={'units': 'm s^-2'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}

	def __init__(
		self, grid, physical_constants=None, *,
		backend="numpy", backend_opts=None, build_info=None, dtype=datatype,
		exec_info=None, halo=None, rebuild=False
	):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational acceleration', in units compatible with [m s^-2].
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.IsentropicDiagnostics._d_physical_constants`
			for the default values.
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
		# store the input arguments needed at run-time
		self._grid = grid
		self._backend = backend
		self._dtype = dtype
		self._exec_info = exec_info
		self._halo = halo

		# set the values of the physical constants
		pcs = get_physical_constants(self._d_physical_constants, physical_constants)
		self._pcs = pcs

		# assign to each grid point on the interface levels
		# the corresponding z-quota; this is required to diagnose
		# the geometrical height at the interface levels
		theta_1d = grid.z_on_interface_levels.to_units("K").values[np.newaxis, np.newaxis, :]
		theta = np.tile(theta_1d, (grid.nx+1, grid.ny+1, 1))

		# make theta a gt4py storage
		halo = (0, 0, 0) if halo is None else halo
		shape = (grid.nx+1, grid.ny+1, grid.nz+1)
		iteration_domain = tuple(shape[i] - 2*halo[i] for i in range(3))
		self._theta = gt.storage.from_array(
			theta,
			StorageDescriptor(dtype, iteration_domain=iteration_domain, halo=halo),
			backend=backend
		)

		# instantiate the underlying gt4py stencils
		decorator = gt.stencil(
			backend, backend_opts=backend_opts, build_info=build_info,
			rebuild=rebuild, externals={
				"pref": pcs["air_pressure_at_sea_level"],
				"rd": pcs["gas_constant_of_dry_air"],
				"g": pcs["gravitational_acceleration"],
				"cp": pcs["specific_heat_of_dry_air_at_constant_pressure"]
			}
		)
		self._stencil_diagnosing_air_pressure = decorator(
			self._stencil_diagnosing_air_pressure_defs
		)
		self._stencil_diagnosing_exner = decorator(
			self._stencil_diagnosing_exner_defs
		)
		self._stencil_diagnosing_montgomery = decorator(
			self._stencil_diagnosing_montgomery_defs
		)
		self._stencil_diagnosing_height = decorator(
			self._stencil_diagnosing_height_defs
		)
		self._stencil_diagnosing_air_density = decorator(
			self._stencil_diagnosing_air_density_defs
		)
		self._stencil_diagnosing_air_temperature = decorator(
			self._stencil_diagnosing_air_temperature_defs
		)

	def get_diagnostic_variables(self, s, pt, p, exn, mtg, h):
		"""
		With the help of the isentropic density and the upper boundary
		condition on the pressure distribution, diagnose the pressure,
		the Exner function, the Montgomery potential, and the geometric
		height of the half-levels.

		Parameters
		----------
		s : gridtools.storage.StorageDescriptor
			The isentropic density, in units of [kg m^-2 K^-1].
		pt : float
			The upper boundary condition on the pressure distribution,
			in units of [Pa].
		p : gridtools.storage.StorageDescriptor
			The buffer for the pressure at the interface levels, in units of [Pa].
		exn : gridtools.storage.StorageDescriptor
			The buffer for the Exner function at the interface levels,
			in units of [J K^-1 kg^-1].
		mtg : gridtools.storage.StorageDescriptor
			The buffer for the Montgomery potential, in units of [J kg^-1].
		h : gridtools.storage.StorageDescriptor
			The buffer for the geometric height of the interface levels, in units of [m].
		"""
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dz = self._grid.dz.to_units("K").values.item()
		g = self._pcs["gravitational_acceleration"]

		# apply the upper boundary condition on the pressure field
		p[:, :, 0] = pt

		# retrieve pressure at all other locations
		self._stencil_diagnosing_air_pressure(
			in_s=s, inout_p=p, dz=dz,
			origin={"_all_": (0, 0, 1)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

		# compute the Exner function
		self._stencil_diagnosing_exner(
			in_p=p, out_exn=exn,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz+1),
			exec_info=self._exec_info
		)

		# compute the Montgomery potential at the lower main level
		mtg_s = self._theta[:, :, -1] * exn[:, :, -1] \
			+ g * self._grid.topography.profile.to_units('m').values
		mtg[:, :, -2] = mtg_s + 0.5 * dz * exn[:, :, -1]

		# compute the Montgomery potential at all other locations
		self._stencil_diagnosing_montgomery(
			in_exn=exn, inout_mtg=mtg, dz=dz,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz-1),
			exec_info=self._exec_info
		)

		# compute the geometrical height of the isentropes
		h[:, :, -1] = self._grid.topography.profile.to_units('m').values[...]
		self._stencil_diagnosing_height(
			in_theta=self._theta, in_p=p, in_exn=exn, inout_h=h,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

	def get_montgomery_potential(self, s, pt, mtg):
		"""
		With the help of the isentropic density and the upper boundary
		condition on the pressure distribution, diagnose the Montgomery
		potential.

		Parameters
		----------
		s : gridtools.storage.StorageDescriptor
			The isentropic density, in units of [kg m^-2 K^-1].
		pt : float
			The upper boundary condition on the pressure distribution,
			in units of [Pa].
		mtg : gridtools.storage.StorageDescriptor
			The buffer for the Montgomery potential, in units of [J kg^-1].
		"""
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dz = self._grid.dz.to_units("K").values.item()
		g = self._pcs["gravitational_acceleration"]
		backend, dtype = self._backend, self._dtype
		halo = (0, 0, 0) if self._halo is None else self._halo
		iteration_domain = (
			nx + 1 - 2*halo[0], ny + 1 - 2*halo[1], nz + 1 - 2*halo[2]
		)

		if not hasattr(self, '_p'):
			# allocate temporary storage storing the pressure field
			self._p = gt.storage.zeros(
				StorageDescriptor(dtype, iteration_domain=iteration_domain, halo=halo),
				backend=backend
			)

		if not hasattr(self, '_exn'):
			# allocate temporary storage storing the Exner function
			self._exn = gt.storage.zeros(
				StorageDescriptor(dtype, iteration_domain=iteration_domain, halo=halo),
				backend=backend
			)

		# apply the upper boundary condition on the pressure field
		self._p[:, :, 0] = pt

		# retrieve pressure at all other locations
		self._stencil_diagnosing_air_pressure(
			in_s=s, inout_p=self._p, dz=dz,
			origin={"_all_": (0, 0, 1)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

		# compute the Exner function
		self._stencil_diagnosing_exner(
			in_p=self._p, out_exn=self._exn,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz+1),
			exec_info=self._exec_info
		)

		# compute the Montgomery potential at the lower main level
		mtg_s = self._theta[:, :, -1] * self._exn[:, :, -1] \
			+ g * self._grid.topography.profile.to_units('m').values
		mtg[:, :, -2] = mtg_s + 0.5 * dz * self._exn[:, :, -1]

		# compute the Montgomery potential at all other locations
		self._stencil_diagnosing_montgomery(
			in_exn=self._exn, inout_mtg=mtg, dz=dz,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz-1),
			exec_info=self._exec_info
		)

	def get_height(self, s, pt, h):
		"""
		With the help of the isentropic density and the upper boundary
		condition on the pressure distribution, diagnose the geometric
		height of the half-levels.

		Parameters
		----------
		s : gridtools.storage.StorageDescriptor
			The isentropic density, in units of [kg m^-2 K^-1].
		pt : float
			The upper boundary condition on the pressure distribution,
			in units of [Pa].
		h : gridtools.storage.StorageDescriptor
			The buffer for the geometric height of the interface levels, in units of [m].
		"""
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dz = self._grid.dz.to_units("K").values.item()
		g = self._pcs["gravitational_acceleration"]
		backend, dtype = self._backend, self._dtype
		halo = (0, 0, 0) if self._halo is None else self._halo
		iteration_domain = (
			nx + 1 - 2*halo[0], ny + 1 - 2*halo[1], nz + 1 - 2*halo[2]
		)

		if not hasattr(self, '_p'):
			# allocate temporary storage storing the pressure field
			self._p = gt.storage.zeros(
				StorageDescriptor(dtype, iteration_domain=iteration_domain, halo=halo),
				backend=backend
			)

		if not hasattr(self, '_exn'):
			# allocate temporary storage storing the Exner function
			self._exn = gt.storage.zeros(
				StorageDescriptor(dtype, iteration_domain=iteration_domain, halo=halo),
				backend=backend
			)

		# apply the upper boundary condition on the pressure field
		self._p[:, :, 0] = pt

		# retrieve pressure at all other locations
		self._stencil_diagnosing_air_pressure(
			in_s=s, inout_p=self._p, dz=dz,
			origin={"_all_": (0, 0, 1)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

		# compute the Exner function
		self._stencil_diagnosing_exner(
			in_p=self._p, out_exn=self._exn,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz+1),
			exec_info=self._exec_info
		)

		# compute the geometrical height of the isentropes
		h[:, :, -1] = self._grid.topography.profile.to_units('m').values[...]
		self._stencil_diagnosing_height(
			in_theta=self._theta, in_p=self._p, in_exn=self._exn, inout_h=h,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

	def get_air_density(self, s, h, rho):
		"""
		With the help of the isentropic density and the geometric height
		of the interface levels, diagnose the air density.

		Parameters
		----------
		s : gridtools.storage.StorageDescriptor
			The isentropic density, in units of [kg m^-2 K^-1].
		h : gridtools.storage.StorageDescriptor
			The geometric height of the interface levels, in units of [m].
		rho : gridtools.storage.StorageDescriptor
			The buffer for the air density, in units of [kg m^-3].
		"""
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# run the stencil
		self._stencil_diagnosing_air_density(
			in_theta=self._theta, in_s=s, in_h=h, out_rho=rho,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

	def get_air_temperature(self, exn, temp):
		"""
		With the help of the Exner function, diagnose the air temperature.

		Parameters
		----------
		exn : gridtools.storage.StorageDescriptor
			The Exner function at the interface levels, in units of [J K^-1 kg^-1].
		temp : gridtools.storage.StorageDescriptor
			The buffer for the temperature, in units of [K].
		"""
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# run the stencil
		self._stencil_diagnosing_air_temperature(
			in_theta=self._theta, in_exn=exn, out_temp=temp,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

	@staticmethod
	def _stencil_diagnosing_air_pressure_defs(
		in_s: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		inout_p: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		*,
		dz: float
	):
		with gt.region(iteration=gt.FORWARD, k_interval=None):
			inout_p = inout_p[-1] + g * dz * in_s[-1]

	@staticmethod
	def _stencil_diagnosing_exner_defs(
		in_p: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		out_exn: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True])
	):
		out_exn = cp * (in_p[0] / pref) ** (rd / cp)

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(
		in_exn: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		inout_mtg: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		*,
		dz: float
	):
		with gt.region(iteration=gt.BACKWARD, k_interval=None):
			inout_mtg = inout_mtg[1] + dz * in_exn[1]

	@staticmethod
	def _stencil_diagnosing_height_defs(
		in_theta: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		in_p: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		in_exn: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		inout_h: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True])
	):
		with gt.region(iteration=gt.BACKWARD, k_interval=None):
			inout_h = inout_h[1] - \
				rd * (in_theta[0] * in_exn[0] + in_theta[1] * in_exn[1]) * \
				(in_p[0] - in_p[1]) / (cp * g * (in_p[0] + in_p[1]))

	@staticmethod
	def _stencil_diagnosing_air_density_defs(
		in_theta: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		in_s: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		in_h: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		out_rho: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True])
	):
		out_rho = in_s[0] * (in_theta[0] - in_theta[1]) / (in_h[0] - in_h[1])

	@staticmethod
	def _stencil_diagnosing_air_temperature_defs(
		in_theta: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		in_exn: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True]),
		out_temp: StorageDescriptor(np.float64, grid_group="domain", mask=[False, False, True])
	):
		out_temp = .5 * (in_theta[0] * in_exn[0] + in_theta[1] * in_exn[1]) / cp
