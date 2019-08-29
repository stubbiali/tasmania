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

		# auxiliary fields
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

		# allocate a gt4py storage for the topography
		self._topo = gt.storage.zeros(
			StorageDescriptor(
				dtype, mask=[True, True, True],  # mask=[True, True, False]
				iteration_domain=iteration_domain, halo=halo
			),
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
		self._stencil_diagnostic_variables = decorator(
			self._stencil_diagnostic_variables_defs)
		self._stencil_density_and_temperature = decorator(
			self._stencil_density_and_temperature_defs)
		self._stencil_montgomery = decorator(self._stencil_montgomery_defs)
		self._stencil_height = decorator(self._stencil_height_defs)

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
		dz = self._grid.dz.to_units('K').values.item()

		# retrieve all the diagnostic variables
		self._topo.data[:-1, :-1, -1] = \
			self._grid.topography.profile.to_units('m').values[...]
		self._stencil_diagnostic_variables(
			in_theta=self._theta, in_hs=self._topo, in_s=s, inout_p=p,
			out_exn=exn, inout_mtg=mtg, inout_h=h, dz=dz, pt=pt,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz+1),
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
		dz = self._grid.dz.to_units('K').values.item()
		theta_s = self._grid.z_on_interface_levels.to_units('K').values[-1]

		# run the stencil
		self._topo.data[:-1, :-1, -1] = \
			self._grid.topography.profile.to_units('m').values[...]
		self._stencil_montgomery(
			in_hs=self._topo, in_s=s, inout_mtg=mtg, dz=dz, pt=pt, theta_s=theta_s,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz+1),
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
		dz = self._grid.dz.to_units('K').values.item()

		# run the stencil
		self._topo.data[:-1, :-1, -1] = \
			self._grid.topography.profile.to_units('m').values[...]
		self._stencil_height(
			in_theta=self._theta, in_hs=self._topo, in_s=s, inout_h=h, dz=dz, pt=pt,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz+1),
			exec_info=self._exec_info
		)

	def get_density_and_temperature(self, s, exn, h, rho, t):
		"""
		With the help of the isentropic density and the geometric height
		of the interface levels, diagnose the air density and temperature.

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
		self._stencil_density_and_temperature(
			in_theta=self._theta, in_s=s, in_exn=exn, in_h=h, out_rho=rho, out_t=t,
			origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz),
			exec_info=self._exec_info
		)

	@staticmethod
	def _stencil_diagnostic_variables_defs(
		in_theta: gt.storage.f64_sd,
		in_hs: gt.storage.f64_sd,
		in_s: gt.storage.f64_sd,
		inout_p: gt.storage.f64_sd,
		out_exn: gt.storage.f64_sd,
		inout_mtg: gt.storage.f64_sd,
		inout_h: gt.storage.f64_sd,
		*,
		dz: float,
		pt: float,
	):
		# retrieve the pressure
		with gt.region(iteration=gt.FORWARD, k_interval=(0, 1)):
			inout_p = pt
		with gt.region(iteration=gt.FORWARD, k_interval=(1, None)):
			inout_p = inout_p[0, 0, -1] + g * dz * in_s[0, 0, -1]

		# compute the Exner function
		with gt.region(iteration=gt.PARALLEL, k_interval=(0, None)):
			out_exn = cp * (inout_p[0, 0, 0] / pref) ** (rd / cp)

		# compute the Montgomery potential
		with gt.region(iteration=gt.BACKWARD, k_interval=(-2, -1)):
			mtg_s = in_theta[0, 0, 1] * out_exn[0, 0, 1] + g * in_hs[0, 0, 1]
			inout_mtg = mtg_s + 0.5 * dz * out_exn[0, 0, 1]
		with gt.region(iteration=gt.BACKWARD, k_interval=(0, -2)):
			inout_mtg = inout_mtg[0, 0, 1] + dz * out_exn[0, 0, 1]

		# compute the geometric height of the isentropes
		with gt.region(iteration=gt.BACKWARD, k_interval=(-1, None)):
			inout_h = in_hs[0, 0, 0]
		with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
			inout_h = inout_h[0, 0, 1] - rd * \
				(in_theta[0, 0, 0] * out_exn[0, 0, 0] +
					in_theta[0, 0, 1] * out_exn[0, 0, 1]) * \
				(inout_p[0, 0, 0] - inout_p[0, 0, 1]) / \
				(cp * g * (inout_p[0, 0, 0] + inout_p[0, 0, 1]))

	@staticmethod
	def _stencil_montgomery_defs(
		in_hs: gt.storage.f64_sd,
		in_s: gt.storage.f64_sd,
		inout_mtg: gt.storage.f64_sd,
		*,
		dz: float,
		pt: float,
		theta_s: float
	):
		# retrieve the pressure
		with gt.region(iteration=gt.FORWARD, k_interval=(0, 1)):
			inout_p = pt
		with gt.region(iteration=gt.FORWARD, k_interval=(1, None)):
			inout_p = inout_p[0, 0, -1] + g * dz * in_s[0, 0, -1]

		# compute the Exner function
		with gt.region(iteration=gt.PARALLEL, k_interval=(0, None)):
			out_exn = cp * (inout_p[0, 0, 0] / pref) ** (rd / cp)

		# compute the Montgomery potential
		with gt.region(iteration=gt.BACKWARD, k_interval=(-2, -1)):
			mtg_s = theta_s * out_exn[0, 0, 1] + g * in_hs[0, 0, 1]
			inout_mtg = mtg_s + 0.5 * dz * out_exn[0, 0, 1]
		with gt.region(iteration=gt.BACKWARD, k_interval=(0, -2)):
			inout_mtg = inout_mtg[0, 0, 1] + dz * out_exn[0, 0, 1]

	@staticmethod
	def _stencil_height_defs(
		in_theta: gt.storage.f64_sd,
		in_hs: gt.storage.f64_sd,
		in_s: gt.storage.f64_sd,
		inout_h: gt.storage.f64_sd,
		*,
		dz: float,
		pt: float,
	):
		# retrieve the pressure
		with gt.region(iteration=gt.FORWARD, k_interval=(0, 1)):
			inout_p = pt
		with gt.region(iteration=gt.FORWARD, k_interval=(1, None)):
			inout_p = inout_p[0, 0, -1] + g * dz * in_s[0, 0, -1]

		# compute the Exner function
		with gt.region(iteration=gt.PARALLEL, k_interval=(0, None)):
			out_exn = cp * (inout_p[0, 0, 0] / pref) ** (rd / cp)

		# compute the Montgomery potential
		with gt.region(iteration=gt.BACKWARD, k_interval=(-2, -1)):
			mtg_s = in_theta[0, 0, 1] * out_exn[0, 0, 1] + g * in_hs[0, 0, 1]
			inout_mtg = mtg_s + 0.5 * dz * out_exn[0, 0, 1]
		with gt.region(iteration=gt.BACKWARD, k_interval=(0, -2)):
			inout_mtg = inout_mtg[0, 0, 1] + dz * out_exn[0, 0, 1]

		# compute the geometric height of the isentropes
		with gt.region(iteration=gt.BACKWARD, k_interval=(-1, None)):
			inout_h = in_hs[0, 0, 0]
		with gt.region(iteration=gt.BACKWARD, k_interval=(0, -1)):
			inout_h = inout_h[0, 0, 1] - rd * \
				(in_theta[0, 0, 0] * out_exn[0, 0, 0] +
					in_theta[0, 0, 1] * out_exn[0, 0, 1]) * \
				(inout_p[0, 0, 0] - inout_p[0, 0, 1]) / \
				(cp * g * (inout_p[0, 0, 0] + inout_p[0, 0, 1]))

	@staticmethod
	def _stencil_density_and_temperature_defs(
		in_theta: gt.storage.f64_sd,
		in_s: gt.storage.f64_sd,
		in_exn: gt.storage.f64_sd,
		in_h: gt.storage.f64_sd,
		out_rho: gt.storage.f64_sd,
		out_t: gt.storage.f64_sd,
	):
		# compute the air density
		out_rho = in_s[0, 0, 0] * (in_theta[0, 0, 0] - in_theta[0, 0, 1]) / \
				  (in_h[0, 0, 0] - in_h[0, 0, 1])

		# compute the air temperature
		out_t = .5 / cp * (
			in_theta[0, 0, 0] * in_exn[0, 0, 0] +
			in_theta[0, 0, 1] * in_exn[0, 0, 1]
		)
