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
	IsentropicBoussinesqDiagnostics1
	IsentropicBoussinesqDiagnostics2
	IsentropicBoussinesqDiagnostics3
	IsentropicBoussinesqDiagnostics4
	IsentropicBoussinesqDiagnostics5
	IsentropicBoussinesqDiagnostics6
	IsentropicBoussinesqDiagnostics7
"""
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.utils.data_utils import get_physical_constants

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class IsentropicBoussinesqDiagnostics1(DiagnosticComponent):
	"""
	With the help of the isentropic density and the second derivative with
	respect to the potential temperature of the Montgomery potential, this
	class diagnoses

		* the height of the interface levels,
		* the Exner function,
		* the Montgomery potential and
		* the pressure

	under the Boussinesq assumption. Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# default values for the physical constants used in the class
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
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# store useful input arguments
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type)

		# allocate globals for the stencils
		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._pref     = gt.Global(pcs['air_pressure_at_sea_level'])
		self._rd 	   = gt.Global(pcs['gas_constant_of_dry_air'])
		self._g 	   = gt.Global(pcs['gravitational_acceleration'])
		self._cp       = gt.Global(pcs['specific_heat_of_dry_air_at_constant_pressure'])
		self._thetabar = gt.Global(self.grid.z_on_interface_levels.to_units('K').values[-1])
		self._dz       = gt.Global(self.grid.dz.to_units('K').values.item())

		# allocate input/output fields for the stencils
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._s     = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._ddmtg = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._h     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._exn   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._mtg   = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._p     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._diagnostics = {
			'air_pressure_on_interface_levels': self._p,
			'exner_function_on_interface_levels': self._exn,
			'height_on_interface_levels': self._h,
			'montgomery_potential': self._mtg
		}
		if moist:
			self._rho  = np.zeros((nx, ny, nz), dtype=dtype)
			self._temp = np.zeros((nx, ny, nz), dtype=dtype)
			self._diagnostics.update({
				'air_density': self._rho,
				'air_temperature': self._temp
			})

		# initialize the stencil diagnosing the pressure
		self._stencil_diagnosing_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_pressure_defs,
			inputs={'in_s': self._s, 'in_p': self._p},
			global_inputs={'g': self._g, 'dtheta': self._dz},
			outputs={'out_p': self._p},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

		# initialize the stencil diagnosing the Exner function
		self._stencil_diagnosing_exner = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_exner_defs,
			inputs={'in_p': self._p},
			global_inputs={'pref': self._pref, 'rd': self._rd, 'cp': self._cp},
			outputs={'out_exn': self._exn},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the height of the interface levels
		self._stencil_diagnosing_height = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_ddmtg': self._ddmtg, 'in_h': self._h},
			global_inputs={'g': self._g, 'thetabar': self._thetabar, 'dtheta': self._dz},
			outputs={'out_h': self._h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		# initialize the stencil diagnosing the Montgomery potential
		self._stencil_diagnosing_montgomery = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_h': self._h, 'in_mtg': self._mtg},
			global_inputs={'g': self._g, 'thetabar': self._thetabar, 'dtheta': self._dz},
			outputs={'out_mtg': self._mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		if moist:
			# initialize the stencil diagnosing the density and temperature
			self._stencil_diagnosing_density_temperature = gt.NGStencil(
				definitions_func=self._stencil_diagnosing_density_temperature_defs,
				inputs={'in_s': self._s, 'in_p': self._p, 'in_h': self._h},
				global_inputs={'rd': self._rd, 'dtheta': self._dz},
				outputs={'out_rho': self._rho, 'out_temp': self._temp},
				domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
				mode=backend
			)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'dd_montgomery_potential': {'dims': dims, 'units': 'm^2 K^-2 s^-2'}
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels': {'dims': dims_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_z, 'units': 'J kg^-1 K^-1'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'}
		}
		if self._moist:
			return_dict.update({
				'air_density': {'dims': dims, 'units': 'kg m^-3'},
				'air_temperature': {'dims': dims, 'units': 'K'}
			})

		return return_dict

	def array_call(self, state):
		# diagnose the pressure
		self._s[...] = state['air_isentropic_density'][...]
		self._p[:, :, 0] = self._pt
		self._stencil_diagnosing_pressure.compute()

		# diagnose the Exner function
		self._stencil_diagnosing_exner.compute()

		# diagnose the height of the interface levels
		self._ddmtg[...] = state['dd_montgomery_potential'][...]
		self._h[:, :, -1] = self.grid.topography.profile.to_units('m').values
		self._stencil_diagnosing_height.compute()

		# diagnose the Montgomery potential
		g = self._g.value
		thetabar = self._thetabar.value
		dtheta = self._dz.value
		mtg_s = g * self._h[:, :, -1] + thetabar * self._exn[:, :, -1]
		self._mtg[:, :, -1] = mtg_s - 0.5 * dtheta * (g / thetabar) * self._h[:, :, -1]
		self._stencil_diagnosing_montgomery.compute()

		if self._moist:
			# diagnose the density and temperature
			self._stencil_diagnosing_density_temperature.compute()

		return self._diagnostics

	@staticmethod
	def _stencil_diagnosing_pressure_defs(g, dtheta, in_s, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_p = gt.Equation()

		# computations
		out_p[k] = in_p[k-1] + g * dtheta * in_s[k-1]

		return out_p

	@staticmethod
	def _stencil_diagnosing_exner_defs(pref, rd, cp, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_exn = gt.Equation()

		# computations
		out_exn[k] = cp * (in_p[k] / pref) ** (rd / cp)

		return out_exn

	@staticmethod
	def _stencil_diagnosing_height_defs(g, thetabar, dtheta, in_ddmtg, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_h = gt.Equation()

		# computations
		out_h[k] = in_h[k+1] - dtheta * (thetabar / g) * in_ddmtg[k]

		return out_h

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(g, thetabar, dtheta, in_h, in_mtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_mtg = gt.Equation()

		# computations
		out_mtg[k] = in_mtg[k+1] - dtheta * (g / thetabar) * in_h[k+1]

		return out_mtg

	@staticmethod
	def _stencil_diagnosing_density_temperature_defs(rd, dtheta, in_s, in_p, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_rho = gt.Equation()
		out_temp = gt.Equation()

		# computations
		out_rho[k] = in_s[k] * dtheta / (in_h[k] - in_h[k+1])
		out_temp[k] = 0.5 * (in_p[k] + in_p[k+1]) / (rd * out_rho[k])

		return out_rho, out_temp


class IsentropicBoussinesqDiagnostics2(DiagnosticComponent):
	"""
	With the help of the isentropic density, this class diagnoses

		* the height of the interface levels,
		* the Exner function,
		* the Montgomery potential and
		* the pressure

	under the Boussinesq assumption. Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# default values for the physical constants used in the class
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
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# store useful input arguments
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type)

		# allocate globals for the stencils
		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._pref     = gt.Global(pcs['air_pressure_at_sea_level'])
		self._rd 	   = gt.Global(pcs['gas_constant_of_dry_air'])
		self._g 	   = gt.Global(pcs['gravitational_acceleration'])
		self._cp       = gt.Global(pcs['specific_heat_of_dry_air_at_constant_pressure'])
		self._thetabar = gt.Global(self.grid.z_on_interface_levels.to_units('K').values[-1])
		self._dz       = gt.Global(self.grid.dz.to_units('K').values.item())
		self._rhoref   = gt.Global(1.0)

		# allocate input/output fields for the stencils
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._s     = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._exn   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._p     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._mtg   = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._h     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._diagnostics = {
			'air_pressure_on_interface_levels': self._p,
			'exner_function_on_interface_levels': self._exn,
			'height_on_interface_levels': self._h,
			'montgomery_potential': self._mtg
		}
		if moist:
			self._rho  = np.zeros((nx, ny, nz), dtype=dtype)
			self._temp = np.zeros((nx, ny, nz), dtype=dtype)
			self._diagnostics.update({
				'air_density': self._rho,
				'air_temperature': self._temp
			})

		# initialize the stencil diagnosing the Exner function
		self._stencil_diagnosing_exner = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_exner_defs,
			inputs={'in_s': self._s, 'in_exn': self._exn},
			global_inputs={
				'g': self._g, 'thetabar': self._thetabar,
				'dtheta': self._dz, 'rhoref': self._rhoref
			},
			outputs={'out_exn': self._exn},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

		# initialize the stencil diagnosing the pressure
		self._stencil_diagnosing_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_pressure_defs,
			inputs={'in_exn': self._exn},
			global_inputs={'pref': self._pref, 'rd': self._rd, 'cp': self._cp},
			outputs={'out_p': self._p},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the Montgomery potential
		self._stencil_diagnosing_montgomery = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_exn': self._exn, 'in_mtg': self._mtg},
			global_inputs={'dtheta': self._dz},
			outputs={'out_mtg': self._mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		# initialize the stencil diagnosing the height of the interface levels
		self._stencil_diagnosing_height = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_exn': self._exn},
			global_inputs={'g': self._g, 'cp': self._cp, 'thetabar': self._thetabar},
			outputs={'out_h': self._h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend
		)

		if moist:
			# initialize the stencil diagnosing the density and temperature
			self._stencil_diagnosing_density_temperature = gt.NGStencil(
				definitions_func=self._stencil_diagnosing_density_temperature_defs,
				inputs={'in_s': self._s, 'in_p': self._p, 'in_h': self._h},
				global_inputs={'rd': self._rd, 'dtheta': self._dz},
				outputs={'out_rho': self._rho, 'out_temp': self._temp},
				domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
				mode=backend
			)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels': {'dims': dims_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_z, 'units': 'J kg^-1 K^-1'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'}
		}
		if self._moist:
			return_dict.update({
				'air_density': {'dims': dims, 'units': 'kg m^-3'},
				'air_temperature': {'dims': dims, 'units': 'K'}
			})

		return return_dict

	def array_call(self, state):
		# diagnose the Exner function
		self._s[...] = state['air_isentropic_density'][...]
		cp, pref, rd = self._cp.value, self._pref.value, self._rd.value
		self._exn[:, :, 0] = cp * (self._pt / pref) ** (rd / cp)
		self._stencil_diagnosing_exner.compute()

		# diagnose the pressure
		self._stencil_diagnosing_pressure.compute()

		# diagnose the Montgomery potential
		g = self._g.value
		thetabar = self._thetabar.value
		dtheta = self._dz.value
		hs = self.grid.topography.profile.to_units('m').values
		mtg_s = g * hs + thetabar * self._exn[:, :, -1]
		self._mtg[:, :, -1] = mtg_s + 0.5 * dtheta * self._exn[:, :, -1]
		self._stencil_diagnosing_montgomery.compute()

		# diagnose the height of the interface levels
		self._h[:, :, -1] = hs
		self._stencil_diagnosing_height.compute()

		if self._moist:
			# diagnose the density and temperature
			self._stencil_diagnosing_density_temperature.compute()

		return self._diagnostics

	@staticmethod
	def _stencil_diagnosing_exner_defs(g, rhoref, thetabar, dtheta, in_s, in_exn):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_exn = gt.Equation()

		# computations
		out_exn[k] = in_exn[k-1] + dtheta * g / (thetabar * rhoref) * in_s[k-1]

		return out_exn

	@staticmethod
	def _stencil_diagnosing_pressure_defs(cp, pref, rd, in_exn):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_p = gt.Equation()

		# computations
		out_p[k] = pref * (in_exn[k] / cp) ** (cp / rd)

		return out_p

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(dtheta, in_exn, in_mtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_mtg = gt.Equation()

		# computations
		out_mtg[k] = in_mtg[k+1] + dtheta * in_exn[k+1]

		return out_mtg

	@staticmethod
	def _stencil_diagnosing_height_defs(g, cp, thetabar, in_exn):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_h = gt.Equation()

		# computations
		out_h[k] = thetabar * (cp - in_exn[k]) / g

		return out_h

	@staticmethod
	def _stencil_diagnosing_density_temperature_defs(rd, dtheta, in_s, in_p, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_rho = gt.Equation()
		out_temp = gt.Equation()

		# computations
		out_rho[k] = in_s[k] * dtheta / (in_h[k] - in_h[k+1])
		out_temp[k] = 0.5 * (in_p[k] + in_p[k+1]) / (rd * out_rho[k])

		return out_rho, out_temp


class IsentropicBoussinesqDiagnostics3(DiagnosticComponent):
	"""
	With the help of the isentropic density, this class diagnoses

		* the height of the interface levels,
		* the Exner function,
		* the Montgomery potential and
		* the pressure

	under the Boussinesq assumption. Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# default values for the physical constants used in the class
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
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# store useful input arguments
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type)

		# allocate globals for the stencils
		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._pref     = gt.Global(pcs['air_pressure_at_sea_level'])
		self._rd 	   = gt.Global(pcs['gas_constant_of_dry_air'])
		self._g 	   = gt.Global(pcs['gravitational_acceleration'])
		self._cp       = gt.Global(pcs['specific_heat_of_dry_air_at_constant_pressure'])
		self._thetabar = gt.Global(self.grid.z_on_interface_levels.to_units('K').values[-1])
		self._dz       = gt.Global(self.grid.dz.to_units('K').values.item())
		self._rhoref   = gt.Global(1.0)

		# allocate input/output fields for the stencils
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._s     = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._p     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._exn   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._h     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._mtg   = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._diagnostics = {
			'air_pressure_on_interface_levels': self._p,
			'exner_function_on_interface_levels': self._exn,
			'height_on_interface_levels': self._h,
			'montgomery_potential': self._mtg
		}
		if moist:
			self._rho  = np.zeros((nx, ny, nz), dtype=dtype)
			self._temp = np.zeros((nx, ny, nz), dtype=dtype)
			self._diagnostics.update({
				'air_density': self._rho,
				'air_temperature': self._temp
			})

		# initialize the stencil diagnosing the pressure
		self._stencil_diagnosing_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_pressure_defs,
			inputs={'in_s': self._s, 'in_p': self._p},
			global_inputs={'g': self._g, 'dtheta': self._dz},
			outputs={'out_p': self._p},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

		# initialize the stencil diagnosing the Exner function
		self._stencil_diagnosing_exner = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_exner_defs,
			inputs={'in_p': self._p},
			global_inputs={
				'cp': self._cp, 'pref': self._pref, 'rd': self._rd
			},
			outputs={'out_exn': self._exn},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the height of the interface levels
		self._stencil_diagnosing_height = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_s': self._s, 'in_h': self._h},
			global_inputs={'rhoref': self._rhoref, 'dtheta': self._dz},
			outputs={'out_h': self._h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		# initialize the stencil diagnosing the Montgomery potential
		self._stencil_diagnosing_montgomery = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_h': self._h, 'in_mtg': self._mtg},
			global_inputs={'g': self._g, 'thetabar': self._thetabar, 'dtheta': self._dz},
			outputs={'out_mtg': self._mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		if moist:
			# initialize the stencil diagnosing the density and temperature
			self._stencil_diagnosing_density_temperature = gt.NGStencil(
				definitions_func=self._stencil_diagnosing_density_temperature_defs,
				inputs={'in_s': self._s, 'in_p': self._p, 'in_h': self._h},
				global_inputs={'rd': self._rd, 'dtheta': self._dz},
				outputs={'out_rho': self._rho, 'out_temp': self._temp},
				domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
				mode=backend
			)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels': {'dims': dims_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_z, 'units': 'J kg^-1 K^-1'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'}
		}
		if self._moist:
			return_dict.update({
				'air_density': {'dims': dims, 'units': 'kg m^-3'},
				'air_temperature': {'dims': dims, 'units': 'K'}
			})

		return return_dict

	def array_call(self, state):
		# diagnose the pressure
		self._s[...] = state['air_isentropic_density'][...]
		self._p[:, :, 0] = self._pt
		self._stencil_diagnosing_pressure.compute()

		# diagnose the Exner function
		self._stencil_diagnosing_exner.compute()

		# diagnose the height of the interface levels
		self._h[:, :, -1] = self.grid.topography.profile.to_units('m').values
		self._stencil_diagnosing_height.compute()

		# diagnose the Montgomery potential
		g = self._g.value
		thetabar = self._thetabar.value
		dtheta = self._dz.value
		mtg_s = g * self._h[:, :, -1] + thetabar * self._exn[:, :, -1]
		self._mtg[:, :, -1] = mtg_s - 0.5 * dtheta * (g / thetabar) * self._h[:, :, -1]
		self._stencil_diagnosing_montgomery.compute()

		if self._moist:
			# diagnose the density and temperature
			self._stencil_diagnosing_density_temperature.compute()

		return self._diagnostics

	@staticmethod
	def _stencil_diagnosing_pressure_defs(g, dtheta, in_s, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_p = gt.Equation()

		# computations
		out_p[k] = in_p[k-1] + dtheta * g * in_s[k-1]

		return out_p

	@staticmethod
	def _stencil_diagnosing_exner_defs(cp, pref, rd, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_exn = gt.Equation()

		# computations
		out_exn[k] = cp * (in_p[k] / pref) ** (rd / cp)

		return out_exn

	@staticmethod
	def _stencil_diagnosing_height_defs(rhoref, dtheta, in_s, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_h = gt.Equation()

		# computations
		out_h[k] = in_h[k+1] + dtheta * in_s[k] / rhoref

		return out_h

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(g, dtheta, thetabar, in_h, in_mtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_mtg = gt.Equation()

		# computations
		out_mtg[k] = in_mtg[k+1] - dtheta * (g / thetabar) * in_h[k+1]

		return out_mtg

	@staticmethod
	def _stencil_diagnosing_density_temperature_defs(rd, dtheta, in_s, in_p, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_rho = gt.Equation()
		out_temp = gt.Equation()

		# computations
		out_rho[k] = in_s[k] * dtheta / (in_h[k] - in_h[k+1])
		out_temp[k] = 0.5 * (in_p[k] + in_p[k+1]) / (rd * out_rho[k])

		return out_rho, out_temp


class IsentropicBoussinesqDiagnostics4(DiagnosticComponent):
	"""
	With the help of the isentropic density, this class diagnoses

		* the height of the interface levels,
		* the Exner function,
		* the Montgomery potential and
		* the pressure

	under the Boussinesq assumption. Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# default values for the physical constants used in the class
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
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# store useful input arguments
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type)

		# allocate globals for the stencils
		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._pref     = gt.Global(pcs['air_pressure_at_sea_level'])
		self._rd 	   = gt.Global(pcs['gas_constant_of_dry_air'])
		self._g 	   = gt.Global(pcs['gravitational_acceleration'])
		self._cp       = gt.Global(pcs['specific_heat_of_dry_air_at_constant_pressure'])
		self._thetabar = gt.Global(self.grid.z_on_interface_levels.to_units('K').values[-1])
		self._dz       = gt.Global(self.grid.dz.to_units('K').values.item())
		self._rhoref   = gt.Global(1.0)

		# allocate input/output fields for the stencils
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._s     = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._p     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._exn   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._h     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._mtg   = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._diagnostics = {
			'air_pressure_on_interface_levels': self._p,
			'exner_function_on_interface_levels': self._exn,
			'height_on_interface_levels': self._h,
			'montgomery_potential': self._mtg
		}
		if moist:
			self._rho  = np.zeros((nx, ny, nz), dtype=dtype)
			self._temp = np.zeros((nx, ny, nz), dtype=dtype)
			self._diagnostics.update({
				'air_density': self._rho,
				'air_temperature': self._temp
			})

		# initialize the stencil diagnosing the pressure
		self._stencil_diagnosing_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_pressure_defs,
			inputs={'in_s': self._s, 'in_p': self._p},
			global_inputs={'g': self._g, 'dtheta': self._dz},
			outputs={'out_p': self._p},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

		# initialize the stencil diagnosing the Exner function
		self._stencil_diagnosing_exner = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_exner_defs,
			inputs={'in_p': self._p},
			global_inputs={
				'cp': self._cp, 'pref': self._pref, 'rd': self._rd
			},
			outputs={'out_exn': self._exn},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the height of the interface levels
		self._stencil_diagnosing_height = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_exn': self._exn},
			global_inputs={'g': self._g, 'cp': self._cp, 'thetabar': self._thetabar},
			outputs={'out_h': self._h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend
		)

		# initialize the stencil diagnosing the Montgomery potential
		self._stencil_diagnosing_montgomery = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_h': self._h, 'in_mtg': self._mtg},
			global_inputs={
				'g': self._g, 'cp': self._cp, 'dtheta': self._dz,
				'thetabar': self._thetabar
			},
			outputs={'out_mtg': self._mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		if moist:
			# initialize the stencil diagnosing the density and temperature
			self._stencil_diagnosing_density_temperature = gt.NGStencil(
				definitions_func=self._stencil_diagnosing_density_temperature_defs,
				inputs={'in_s': self._s, 'in_p': self._p, 'in_h': self._h},
				global_inputs={'rd': self._rd, 'dtheta': self._dz},
				outputs={'out_rho': self._rho, 'out_temp': self._temp},
				domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
				mode=backend
			)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels': {'dims': dims_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_z, 'units': 'J kg^-1 K^-1'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'}
		}
		if self._moist:
			return_dict.update({
				'air_density': {'dims': dims, 'units': 'kg m^-3'},
				'air_temperature': {'dims': dims, 'units': 'K'}
			})

		return return_dict

	def array_call(self, state):
		# diagnose the pressure
		self._s[...] = state['air_isentropic_density'][...]
		self._p[:, :, 0] = self._pt
		self._stencil_diagnosing_pressure.compute()

		# diagnose the Exner function
		self._stencil_diagnosing_exner.compute()

		# diagnose the height of the interface levels
		self._h[:, :, -1] = self.grid.topography.profile.to_units('m').values
		self._stencil_diagnosing_height.compute()

		# diagnose the Montgomery potential
		g = self._g.value
		cp = self._cp.value
		thetabar = self._thetabar.value
		dtheta = self._dz.value
		mtg_s = g * self._h[:, :, -1] + thetabar * self._exn[:, :, -1]
		self._mtg[:, :, -1] = mtg_s + \
			0.5 * dtheta * (thetabar * cp - g * self._h[:, :, -1]) / thetabar
		self._stencil_diagnosing_montgomery.compute()

		if self._moist:
			# diagnose the density and temperature
			self._stencil_diagnosing_density_temperature.compute()

		return self._diagnostics

	@staticmethod
	def _stencil_diagnosing_pressure_defs(g, dtheta, in_s, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_p = gt.Equation()

		# computations
		out_p[k] = in_p[k-1] + dtheta * g * in_s[k-1]

		return out_p

	@staticmethod
	def _stencil_diagnosing_exner_defs(cp, pref, rd, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_exn = gt.Equation()

		# computations
		out_exn[k] = cp * (in_p[k] / pref) ** (rd / cp)

		return out_exn

	@staticmethod
	def _stencil_diagnosing_height_defs(g, cp, thetabar, in_exn):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_h = gt.Equation()

		# computations
		out_h[k] = thetabar * (cp - in_exn[k]) / g

		return out_h

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(g, cp, dtheta, thetabar, in_h, in_mtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_mtg = gt.Equation()

		# computations
		out_mtg[k] = in_mtg[k+1] + dtheta * (cp * thetabar - g * in_h[k+1]) / thetabar

		return out_mtg

	@staticmethod
	def _stencil_diagnosing_density_temperature_defs(rd, dtheta, in_s, in_p, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_rho = gt.Equation()
		out_temp = gt.Equation()

		# computations
		out_rho[k] = in_s[k] * dtheta / (in_h[k] - in_h[k+1])
		out_temp[k] = 0.5 * (in_p[k] + in_p[k+1]) / (rd * out_rho[k])

		return out_rho, out_temp


class IsentropicBoussinesqDiagnostics5(DiagnosticComponent):
	"""
	With the help of the isentropic density, this class diagnoses

		* the height of the interface levels,
		* the Exner function,
		* the Montgomery potential and
		* the pressure

	under the Boussinesq assumption. Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# default values for the physical constants used in the class
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
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# store useful input arguments
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type)

		# allocate globals for the stencils
		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._pref     = gt.Global(pcs['air_pressure_at_sea_level'])
		self._rd 	   = gt.Global(pcs['gas_constant_of_dry_air'])
		self._g 	   = gt.Global(pcs['gravitational_acceleration'])
		self._cp       = gt.Global(pcs['specific_heat_of_dry_air_at_constant_pressure'])
		self._thetabar = gt.Global(self.grid.z_on_interface_levels.to_units('K').values[-1])
		self._dz       = gt.Global(self.grid.dz.to_units('K').values.item())
		self._rhoref   = gt.Global(1.0)

		# allocate input/output fields for the stencils
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._s     = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._p     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._exn   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._h     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._mtg   = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._diagnostics = {
			'air_pressure_on_interface_levels': self._p,
			'exner_function_on_interface_levels': self._exn,
			'height_on_interface_levels': self._h,
			'montgomery_potential': self._mtg
		}
		if moist:
			self._rho  = np.zeros((nx, ny, nz), dtype=dtype)
			self._temp = np.zeros((nx, ny, nz), dtype=dtype)
			self._diagnostics.update({
				'air_density': self._rho,
				'air_temperature': self._temp
			})

		# initialize the stencil diagnosing the pressure
		self._stencil_diagnosing_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_pressure_defs,
			inputs={'in_s': self._s, 'in_p': self._p},
			global_inputs={'g': self._g, 'dtheta': self._dz},
			outputs={'out_p': self._p},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

		# initialize the stencil diagnosing the Exner function
		self._stencil_diagnosing_exner = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_exner_defs,
			inputs={'in_p': self._p},
			global_inputs={
				'cp': self._cp, 'pref': self._pref, 'rd': self._rd
			},
			outputs={'out_exn': self._exn},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the height of the interface levels
		self._stencil_diagnosing_height = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_exn': self._exn},
			global_inputs={'g': self._g, 'cp': self._cp, 'thetabar': self._thetabar},
			outputs={'out_h': self._h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend
		)

		# initialize the stencil diagnosing the Montgomery potential
		self._stencil_diagnosing_montgomery = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_h': self._h, 'in_mtg': self._mtg},
			global_inputs={'g': self._g, 'thetabar': self._thetabar, 'dtheta': self._dz},
			outputs={'out_mtg': self._mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		if moist:
			# initialize the stencil diagnosing the density and temperature
			self._stencil_diagnosing_density_temperature = gt.NGStencil(
				definitions_func=self._stencil_diagnosing_density_temperature_defs,
				inputs={'in_s': self._s, 'in_p': self._p, 'in_h': self._h},
				global_inputs={'rd': self._rd, 'dtheta': self._dz},
				outputs={'out_rho': self._rho, 'out_temp': self._temp},
				domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
				mode=backend
			)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels': {'dims': dims_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_z, 'units': 'J kg^-1 K^-1'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'}
		}
		if self._moist:
			return_dict.update({
				'air_density': {'dims': dims, 'units': 'kg m^-3'},
				'air_temperature': {'dims': dims, 'units': 'K'}
			})

		return return_dict

	def array_call(self, state):
		# diagnose the pressure
		self._s[...] = state['air_isentropic_density'][...]
		self._p[:, :, 0] = self._pt
		self._stencil_diagnosing_pressure.compute()

		# diagnose the Exner function
		self._stencil_diagnosing_exner.compute()

		# diagnose the height of the interface levels
		self._h[:, :, -1] = self.grid.topography.profile.to_units('m').values
		self._stencil_diagnosing_height.compute()

		# diagnose the Montgomery potential
		g = self._g.value
		thetabar = self._thetabar.value
		dtheta = self._dz.value
		mtg_s = g * self._h[:, :, -1] + thetabar * self._exn[:, :, -1]
		self._mtg[:, :, -1] = mtg_s - 0.5 * dtheta * (g / thetabar) * self._h[:, :, -1]
		self._stencil_diagnosing_montgomery.compute()

		if self._moist:
			# diagnose the density and temperature
			self._stencil_diagnosing_density_temperature.compute()

		return self._diagnostics

	@staticmethod
	def _stencil_diagnosing_pressure_defs(g, dtheta, in_s, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_p = gt.Equation()

		# computations
		out_p[k] = in_p[k-1] + dtheta * g * in_s[k-1]

		return out_p

	@staticmethod
	def _stencil_diagnosing_exner_defs(cp, pref, rd, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_exn = gt.Equation()

		# computations
		out_exn[k] = cp * (in_p[k] / pref) ** (rd / cp)

		return out_exn

	@staticmethod
	def _stencil_diagnosing_height_defs(g, cp, thetabar, in_exn):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_h = gt.Equation()

		# computations
		out_h[k] = thetabar * (cp - in_exn[k]) / g

		return out_h

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(g, dtheta, thetabar, in_h, in_mtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_mtg = gt.Equation()

		# computations
		out_mtg[k] = in_mtg[k+1] - dtheta * (g / thetabar) * in_h[k+1]

		return out_mtg

	@staticmethod
	def _stencil_diagnosing_density_temperature_defs(rd, dtheta, in_s, in_p, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_rho = gt.Equation()
		out_temp = gt.Equation()

		# computations
		out_rho[k] = in_s[k] * dtheta / (in_h[k] - in_h[k+1])
		out_temp[k] = 0.5 * (in_p[k] + in_p[k+1]) / (rd * out_rho[k])

		return out_rho, out_temp


class IsentropicBoussinesqDiagnostics6(DiagnosticComponent):
	"""
	With the help of the isentropic density, this class diagnoses

		* the height of the interface levels,
		* the Exner function,
		* the Montgomery potential and
		* the pressure

	under the Boussinesq assumption. Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# default values for the physical constants used in the class
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
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# store useful input arguments
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type)

		# allocate globals for the stencils
		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._pref     = gt.Global(pcs['air_pressure_at_sea_level'])
		self._rd 	   = gt.Global(pcs['gas_constant_of_dry_air'])
		self._g 	   = gt.Global(pcs['gravitational_acceleration'])
		self._cp       = gt.Global(pcs['specific_heat_of_dry_air_at_constant_pressure'])
		self._thetabar = gt.Global(self.grid.z_on_interface_levels.to_units('K').values[-1])
		self._dz       = gt.Global(self.grid.dz.to_units('K').values.item())
		self._rhoref   = gt.Global(1.0)

		# allocate input/output fields for the stencils
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._s     = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._p     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._exn   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._dmtg  = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._h     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._mtg   = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._diagnostics = {
			'air_pressure_on_interface_levels': self._p,
			'exner_function_on_interface_levels': self._exn,
			'height_on_interface_levels': self._h,
			'montgomery_potential': self._mtg
		}
		if moist:
			self._rho  = np.zeros((nx, ny, nz), dtype=dtype)
			self._temp = np.zeros((nx, ny, nz), dtype=dtype)
			self._diagnostics.update({
				'air_density': self._rho,
				'air_temperature': self._temp
			})

		# initialize the stencil diagnosing the pressure
		self._stencil_diagnosing_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_pressure_defs,
			inputs={'in_s': self._s, 'in_p': self._p},
			global_inputs={'g': self._g, 'dtheta': self._dz},
			outputs={'out_p': self._p},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

		# initialize the stencil diagnosing the Exner function
		self._stencil_diagnosing_exner = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_exner_defs,
			inputs={'in_p': self._p},
			global_inputs={
				'cp': self._cp, 'pref': self._pref, 'rd': self._rd
			},
			outputs={'out_exn': self._exn},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the first derivative with respect
		# to the potential temperature of the Montgomery potential
		self._stencil_diagnosing_dmtg = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_dmtg_defs,
			inputs={'in_s': self._s, 'in_dmtg': self._dmtg},
			global_inputs={
				'g': self._g, 'rhoref': self._rhoref,
				'thetabar': self._thetabar, 'dtheta': self._dz
			},
			outputs={'out_dmtg': self._dmtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		# initialize the stencil diagnosing the height of the interface levels
		self._stencil_diagnosing_height = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_dmtg': self._dmtg},
			global_inputs={'g': self._g, 'thetabar': self._thetabar},
			outputs={'out_h': self._h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the Montgomery potential
		self._stencil_diagnosing_montgomery = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_dmtg': self._dmtg, 'in_mtg': self._mtg},
			global_inputs={'dtheta': self._dz},
			outputs={'out_mtg': self._mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		if moist:
			# initialize the stencil diagnosing the density and temperature
			self._stencil_diagnosing_density_temperature = gt.NGStencil(
				definitions_func=self._stencil_diagnosing_density_temperature_defs,
				inputs={'in_s': self._s, 'in_p': self._p, 'in_h': self._h},
				global_inputs={'rd': self._rd, 'dtheta': self._dz},
				outputs={'out_rho': self._rho, 'out_temp': self._temp},
				domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
				mode=backend
			)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels': {'dims': dims_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_z, 'units': 'J kg^-1 K^-1'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'}
		}
		if self._moist:
			return_dict.update({
				'air_density': {'dims': dims, 'units': 'kg m^-3'},
				'air_temperature': {'dims': dims, 'units': 'K'}
			})

		return return_dict

	def array_call(self, state):
		# diagnose the pressure
		self._s[...] = state['air_isentropic_density'][...]
		self._p[:, :, 0] = self._pt
		self._stencil_diagnosing_pressure.compute()

		# diagnose the Exner function
		self._stencil_diagnosing_exner.compute()

		# diagnose the first derivative with respect to the
		# potential temperature of the Montgomery potential
		g = self._g.value
		thetabar = self._thetabar.value
		hs = self.grid.topography.profile.to_units('m').values
		self._dmtg[:, :, -1] = - (g / thetabar) * hs
		self._stencil_diagnosing_dmtg.compute()

		# diagnose the height of the interface levels
		self._stencil_diagnosing_height.compute()

		# diagnose the Montgomery potential
		dtheta = self._dz.value
		mtg_s = g * hs + thetabar * self._exn[:, :, -1]
		self._mtg[:, :, -1] = mtg_s - 0.5 * dtheta * (g / thetabar) * hs
		self._stencil_diagnosing_montgomery.compute()

		if self._moist:
			# diagnose the density and temperature
			self._stencil_diagnosing_density_temperature.compute()

		return self._diagnostics

	@staticmethod
	def _stencil_diagnosing_pressure_defs(g, dtheta, in_s, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_p = gt.Equation()

		# computations
		out_p[k] = in_p[k-1] + dtheta * g * in_s[k-1]

		return out_p

	@staticmethod
	def _stencil_diagnosing_exner_defs(cp, pref, rd, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_exn = gt.Equation()

		# computations
		out_exn[k] = cp * (in_p[k] / pref) ** (rd / cp)

		return out_exn

	@staticmethod
	def _stencil_diagnosing_dmtg_defs(g, rhoref, thetabar, dtheta, in_s, in_dmtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_dmtg = gt.Equation()

		# computations
		out_dmtg[k] = in_dmtg[k+1] - dtheta * g / (rhoref * thetabar) * in_s[k]

		return out_dmtg

	@staticmethod
	def _stencil_diagnosing_height_defs(g, thetabar, in_dmtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_h = gt.Equation()

		# computations
		out_h[k] = - (thetabar / g) * in_dmtg[k]

		return out_h

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(dtheta, in_dmtg, in_mtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_mtg = gt.Equation()

		# computations
		out_mtg[k] = in_mtg[k+1] + dtheta * in_dmtg[k+1]

		return out_mtg

	@staticmethod
	def _stencil_diagnosing_density_temperature_defs(rd, dtheta, in_s, in_p, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_rho = gt.Equation()
		out_temp = gt.Equation()

		# computations
		out_rho[k] = in_s[k] * dtheta / (in_h[k] - in_h[k+1])
		out_temp[k] = 0.5 * (in_p[k] + in_p[k+1]) / (rd * out_rho[k])

		return out_rho, out_temp


class IsentropicBoussinesqDiagnostics7(DiagnosticComponent):
	"""
	With the help of the isentropic density and the second derivative
	with respect to the potential temperature of the Montgomery potential,
	this class diagnoses

		* the height of the interface levels,
		* the Exner function,
		* the Montgomery potential and
		* the pressure

	under the Boussinesq assumption. Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# default values for the physical constants used in the class
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
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# store useful input arguments
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type)

		# allocate globals for the stencils
		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._pref     = gt.Global(pcs['air_pressure_at_sea_level'])
		self._rd 	   = gt.Global(pcs['gas_constant_of_dry_air'])
		self._g 	   = gt.Global(pcs['gravitational_acceleration'])
		self._cp       = gt.Global(pcs['specific_heat_of_dry_air_at_constant_pressure'])
		self._thetabar = gt.Global(self.grid.z_on_interface_levels.to_units('K').values[-1])
		self._dz       = gt.Global(self.grid.dz.to_units('K').values.item())

		# allocate input/output fields for the stencils
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._s     = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._p     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._exn   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._ddmtg = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._dmtg  = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._h     = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._mtg   = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._diagnostics = {
			'air_pressure_on_interface_levels': self._p,
			'exner_function_on_interface_levels': self._exn,
			'height_on_interface_levels': self._h,
			'montgomery_potential': self._mtg
		}
		if moist:
			self._rho  = np.zeros((nx, ny, nz), dtype=dtype)
			self._temp = np.zeros((nx, ny, nz), dtype=dtype)
			self._diagnostics.update({
				'air_density': self._rho,
				'air_temperature': self._temp
			})

		# initialize the stencil diagnosing the pressure
		self._stencil_diagnosing_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_pressure_defs,
			inputs={'in_s': self._s, 'in_p': self._p},
			global_inputs={'g': self._g, 'dtheta': self._dz},
			outputs={'out_p': self._p},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

		# initialize the stencil diagnosing the Exner function
		self._stencil_diagnosing_exner = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_exner_defs,
			inputs={'in_p': self._p},
			global_inputs={
				'cp': self._cp, 'pref': self._pref, 'rd': self._rd
			},
			outputs={'out_exn': self._exn},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the first derivative with respect
		# to the potential temperature of the Montgomery potential
		self._stencil_diagnosing_dmtg = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_dmtg_defs,
			inputs={'in_ddmtg': self._ddmtg, 'in_dmtg': self._dmtg},
			global_inputs={'dtheta': self._dz},
			outputs={'out_dmtg': self._dmtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		# initialize the stencil diagnosing the height of the interface levels
		self._stencil_diagnosing_height = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_dmtg': self._dmtg},
			global_inputs={'g': self._g, 'thetabar': self._thetabar},
			outputs={'out_h': self._h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz)),
			mode=backend
		)

		# initialize the stencil diagnosing the Montgomery potential
		self._stencil_diagnosing_montgomery = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_dmtg': self._dmtg, 'in_mtg': self._mtg},
			global_inputs={'dtheta': self._dz},
			outputs={'out_mtg': self._mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

		if moist:
			# initialize the stencil diagnosing the density and temperature
			self._stencil_diagnosing_density_temperature = gt.NGStencil(
				definitions_func=self._stencil_diagnosing_density_temperature_defs,
				inputs={'in_s': self._s, 'in_p': self._p, 'in_h': self._h},
				global_inputs={'rd': self._rd, 'dtheta': self._dz},
				outputs={'out_rho': self._rho, 'out_temp': self._temp},
				domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
				mode=backend
			)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'dd_montgomery_potential': {'dims': dims, 'units': 'm^2 K^-2 s^-2'}
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels': {'dims': dims_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_z, 'units': 'J kg^-1 K^-1'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'}
		}
		if self._moist:
			return_dict.update({
				'air_density': {'dims': dims, 'units': 'kg m^-3'},
				'air_temperature': {'dims': dims, 'units': 'K'}
			})

		return return_dict

	def array_call(self, state):
		# diagnose the pressure
		self._s[...] = state['air_isentropic_density'][...]
		self._p[:, :, 0] = self._pt
		self._stencil_diagnosing_pressure.compute()

		# diagnose the Exner function
		self._stencil_diagnosing_exner.compute()

		# diagnose the first derivative with respect to the
		# potential temperature of the Montgomery potential
		self._ddmtg[...] = state['dd_montgomery_potential'][...]
		g = self._g.value
		thetabar = self._thetabar.value
		hs = self.grid.topography.profile.to_units('m').values
		self._dmtg[:, :, -1] = - (g / thetabar) * hs
		self._stencil_diagnosing_dmtg.compute()

		# diagnose the height of the interface levels
		self._stencil_diagnosing_height.compute()

		# diagnose the Montgomery potential
		dtheta = self._dz.value
		mtg_s = g * hs + thetabar * self._exn[:, :, -1]
		self._mtg[:, :, -1] = mtg_s + 0.5 * dtheta * self._dmtg[:, :, -1]
		self._stencil_diagnosing_montgomery.compute()

		if self._moist:
			# diagnose the density and temperature
			self._stencil_diagnosing_density_temperature.compute()

		return self._diagnostics

	@staticmethod
	def _stencil_diagnosing_pressure_defs(g, dtheta, in_s, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_p = gt.Equation()

		# computations
		out_p[k] = in_p[k-1] + dtheta * g * in_s[k-1]

		return out_p

	@staticmethod
	def _stencil_diagnosing_exner_defs(cp, pref, rd, in_p):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_exn = gt.Equation()

		# computations
		out_exn[k] = cp * (in_p[k] / pref) ** (rd / cp)

		return out_exn

	@staticmethod
	def _stencil_diagnosing_dmtg_defs(dtheta, in_ddmtg, in_dmtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_dmtg = gt.Equation()

		# computations
		out_dmtg[k] = in_dmtg[k+1] + dtheta * in_ddmtg[k]

		return out_dmtg

	@staticmethod
	def _stencil_diagnosing_height_defs(g, thetabar, in_dmtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_h = gt.Equation()

		# computations
		out_h[k] = - (thetabar / g) * in_dmtg[k]

		return out_h

	@staticmethod
	def _stencil_diagnosing_montgomery_defs(dtheta, in_dmtg, in_mtg):
		# vertical index
		k = gt.Index(axis=2)

		# output field
		out_mtg = gt.Equation()

		# computations
		out_mtg[k] = in_mtg[k+1] + dtheta * in_dmtg[k+1]

		return out_mtg

	@staticmethod
	def _stencil_diagnosing_density_temperature_defs(rd, dtheta, in_s, in_p, in_h):
		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_rho = gt.Equation()
		out_temp = gt.Equation()

		# computations
		out_rho[k] = in_s[k] * dtheta / (in_h[k] - in_h[k+1])
		out_temp[k] = 0.5 * (in_p[k] + in_p[k+1]) / (rd * out_rho[k])

		return out_rho, out_temp
