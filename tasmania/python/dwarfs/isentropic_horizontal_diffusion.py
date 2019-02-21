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
import numpy as np
import sympl

import gridtools as gt
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion


mfwv  = 'mass_fraction_of_water_vapor_in_air'
mfclw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw  = 'mass_fraction_of_precipitation_water_in_air'


class IsentropicHorizontalDiffusion(sympl.TendencyComponent):
	def __init__(
		self, smooth_type, grid, smooth_damp_depth, smooth_coeff, smooth_coeff_max, 
		moist=False, smooth_moist_damp_depth=0, smooth_moist_coeff=1.0, 
		smooth_moist_coeff_max=1.0, backend=gt.mode.NUMPY, dtype=np.float64,
		**kwargs
	):
		self._grid = grid
		self._moist = moist
		self._dtype = dtype

		super().__init__(**kwargs)

		nx, ny, nz = grid.nx, grid.ny, grid.nz
		self._diffuser = HorizontalDiffusion.factory(
			smooth_type, (nx, ny, nz), grid, smooth_damp_depth, smooth_coeff, 
			smooth_coeff_max, 'm', 'm', backend, dtype
		)
		self._diffuser_moist = None if moist is None else \
			HorizontalDiffusion.factory(
				smooth_type, (nx, ny, nz), grid, smooth_moist_damp_depth, 
				smooth_moist_coeff, smooth_moist_coeff_max, 'm', 'm', backend, dtype
			)

	@property
	def input_properties(self):
		g = self._grid
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		if self._moist:
			return_dict[mfwv]  = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfclw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw]  = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def tendency_properties(self):
		g = self._grid
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic':  {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		if self._moist:
			return_dict[mfwv]  = {'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict[mfclw] = {'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict[mfpw]  = {'dims': dims, 'units': 'g g^-1 s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		if not hasattr(self, '_s_tnd'):
			self._allocate_outputs()

		self._diffuser(state['air_isentropic_density'], self._s_tnd)
		self._diffuser(state['x_momentum_isentropic'],  self._su_tnd)
		self._diffuser(state['y_momentum_isentropic'],  self._sv_tnd)

		return_dict = {
			'air_isentropic_density': self._s_tnd,
			'x_momentum_isentropic':  self._su_tnd,
			'y_momentum_isentropic':  self._sv_tnd,
		}

		if self._moist:
			self._diffuser_moist(state[mfwv],  self._qv_tnd)
			self._diffuser_moist(state[mfclw], self._qc_tnd)
			self._diffuser_moist(state[mfpw],  self._qr_tnd)
			return_dict[mfwv]  = self._qv_tnd
			return_dict[mfclw] = self._qc_tnd
			return_dict[mfpw]  = self._qr_tnd

		return return_dict, {}

	def _allocate_outputs(self):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		self._s_tnd  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._qv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
			self._qc_tnd = np.zeros((nx, ny, nz), dtype=dtype)
			self._qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)
