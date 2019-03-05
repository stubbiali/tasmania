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
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing


mfwv  = 'mass_fraction_of_water_vapor_in_air'
mfclw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw  = 'mass_fraction_of_precipitation_water_in_air'


class IsentropicHorizontalSmoothing(sympl.DiagnosticComponent):
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
		self._smoother = HorizontalSmoothing.factory(
			smooth_type, (nx, ny, nz), grid, smooth_damp_depth, smooth_coeff, 
			smooth_coeff_max, backend, dtype
		)
		self._smoother_moist = None if moist is None else \
			HorizontalSmoothing.factory(
				smooth_type, (nx, ny, nz), grid, smooth_moist_damp_depth, 
				smooth_moist_coeff, smooth_moist_coeff_max, backend, dtype
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
	def diagnostic_properties(self):
		return self.input_properties

	def array_call(self, state):
		if not hasattr(self, '_s_out'):
			self._allocate_outputs()

		self._smoother(state['air_isentropic_density'], self._s_out)
		self._smoother(state['x_momentum_isentropic'],  self._su_out)
		self._smoother(state['y_momentum_isentropic'],  self._sv_out)
		return_dict = {
			'air_isentropic_density': self._s_out,
			'x_momentum_isentropic':  self._su_out,
			'y_momentum_isentropic':  self._sv_out,
		}

		if self._moist:
			self._smoother_moist(state[mfwv],  self._qv_out)
			self._smoother_moist(state[mfclw], self._qc_out)
			self._smoother_moist(state[mfpw],  self._qr_out)
			return_dict[mfwv]  = self._qv_out
			return_dict[mfclw] = self._qc_out
			return_dict[mfpw]  = self._qr_out

		return return_dict

	def _allocate_outputs(self):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		self._s_out  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su_out = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv_out = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._qv_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qc_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qr_out = np.zeros((nx, ny, nz), dtype=dtype)
