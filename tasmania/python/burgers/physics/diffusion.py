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
from sympl import TendencyComponent

import gridtools as gt
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float64


class BurgersHorizontalDiffusion(TendencyComponent):
	def __init__(
		self, grid, diffusion_type, diffusion_coeff,
		backend=gt.mode.NUMPY, dtype=datatype
	):
		assert grid.nz == 1

		self._grid = grid

		super().__init__()

		nx, ny = grid.nx, grid.ny

		self._diffuser = HorizontalDiffusion.factory(
			diffusion_type, (nx, ny, 1), grid, diffusion_damp_depth=0,
			diffusion_coeff=diffusion_coeff.to_units('m^2 s^-1').values.item(),
			diffusion_coeff_max=diffusion_coeff.to_units('m^2 s^-1').values.item(),
			xaxis_units='m', yaxis_units='m', backend=backend, dtype=dtype
		)

		self._out_u = np.zeros((nx, ny, 1), dtype=dtype)
		self._out_v = np.zeros((nx, ny, 1), dtype=dtype)

	@property
	def input_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-1'},
			'y_velocity': {'dims': dims, 'units': 'm s^-1'},
		}

	@property
	def tendency_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-2'},
			'y_velocity': {'dims': dims, 'units': 'm s^-2'},
		}

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		self._diffuser(state['x_velocity'], self._out_u)
		self._diffuser(state['y_velocity'], self._out_v)

		tendencies = {'x_velocity': self._out_u, 'y_velocity': self._out_v}
		diagnostics = {}

		return tendencies, diagnostics
