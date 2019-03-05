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
import gridtools as gt
from tasmania.conf import datatype
from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.framework.dycore import DynamicalCore


class BurgersDynamicalCore(DynamicalCore):
	def __init__(
		self, grid, time_units, intermediate_tendencies=None,
		time_integration_scheme='forward_euler', flux_scheme='upwind',
		boundary=None, backend=gt.mode.NUMPY, dtype=datatype
	):
		assert grid.nz == 1

		super().__init__(
			grid, time_units, intermediate_tendencies,
			intermediate_diagnostics=None, substeps=0,
			fast_tendencies=None, fast_diagnostics=None
		)

		self._boundary = boundary

		self._stepper = BurgersStepper.factory(
			time_integration_scheme, grid, flux_scheme, boundary,
			backend, dtype
		)

	@property
	def _input_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-1'},
			'y_velocity': {'dims': dims, 'units': 'm s^-1'},
		}

	@property
	def _substep_input_properties(self):
		return {}

	@property
	def _tendency_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-2'},
			'y_velocity': {'dims': dims, 'units': 'm s^-2'},
		}

	@property
	def _substep_tendency_properties(self):
		return {}

	@property
	def _output_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-1'},
			'y_velocity': {'dims': dims, 'units': 'm s^-1'},
		}

	@property
	def _substep_output_properties(self):
		return {}

	@property
	def stages(self):
		return self._stepper.stages

	def substep_fractions(self):
		return 1

	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		raw_state_cd = {
			'time': raw_state['time'],
			'x_velocity':
				self._boundary.from_physical_to_computational_domain(raw_state['x_velocity']),
			'y_velocity':
				self._boundary.from_physical_to_computational_domain(raw_state['y_velocity']),
		}

		raw_tendencies_cd = {'time': raw_tendencies.get('time', raw_state['time'])}
		if 'x_velocity' in raw_tendencies:
			raw_tendencies_cd['x_velocity'] = \
				self._boundary.from_physical_to_computational_domain(
					raw_tendencies['x_velocity']
				)
		if 'y_velocity' in raw_tendencies:
			raw_tendencies_cd['y_velocity'] = \
				self._boundary.from_physical_to_computational_domain(
					raw_tendencies['y_velocity']
				)

		out_state_cd = self._stepper(stage, raw_state_cd, raw_tendencies_cd, timestep)

		out_state = {
			'time': out_state_cd['time'],
			'x_velocity': self._boundary.from_computational_to_physical_domain(
				out_state_cd['x_velocity'], (self._grid.nx, self._grid.ny, self._grid.nz)
			),
			'y_velocity': self._boundary.from_computational_to_physical_domain(
				out_state_cd['y_velocity'], (self._grid.nx, self._grid.ny, self._grid.nz)
			),
		}

		self._boundary.enforce(
			out_state['x_velocity'], raw_state['x_velocity'],
			field_name='x_velocity', time=out_state['time']
		)
		self._boundary.enforce(
			out_state['y_velocity'], raw_state['y_velocity'],
			field_name='y_velocity', time=out_state['time']
		)

		return out_state

	def substep_array_call(
		self, stage, substep, raw_state, raw_stage_state, raw_tmp_state,
		raw_tendencies, timestep
	):
		pass
