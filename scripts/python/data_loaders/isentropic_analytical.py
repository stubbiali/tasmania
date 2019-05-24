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
import json
from sympl import DataArray
import tasmania as taz
try:
	from .base_loader import BaseLoader
	from .mounter import DatasetMounter
except ImportError:
	from base_loader import BaseLoader
	from mounter import DatasetMounter


class IsentropicAnalyticalLoader(BaseLoader):
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			filename = ''.join(data['filename'])

			self._dsmounter = DatasetMounter(filename)

			self._u = DataArray(
				data['initial_x_velocity']['value'],
				attrs={'units': data['initial_x_velocity']['units']}
			)
			self._t = DataArray(
				data['temperature']['value'],
				attrs={'units': data['temperature']['units']}
			)
			self._h = DataArray(
				data['mountain_height']['value'],
				attrs={'units': data['mountain_height']['units']}
			)
			self._a = DataArray(
				data['mountain_width']['value'],
				attrs={'units': data['mountain_width']['units']}
			)

	def get_nt(self):
		return self._dsmounter.get_nt()

	def get_grid(self):
		return self._dsmounter.get_grid()

	def get_initial_time(self):
		return self._dsmounter.get_state(0)['time']

	def get_state(self, tlevel):
		grid = self.get_grid()
		init_state = self._dsmounter.get_state(0)

		u, _ = taz.get_isothermal_isentropic_analytical_solution(
			grid, self._u, self._t, self._h, self._a
		)
		final_state = {'x_velocity_at_u_locations': u}

		state = taz.dict_subtract(
			final_state, init_state, unshared_variables_in_output=False
		)
		state.update({
			key: value for key, value in init_state.items()
			if key != 'x_velocity_at_u_locations'
		})

		return state
