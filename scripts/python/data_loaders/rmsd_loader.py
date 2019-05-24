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
from .base_loader import BaseLoader
import json
from .mounter import DatasetMounter
import tasmania as taz


class RMSDLoader(BaseLoader):
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			filename1 = ''.join(data['filename1'])
			filename2 = ''.join(data['filename2'])

			self._dsmounter1 = DatasetMounter(filename1)
			self._dsmounter2 = DatasetMounter(filename2)

			grid1 = self._dsmounter1.get_grid()
			grid2 = self._dsmounter2.get_grid()

			field_name = data['field_name']
			field_units = data['field_units']

			start, stop, step = data['x1']
			x1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['y1']
			y1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['z1']
			z1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['x2']
			x2 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['y2']
			y2 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['z2']
			z2 = None if start == stop == step is None else slice(start, stop, step)

			self._rmsd = taz.RMSD(
				(grid1, grid2), {field_name: field_units},
				x=(x1, x2), y=(y1, y2), z=(z1, z2)
			)

	def get_grid(self):
		return self._dsmounter1.get_grid()

	def get_nt(self):
		return self._dsmounter1.get_nt()

	def get_initial_time(self):
		return self._dsmounter1.get_state(0)['time']

	def get_state(self, tlevel):
		tlevel = self._dsmounter1.get_nt() + tlevel if tlevel < 0 else tlevel
		state1 = self._dsmounter1.get_state(tlevel)
		state2 = self._dsmounter2.get_state(tlevel)

		diagnostics = self._rmsd(state1, state2)
		state1.update(diagnostics)

		return state1


class RMSDVelocityLoader(RMSDLoader):
	def __init__(self, json_filename):
		super().__init__(json_filename)

	def get_state(self, tlevel):
		tlevel = self._dsmounter1.get_nt() + tlevel if tlevel < 0 else tlevel
		state1 = self._dsmounter1.get_state(tlevel)
		state2 = self._dsmounter2.get_state(tlevel)

		try:
			u = state1['x_momentum'].to_units('kg m^-2 s^-1').values / \
				state1['air_density'].to_units('kg m^-3').values
			state1['x_velocity'] = taz.make_dataarray_3d(u, self.get_grid(), 'm s^-1')
			v = state1['y_momentum'].to_units('kg m^-2 s^-1').values / \
				state1['air_density'].to_units('kg m^-3').values
			state1['y_velocity'] = taz.make_dataarray_3d(v, self.get_grid(), 'm s^-1')

			u = state2['x_momentum'].to_units('kg m^-2 s^-1').values / \
				state2['air_density'].to_units('kg m^-3').values
			state2['x_velocity'] = taz.make_dataarray_3d(u, self.get_grid(), 'm s^-1')
			v = state2['y_momentum'].to_units('kg m^-2 s^-1').values / \
				state2['air_density'].to_units('kg m^-3').values
			state2['y_velocity'] = taz.make_dataarray_3d(v, self.get_grid(), 'm s^-1')
		except KeyError:
			u = state1['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values / \
				state1['air_isentropic_density'].to_units('kg m^-2 K^-1').values
			state1['x_velocity'] = taz.make_dataarray_3d(u, self.get_grid(), 'm s^-1')
			v = state1['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values / \
				state1['air_isentropic_density'].to_units('kg m^-2 K^-1').values
			state1['y_velocity'] = taz.make_dataarray_3d(v, self.get_grid(), 'm s^-1')

			u = state2['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values / \
				state2['air_isentropic_density'].to_units('kg m^-2 K^-1').values
			state2['x_velocity'] = taz.make_dataarray_3d(u, self.get_grid(), 'm s^-1')
			v = state2['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values / \
				state2['air_isentropic_density'].to_units('kg m^-2 K^-1').values
			state2['y_velocity'] = taz.make_dataarray_3d(v, self.get_grid(), 'm s^-1')

		diagnostics = self._rmsd(state1, state2)
		state1.update(diagnostics)

		return state1


if __name__ == '__main__':
	json_filename = '/home/tasmania-user/tasmania/scripts/python/config/rmsd_loader.json'

	_ = RMSDLoader(json_filename)
	_ = RMSDLoader(json_filename)
	obj = RMSDLoader(json_filename)

	nt = obj.get_nt()
	_ = obj.get_grid()
	_ = obj.get_state(nt-1)
	_ = obj.get_initial_time()

	print('All right!')
