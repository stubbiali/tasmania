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
import numpy as np
from sympl import DataArray
import tasmania as taz
try:
	from .base_loader import BaseLoader
	from .mounter import DatasetMounter
except ImportError:
	from base_loader import BaseLoader
	from mounter import DatasetMounter


class DifferenceLoader(BaseLoader):
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			filename1 = ''.join(data['filename1'])
			filename2 = ''.join(data['filename2'])

			self._dsmounter1 = DatasetMounter(filename1)
			self._dsmounter2 = DatasetMounter(filename2)

			self._fname = data['field_name']
			self._funits = data['field_units']

	def get_grid(self):
		return self._dsmounter1.get_grid()

	def get_nt(self):
		return self._dsmounter1.get_nt()

	def get_initial_time(self):
		return self._dsmounter1.get_state(0)['time']

	def get_state(self, tlevel):
		state1 = self._dsmounter1.get_state(tlevel)
		state2 = self._dsmounter2.get_state(tlevel)

		diff = \
			state1[self._fname].to_units(self._funits).values - \
			state2[self._fname].to_units(self._funits).values
		state1['diff_of_' + self._fname] = taz.make_dataarray_3d(
			diff, self.get_grid(), self._funits
		)

		return state1


class VelocityDifferenceLoader(DifferenceLoader):
	def __init__(self, json_filename):
		super().__init__(json_filename)

	def get_state(self, tlevel):
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

		diff = \
			state1[self._fname].to_units(self._funits).values - \
			state2[self._fname].to_units(self._funits).values
		state1['diff_of_' + self._fname] = taz.make_dataarray_3d(
        	diff, self.get_grid(), self._funits
		)

		return state1


class RelativeDifferenceLoader(BaseLoader):
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			filename1 = ''.join(data['filename1'])
			filename2 = ''.join(data['filename2'])

			self._dsmounter1 = DatasetMounter(filename1)
			self._dsmounter2 = DatasetMounter(filename2)

			self._fname = data['field_name']
			self._funits = data['field_units']

	def get_grid(self):
		return self._dsmounter1.get_grid()

	def get_nt(self):
		return self._dsmounter1.get_nt()

	def get_initial_time(self):
		return self._dsmounter1.get_state(0)['time']

	def get_state(self, tlevel):
		state1 = self._dsmounter1.get_state(tlevel)
		state2 = self._dsmounter2.get_state(tlevel)

		field1 = state1[self._fname].to_units(self._funits).values
		field2 = state2[self._fname].to_units(self._funits).values

		field1[np.where(field1 < 1e-3)] = 0.0
		field2[np.where(field2 < 1e-3)] = 0.0

		diff = (field1 - field2) / field2
		diff[np.where(np.isnan(diff))] = 0.0
		diff[np.where(np.isinf(diff))] = 0.0
		state1['rdiff_of_' + self._fname] = taz.make_dataarray_3d(
			diff, self.get_grid(), '1'
		)

		return state1


class RMSDLoader(BaseLoader):
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			filename1 = ''.join(data['filename1'])
			filename2 = ''.join(data['filename2'])

			self._dsmounter1 = DatasetMounter(filename1)
			self._dsmounter2 = DatasetMounter(filename2)

			self._fname = data['field_name']
			self._funits = data['field_units']

			start, stop, step = data['x1']
			self._x1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['y1']
			self._y1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['z1']
			self._z1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['x2']
			self._x2 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['y2']
			self._y2 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['z2']
			self._z2 = None if start == stop == step is None else slice(start, stop, step)

	def get_grid(self):
		return self._dsmounter1.get_grid()

	def get_nt(self):
		return self._dsmounter1.get_nt()

	def get_initial_time(self):
		return self._dsmounter1.get_state(0)['time']

	def get_state(self, tlevel):
		fname, funits = self._fname, self._funits
		x1, y1, z1 = self._x1, self._y1, self._z1
		x2, y2, z2 = self._x2, self._y2, self._z2

		tlevel = self._dsmounter1.get_nt() + tlevel if tlevel < 0 else tlevel
		state1 = self._dsmounter1.get_state(tlevel)
		state2 = self._dsmounter2.get_state(tlevel)

		field1 = state1[fname].to_units(funits).values[x1, y1, z1]
		field2 = state2[fname].to_units(funits).values[x2, y2, z2]
		raw_rmsd = np.linalg.norm(field1 - field2) / np.sqrt(field1.size)
		state1['rmsd_of_' + fname] = DataArray(
			np.array(raw_rmsd)[np.newaxis, np.newaxis, np.newaxis]
		)

		return state1


class RMSDVelocityLoader(RMSDLoader):
	def __init__(self, json_filename):
		super().__init__(json_filename)

	def get_state(self, tlevel):
		fname, funits = self._fname, self._funits
		x1, y1, z1 = self._x1, self._y1, self._z1
		x2, y2, z2 = self._x2, self._y2, self._z2

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

		field1 = state1[fname].to_units(funits).values[x1, y1, z1]
		field2 = state2[fname].to_units(funits).values[x2, y2, z2]
		raw_rmsd = np.linalg.norm(field1 - field2) / np.sqrt(field1.size)
		state1['rmsd_of_' + fname] = DataArray(
			np.array(raw_rmsd)[np.newaxis, np.newaxis, np.newaxis]
		)

		return state1


class RRMSDLoader(BaseLoader):
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			filename1 = ''.join(data['filename1'])
			filename2 = ''.join(data['filename2'])

			self._dsmounter1 = DatasetMounter(filename1)
			self._dsmounter2 = DatasetMounter(filename2)

			self._fname = data['field_name']
			self._funits = data['field_units']

			start, stop, step = data['x1']
			self._x1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['y1']
			self._y1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['z1']
			self._z1 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['x2']
			self._x2 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['y2']
			self._y2 = None if start == stop == step is None else slice(start, stop, step)
			start, stop, step = data['z2']
			self._z2 = None if start == stop == step is None else slice(start, stop, step)

	def get_grid(self):
		return self._dsmounter1.get_grid()

	def get_nt(self):
		return self._dsmounter1.get_nt()

	def get_initial_time(self):
		return self._dsmounter1.get_state(0)['time']

	def get_state(self, tlevel):
		fname, funits = self._fname, self._funits
		x1, y1, z1 = self._x1, self._y1, self._z1
		x2, y2, z2 = self._x2, self._y2, self._z2

		tlevel = self._dsmounter1.get_nt() + tlevel if tlevel < 0 else tlevel
		state1 = self._dsmounter1.get_state(tlevel)
		state2 = self._dsmounter2.get_state(tlevel)

		field1 = state1[fname].to_units(funits).values[x1, y1, z1]
		field2 = state2[fname].to_units(funits).values[x2, y2, z2]
		raw_rmsd = np.linalg.norm(field1 - field2) / np.linalg.norm(field2)
		state1['rrmsd_of_' + fname] = DataArray(
			np.array(raw_rmsd)[np.newaxis, np.newaxis, np.newaxis]
		)

		return state1
