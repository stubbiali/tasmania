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
import tasmania as taz
import json
if __name__ == '__main__':
	from base_loader import BaseLoader
	from mounter import DatasetMounter
else:
	from .base_loader import BaseLoader
	from .mounter import DatasetMounter


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

		diff = state1[self._fname].to_units(self._funits).values - \
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

		diff = (state1[self._fname].to_units(self._funits).values -
			    state2[self._fname].to_units(self._funits).values) / \
			   state1[self._fname].to_units(self._funits).values
		state1['diff_of_' + self._fname] = taz.make_dataarray_3d(
			diff, self.get_grid(), '1'
		)

		return state1


if __name__ == '__main__':
	json_filename = '/home/tasmania-user/tasmania/scripts/python/config/diff_loader.json'

	_ = DifferenceLoader(json_filename)
	_ = DifferenceLoader(json_filename)
	obj = DifferenceLoader(json_filename)

	nt = obj.get_nt()
	_ = obj.get_grid()
	_ = obj.get_state(nt-1)
	_ = obj.get_initial_time()

	print('All right!')
