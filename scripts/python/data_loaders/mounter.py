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


class DatasetMounter:
	_ledger = {}

	def __new__(cls, filename):
		if filename not in DatasetMounter._ledger:
			DatasetMounter._ledger[filename] = super().__new__(cls)
			print('New instance of DatasetMounter created.')
		return DatasetMounter._ledger[filename]

	def __init__(self, filename):
		self._fname = filename
		domain, grid_type, self._states = taz.load_netcdf_dataset(filename)
		self._grid = domain.physical_grid if grid_type == 'physical' \
			else domain.numerical_grid

	def get_grid(self):
		return self._grid

	def get_nt(self):
		return len(self._states)

	def get_state(self, tlevel):
		state = self._states[tlevel]
		self._grid.update_topography(state['time'] - self._states[0]['time'])
		return state

