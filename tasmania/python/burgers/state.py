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
from datetime import timedelta
import numpy as np

from tasmania.python.utils.data_utils import make_dataarray_3d


class ZhaoSolutionFactory:
	def __init__(self, eps):
		self.eps = eps.to_units('m^2 s^-1').values.item()

	def __call__(self, grid, time, slice_x=None, slice_y=None, field_name='x_velocity'):
		eps = self.eps

		slice_x = slice(0, grid.nx) if slice_x is None else slice_x
		slice_y = slice(0, grid.ny) if slice_y is None else slice_y

		mi = slice_x.stop - slice_x.start
		mj = slice_y.stop - slice_y.start

		x = grid.x.to_units('m').values[slice_x]
		x = np.tile(x[:, np.newaxis, np.newaxis], (1, mj, grid.nz))
		y = grid.y.to_units('m').values[slice_y]
		y = np.tile(y[np.newaxis, :, np.newaxis], (mi, 1, grid.nz))

		t = time.total_seconds()

		if field_name == 'x_velocity':
			return - 2.0 * eps * 2.0 * np.pi * np.exp(- 5.0 * np.pi**2 * eps * t) * \
				np.cos(2.0 * np.pi * x) * np.sin(np.pi * y) / \
				(2.0 + np.exp(- 5.0 * np.pi**2 * eps * t) *
				np.sin(2.0 * np.pi * x) * np.sin(np.pi * y))
		elif field_name == 'y_velocity':
			return - 2.0 * eps * np.pi * np.exp(- 5.0 * np.pi**2 * eps * t) * \
				np.sin(2.0 * np.pi * x) * np.cos(np.pi * y) / \
				(2.0 + np.exp(- 5.0 * np.pi**2 * eps * t) *
				np.sin(2.0 * np.pi * x) * np.sin(np.pi * y))
		else:
			raise ValueError()


class ZhaoStateFactory:
	def __init__(self, eps):
		self._solution_factory = ZhaoSolutionFactory(eps)

	def __call__(self, grid, time):
		t = timedelta(seconds=0)

		u = self._solution_factory(grid, t, field_name='x_velocity')
		v = self._solution_factory(grid, t, field_name='y_velocity')

		return {
			'time': time,
			'x_velocity': make_dataarray_3d(u, grid, 'm s^-1', 'x_velocity'),
			'y_velocity': make_dataarray_3d(v, grid, 'm s^-1', 'y_velocity'),
		}
