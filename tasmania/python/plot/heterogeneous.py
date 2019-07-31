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
"""
This module contains:
	Line(Drawer)
"""
import numpy as np
from sympl import DataArray

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.retrievers import DataRetriever
from tasmania.python.plot.plot_utils import make_lineplot


class Line(Drawer):
	"""
	Draw a line by retrieving a scalar value from multiple states.
	"""
	def __init__(
		self, grids, field_name, field_units, x, y, z,
		xdata=None, ydata=None, properties=None
	):
		"""
		Parameters
		----------
		grids : tuple[tasmania.Grid]
			The :class:`tasmania.Grid`s underlying the states.
		field_name : str
			The state quantity to visualize.
		field_units : str
			The units for the quantity to visualize.
		x : `int` or `tuple[int]`
			For each state, the index along the first dimension of the field array
			identifying the grid point to consider. If the same index applies to
			all states, it can be specified as an integer.
		y : `int` or `tuple[int]`
			For each state, the index along the second dimension of the field array
			identifying the grid point to consider. If the same index applies to
			all states, it can be specified as an integer.
		z : `int` or `tuple[int]`
			For each state, the index along the third dimension of the field array
			identifying the grid point to consider. If the same index applies to
			all states, it can be specified as an integer.
		xdata : `np.ndarray`, optional
			The data to be placed on the horizontal axis of the plot. If specified,
			the data retrieved from the states will be placed on the vertical axis
			of the plot. Only allowed if ``ydata`` is not given.
		ydata : `np.ndarray`, optional
			The data to be placed on the vertical axis of the plot. If specified,
			the data retrieved from the states will be placed on the horizontal axis
			of the plot. Only allowed if ``xdata`` is not given.
		properties : `dict`, optional
			Dictionary whose keys are strings denoting plot-specific
			settings, and whose values specify values for those settings.
			See :func:`tasmania.python.plot.utils.make_lineplot`.
		"""
		super().__init__(properties)

		x = [x, ]*len(grids) if isinstance(x, int) else x
		y = [y, ]*len(grids) if isinstance(y, int) else y
		z = [z, ]*len(grids) if isinstance(z, int) else z

		self._retrievers = []
		for k in range(len(grids)):
			slice_x = slice(x[k], x[k]+1 if x[k] != -1 else None)
			slice_y = slice(y[k], y[k]+1 if y[k] != -1 else None)
			slice_z = slice(z[k], z[k]+1 if z[k] != -1 else None)
			self._retrievers.append(
				DataRetriever(grids[k], field_name, field_units, slice_x, slice_y, slice_z)
			)

		assert xdata is None or ydata is None, \
			"Both xdata and ydata are given, but only one is allowed."

		if xdata is not None:
			self._xdata = xdata
			self._ydata = []
			self._data_on_yaxis = True
		else:
			self._xdata = []
			self._ydata = ydata
			self._data_on_yaxis = False

	def __call__(self, state, fig=None, ax=None):
		if self._data_on_yaxis:
			self._ydata.append
