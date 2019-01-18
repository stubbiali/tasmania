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
	TimeSeries(Drawer)
	HovmollerDiagram(Drawer)
"""
import numpy as np
from sympl import DataArray

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.profile import LineProfile
from tasmania.python.plot.retrievers import DataRetriever
from tasmania.python.plot.plot_utils import make_contourf, make_lineplot


class TimeSeries(Drawer):
	"""
	Drawer which visualizes a time series.
	"""
	def __init__(self, grid, field_name, field_units, x=0, y=0, z=0,
				 time_mode='elapsed', init_time=None, time_units='s',
				 time_on_xaxis=True, **kwargs):
		"""
		Parameters
		----------
		grid : grid
			Instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes representing the underlying grid.
		field_name : str
			The state quantity to visualize.
		field_units : str
			The units for the quantity to visualize.
		x : `int`, optional
			Index along the first dimension of the field array identifying the line
			to visualize. Defaults to 0.
		y : `int`, optional
			Index along the second dimension of the field array identifying the line
			to visualize. Defaults to 0.
		z : `int`, optional
			Index along the third dimension of the field array identifying the line
			to visualize. Defaults to 0.
		time_mode : `str`, optional
			Either 'elapsed' - to plot data against the elapsed time, or
			'absolute' - to plot data against the absolute time. Defaults to 'elapsed'.
		init_time : `datetime`, optional
			Initial time of the simulation.
			Only effective if :obj:`time_mode` set on 'elapsed'.
			If not specified, the time at which the first given state
			is defined is assumed to be the initial time of the simulation.
		time_units : `str`, optional
			Units for time. Defaults to 's' (seconds).
			Only effective if :obj:`time_mode` set on 'elapsed'.
		time_on_xaxis : `bool`, optional
			:obj:`True` to place time on the plot x-axis, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		**kwargs :
			Keyword arguments specifying plot-specific settings.
			See :func:`tasmania.plot.utils.make_lineplot`.
		"""
		super().__init__(**kwargs)

		slice_x = slice(x, x+1 if x != -1 else None)
		slice_y = slice(y, y+1 if y != -1 else None)
		slice_z = slice(z, z+1 if z != -1 else None)

		self._retriever = DataRetriever(grid, field_name, field_units,
								  		slice_x, slice_y, slice_z)

		self._tmode  = time_mode
		self._itime  = init_time
		self._uitime = init_time is not None
		self._tunits = time_units
		self._txaxis = time_on_xaxis

		self._time = []
		self._data = []

	def reset(self):
		self._time = []
		self._data = []

		if not self._uitime:
			self._itime = None

	def __call__(self, state, fig=None, ax=None):
		"""
		Call operator updating and visualizing the time series.
		"""
		if self._tmode == 'elapsed':
			self._itime = state['time'] if self._itime is None else self._itime
			ctime = DataArray((state['time'] - self._itime).total_seconds(),
							  attrs={'units': 's'})
			self._time.append(ctime.to_units(self._tunits).values.item())
		else:
			self._time.append(state['time'])

		self._data.append(self._retriever(state).item())

		x = np.array(self._time if self._txaxis else self._data)
		y = np.array(self._data if self._txaxis else self._time)

		if ax is not None:
			make_lineplot(x, y, ax, **self.properties)


class HovmollerDiagram(Drawer):
	"""
	Drawer which generates a Hovmoller diagram.
	"""
	def __init__(self, grid, field_name, field_units,
				 x=None, y=None, z=None,
				 axis_name=None, axis_units=None,
				 axis_x=None, axis_y=None, axis_z=None,
				 time_mode='elapsed', init_time=None, time_units='s',
				 **kwargs):
		"""
		Parameters
		----------
		grid : grid
			Instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes representing the underlying grid.
		field_name : str
			The state quantity to visualize.
		field_units : str
			The units for the quantity to visualize.
		x : `int`, optional
			Index along the first dimension of the field array identifying the line
			to visualize. Not to be specified if both :obj:`y` and :obj:`z` are given.
		y : `int`, optional
			Index along the second dimension of the field array identifying the line
			to visualize. Not to be specified if both :obj:`y` and :obj:`z` are given.
		z : `int`, optional
			Index along the third dimension of the field array identifying the line
			to visualize. Not to be specified if both :obj:`y` and :obj:`z` are given.
		axis_name : `str`, optional
			The name of the spatial axis. Options are:

				* 'x' (default and only effective if :obj:`x` is not given);
				* 'y' (default and only effective if :obj:`y` is not given);
				* 'z' (default and only effective if :obj:`z` is not given);
				* 'height' (only effective if :obj:`z` is not given);
				* 'height_on_interface_levels' (only effective if :obj:`z` is not given);
				* 'air_pressure' (only effective if :obj:`z` is not given).
				* 'air_pressure_on_interface_levels' (only effective if :obj:`z` is not given).

		axis_units : `str`, optional
			Units for the spatial axis. If not specified, the native units of
			the axis are used
		axis_x : `int`, optional
			Index along the first dimension of the axis array identifying the line
			to visualize. Defaults to :obj:`x`.
			Only effective if :obj:`axis_name` is 'height' or 'air_pressure'.
			Not to be specified if both :obj:`axis_y` and :obj:`axis_z` are given.
		axis_y : `int`, optional
			Index along the first dimension of the axis array identifying the line
			to visualize. Defaults to :obj:`x`.
			Only effective if :obj:`axis_name` is 'height' or 'air_pressure'.
			Not to be specified if both :obj:`axis_y` and :obj:`axis_z` are given.
		axis_z : `int`, optional
			Index along the first dimension of the axis array identifying the line
			to visualize. Defaults to :obj:`x`.
			Only effective if :obj:`axis_name` is 'height' or 'air_pressure'.
			Not to be specified if both :obj:`axis_y` and :obj:`axis_z` are given.
		time_mode : `str`, optional
			Either 'elapsed' - to plot data against the elapsed time, or
			'absolute' - to plot data against the absolute time. Defaults to 'elapsed'.
		init_time : `datetime`, optional
			Initial time of the simulation.
			Only effective if :obj:`time_mode` set on 'elapsed'.
			If not specified, the time at which the first given state
			is defined is assumed to be the initial time of the simulation.
		time_units : `str`, optional
			Units for time. Defaults to 's' (seconds).
			Only effective if :obj:`time_mode` set on 'elapsed'.
		**kwargs :
			Keyword arguments specifying plot-specific settings.
			See :func:`tasmania.plot.utils.make_contourf`.
		"""
		super().__init__(**kwargs)

		self._retriever = LineProfile(grid, field_name, field_units, x, y, z,
									  axis_name, axis_units, axis_x, axis_y, axis_z)

		self._tmode  = time_mode
		self._itime  = init_time
		self._uitime = init_time is not None
		self._tunits = time_units
		self._txaxis = z is None

		self._time = []
		self._axis = None
		self._data = None

	def reset(self):
		self._time = []
		self._axis = None
		self._data = None

		if not self._uitime:
			self._itime = None

	def __call__(self, state, fig=None, ax=None):
		"""
		Call operator generating the plot.
		"""
		if self._tmode == 'elapsed':
			self._itime = state['time'] if self._itime is None else self._itime
			ctime = DataArray((state['time'] - self._itime).total_seconds(),
							  attrs={'units': 's'})
			self._time.append(ctime.to_units(self._tunits).values.item())
		else:
			self._time.append(state['time'])

		if self._txaxis:
			field, spatial_axis = self._retriever(state)

			self._axis = spatial_axis[np.newaxis, :] if self._axis is None else \
						 np.concatenate((self._axis, spatial_axis[np.newaxis, :]), axis=0)
			self._data = field[np.newaxis, :] if self._data is None else \
						 np.concatenate((self._data, field[np.newaxis, :]), axis=0)

			x   = np.repeat(np.array(self._time)[:, np.newaxis], self._axis.shape[1], axis=1)
			y   = self._axis
			val = self._data
		else:
			spatial_axis, field = self._retriever(state)

			self._axis = spatial_axis[:, np.newaxis] if self._axis is None else \
						 np.concatenate((self._axis, spatial_axis[:, np.newaxis]), axis=1)
			self._data = field[:, np.newaxis] if self._data is None else \
						 np.concatenate((self._data, field[:, np.newaxis]), axis=1)

			x   = self._axis
			y   = np.repeat(np.array(self._time)[np.newaxis, :], self._axis.shape[0], axis=0)
			val = self._data

		if not((fig is None) or (ax is None)):
			make_contourf(x, y, val, fig, ax, **self.properties)
