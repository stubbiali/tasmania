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
	LineProfile(Drawer)
	make_xplot
	make_yplot
	make_zplot
	make_hplot
"""
import numpy as np

from tasmania.plot.drawer import Drawer
from tasmania.plot.plot_utils import make_lineplot
from tasmania.plot.retrievers import DataRetriever
from tasmania.plot.utils import to_units


class LineProfile(Drawer):
	"""
	Drawer which plots the profile of a given quantity along a line
	perpendicular to a coordinate plane.
	If the line is horizontal (respectively, vertical), the spatial
	coordinate is on the x-axis (resp., y-axis) and the quantity is
	on the y-axis (resp., x-axis).
	"""
	def __init__(self, grid, field_name, field_units,
				 x=None, y=None, z=None,
				 axis_name=None, axis_units=None,
				 axis_x=None, axis_y=None, axis_z=None,
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
		**kwargs :
			Keyword arguments specifying plot-specific settings.
			See :func:`tasmania.plot.utils.make_lineplot`.
		"""
		super().__init__(**kwargs)

		flag_x = 0 if x is None else 1
		flag_y = 0 if y is None else 1
		flag_z = 0 if z is None else 1
		if flag_x + flag_y + flag_z != 2:
			raise ValueError('A line is uniquely identified by two indices, but here '
				'x is{}given, y is{}given and z is{}given.'.format(
					' ' if flag_x else ' not ', ' ' if flag_y else ' not ',
					' ' if flag_z else ' not ',
				)
			)

		slice_x = slice(x, x+1 if x != -1 else None, None) if flag_x else None
		slice_y = slice(y, y+1 if y != -1 else None, None) if flag_y else None
		slice_z = slice(z, z+1 if z != -1 else None, None) if flag_z else None

		retriever = DataRetriever(grid, field_name, field_units,
								  slice_x, slice_y, slice_z)

		if not flag_x:
			self._slave = lambda state, ax: make_xplot(grid, axis_units, retriever,
													   state, ax, **kwargs)
		elif not flag_y:
			self._slave = lambda state, ax: make_yplot(grid, axis_units, retriever,
													   state, ax, **kwargs)
		else:
			aname = 'z' if axis_name is None else axis_name
			if aname != 'z':
				ax = axis_x if axis_x is not None else x
				ay = axis_y if axis_y is not None else y
				aslice_x = slice(ax, ax+1 if ax != -1 else None, None)
				aslice_y = slice(ay, ay+1 if ay != -1 else None, None)
				axis_retriever = DataRetriever(grid, aname, axis_units,
											   aslice_x, aslice_y)
				self._slave = lambda state, ax: make_hplot(axis_retriever, retriever,
														   state, ax, **kwargs)
			else:
				self._slave = lambda state, ax: make_zplot(grid, axis_units, retriever,
														   state, ax, **kwargs)

	def __call__(self, state, fig=None, ax=None):
		"""
		Call operator generating the plot.

		Returns
		-------
		x : array_like
			1-D :class:`numpy.ndarray` gathering the x-coordinates
			of the plotted points.
		y : array_like
			1-D :class:`numpy.ndarray` gathering the y-coordinates
			of the plotted points.
		"""
		return self._slave(state, ax)


def make_xplot(grid, axis_units, field_retriever, state, ax=None, **kwargs):
	y = np.squeeze(field_retriever(state))
	x = to_units(grid.x, axis_units).values if y.shape[0] == grid.nx \
		else to_units(grid.x_at_u_locations, axis_units).values

	if ax is not None:
		make_lineplot(x, y, ax, **kwargs)

	return x, y


def make_yplot(grid, axis_units, field_retriever, state, ax=None, **kwargs):
	y = np.squeeze(field_retriever(state))
	x = to_units(grid.y, axis_units).values if y.shape[0] == grid.ny \
		else to_units(grid.y_at_v_locations, axis_units).values

	if ax is not None:
		make_lineplot(x, y, ax, **kwargs)

	return x, y


def make_zplot(grid, axis_units, field_retriever, state, ax=None, **kwargs):
	x = np.squeeze(field_retriever(state))
	y = to_units(grid.z, axis_units).values if x.shape[0] == grid.nz \
		else to_units(grid.z_on_interface_levels, axis_units).values

	if ax is not None:
		make_lineplot(x, y, ax, **kwargs)

	return x, y


def make_hplot(axis_retriever, field_retriever, state, ax=None, **kwargs):
	xv = np.squeeze(field_retriever(state))
	yv = np.squeeze(axis_retriever(state))

	x = 0.5 * (xv[:-1] + xv[1:]) if xv.shape[0] > yv.shape[0] else xv
	y = 0.5 * (yv[:-1] + yv[1:]) if xv.shape[0] < yv.shape[0] else yv

	if ax is not None:
		make_lineplot(x, y, ax, **kwargs)

	return x, y
