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
	plot_horizontal_profile
	plot_vertical_profile
	plot_vertical_profile_with_respect_to_vertical_height
	plot_xy
"""
import matplotlib as mpl

from tasmania.utils.data_utils import get_numpy_arrays


def plot_horizontal_profile(grid, state, field_to_plot, levels, fig, ax, **kwargs):
	"""
	Given an input model state, visualize a specified field along a line orthogonal to
	either the :math:`xz`- or :math:`yz`-plane. In the plot, the spatial coordinate
	is on the x-axis, while the field goes on the y-axis.

	Parameters
	----------
	grid : computational grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str 
		String specifying the field to plot. This might be:

		* the name of a variable stored in the input model state.

	levels : dict
		Dictionary whose keys are the ids of the two axes orthogonal to the cross line,
		and whose values are the corresponding indices identifying the cross line itself.
		Note that one of the keys must necessarily be :obj:`2`.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments specifying different plotting settings.
		See :func:`~tasmania.plot.line_profile.plot_xy` for the complete list.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.

	Raises
	------
	ValueError :
		If the length of :obj:`levels` is not 2.
	KeyError :
		If :obj:`levels` does not contain the key :obj:`2`.
	"""
	if len(levels.keys()) < 2:
		raise ValueError('Not enough info to identify the cross line.')

	if len(levels.keys()) > 2:
		raise ValueError("""{} indices are given, but only 2 are needed to identify 
						 	the cross line.""".format(len(levels.keys())))

	if 2 not in levels.keys():
		raise KeyError('The input argument ''levels'' must contain the key ''2''.')
	else:
		z_slice = int(levels[2])

	x_slice = int(levels[0]) if 0 in levels.keys() else slice(0, None)
	y_slice = int(levels[1]) if 1 in levels.keys() else slice(0, None)

	if field_to_plot in state.keys():
		field = state[field_to_plot][x_slice, y_slice, z_slice]
	else:
		raise ValueError('Unknown field to plot {}.'.format(field_to_plot))

	if type(y_slice) is int:
		grid1d = grid.x[:] if field.shape[0] == grid.nx else grid.x_at_u_locations[:]
	else:
		grid1d = grid.y[:] if field.shape[0] == grid.ny else grid.y_at_v_locations[:]

	return plot_xy(grid1d, field, fig, ax, **kwargs)


def plot_vertical_profile(grid, state, field_to_plot, levels, fig, ax, **kwargs):
	"""
	Given an input model state, visualize a specified field along a vertical line,
	i.e., a line orthogonal to the :math:`xy`-plane.
	In the plot, the field is on the x-axis, while the vertical, possibly
	terrain-following coordinate goes on the y-axis.

	Parameters
	----------
	grid : computational grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str 
		String specifying the field to plot. This might be:

		* the name of a variable stored in the input model state.

	levels : dict
		Dictionary whose keys are the ids of the two axes orthogonal to the cross line, and whose
		values are the corresponding indices identifying the cross line itself.
		Note that the keys must necessarily be :obj:`0` and :obj:`1`.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments specifying different plotting settings.
		See :func:`~tasmania.plot.line_profile.plot_xy` for the complete list.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.

	Raises
	------
	ValueError :
		If the length of :obj:`levels` is not 2.
	KeyError :
		If :obj:`levels` does not contain the keys :obj:`0` and :obj:`1`.
	"""
	if len(levels.keys()) < 2:
		raise ValueError('Not enough info to identify the cross line.')

	if len(levels.keys()) > 2:
		raise ValueError("""{} indices are given, but only 2 are needed to identify 
						 	the cross line.""".format(len(levels.keys())))

	if not(0 in levels.keys() and 1 in levels.keys()):
		raise KeyError('The input argument ''levels'' must contain the keys ''0'' and ''1''.')

	x_slice = int(levels[0])
	y_slice = int(levels[1])
	z_slice = slice(0,None)

	if field_to_plot in state.keys():
		field = state[field_to_plot][x_slice, y_slice, z_slice]
	else:
		raise ValueError('Unknown field to plot {}.'.format(field_to_plot))

	grid1d = grid.z[:] if field.shape[0] == grid.nz else grid.z_on_interface_levels[:]

	return plot_xy(field, grid1d, fig, ax, **kwargs)


def plot_vertical_profile_with_respect_to_vertical_height(grid, state, field_to_plot,
														  levels, fig, ax, **kwargs):
	"""
	Given an input model state, visualize a specified field along a vertical line,
	i.e., a line orthogonal to the :math:`xy`-plane.
	In the plot, the field is on the x-axis, while the geometric height above the terrain
	goes on the y-axis.

	Parameters
	----------
	grid : computational grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str 
		String specifying the field to plot. This might be:

		* the name of a variable stored in the input model state.

	levels : dict
		Dictionary whose keys are the ids of the two axes orthogonal to the cross line, and whose
		values are the corresponding indices identifying the cross line itself.
		Note that the keys must necessarily be :obj:`0` and :obj:`1`.
	fig : `figure`, optional
		A :class:`matplotlib.pyplot.figure`.
	ax : `axes`, optional
		An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments to specify different plotting settings. 
		See :func:`~tasmania.plot.line_profile.plot_xy` for the complete list.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.

	Note
	----
	Either the grid or the model state must contain `height` or `height_on_interface_levels`.

	Raises
	------
	ValueError :
		If the length of :obj:`levels` is not 2.
	KeyError :
		If :obj:`levels` does not contain the keys :obj:`0` and :obj:`1`.
	"""
	if len(levels.keys()) < 2:
		raise ValueError('Not enough info to identify the cross line.')

	if len(levels.keys()) > 2:
		raise ValueError("""{} indices are given, but only 2 are needed to identify 
						 	the cross line.""".format(len(levels.keys())))

	if not(0 in levels.keys() and 1 in levels.keys()):
		raise KeyError('The input argument ''levels'' must contain the keys ''0'' and ''1''.')

	x_slice = int(levels[0])
	y_slice = int(levels[1])
	z_slice = slice(0, None)

	if field_to_plot in state.keys():
		field = state[field_to_plot][x_slice, y_slice, z_slice]
	else:
		raise ValueError('Unknown field to plot {}.'.format(field_to_plot))

	try:
		z = get_numpy_arrays(state, (x_slice, y_slice, z_slice),
						     ('height_on_interface_levels', 'height'))
	except KeyError:
		try:
			z = grid.height_on_interface_levels[:]
		except AttributeError:
			try:
				z = grid.height[:]
			except AttributeError:
				print("""Either the grid or the model state must contain ''height'' 
						 or ''height_on_interface_levels''.""")
				raise

	if field.shape[0] > z.shape[0]:
		raise ValueError("""Since the field {} is defined on the vertical half levels, 
							either the grid or the state must contain 
							''height_on_interface_levels''.""".format(field_to_plot))

	grid1d = z if field.shape[0] == z.shape else 0.5 * (z[:-1] + z[1:])

	return plot_xy(field, grid1d, fig, ax, **kwargs)


def plot_xy(x, y, fig, ax, **kwargs):
	"""
	Plot a line.

	Parameters
	----------
	x : array_like
		1-D :class:`numpy.ndarray` representing the :math:`x`-coordinates
		of the points to plot.
	y : array_like
		1-D :class:`numpy.ndarray` representing the :math:`y`-coordinates
		of the points to plot.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Default is 16.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	y_factor : float
		Scaling factor for the field. Default is 1.
	linecolor : str
		String specifying the line color. Default is 'blue'.
	linestyle : str
		String specifying the line style. The default line style is '-'.
	linewidth : float
		The line width. Default is 1.5.
	legend_label : str
		The legend label for the line. Default is an empty string.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Get keyword arguments
	fontsize     = kwargs.get('fontsize', 16)
	x_factor     = kwargs.get('x_factor', 1.)
	y_factor     = kwargs.get('y_factor', 1.)
	linecolor	 = kwargs.get('linecolor', 'blue')
	linestyle    = kwargs.get('linestyle', '-')
	linewidth    = kwargs.get('linewidth', 1.5)
	legend_label = kwargs.get('legend_label', '')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x *= x_factor
	y *= y_factor

	# Plot the field
	if legend_label == '' or legend_label is None:
		ax.plot(x, y, color=linecolor, linestyle=linestyle, linewidth=linewidth)
	else:
		ax.plot(x, y, color=linecolor, linestyle=linestyle, linewidth=linewidth,
				label=legend_label)

	# Bring axes back to original units
	x /= x_factor
	y /= y_factor

	return fig, ax
