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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tasmania.utils import plot_utils
from tasmania.utils.utils import equal_to as eq, smaller_than as lt


def plot_topography_3d(grid, state=None, field_to_plot='topography',
					   fig=None, ax=None, **kwargs):
	"""
	An utility to visualize a three-dimensional orography using the
	`mplot3d toolkit <https://matplotlib.org/tutorials/toolkits/mplot3d.html>`_.

	Parameters
	----------
	grid : grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : `dict`, optional
		A model state dictionary.
	field_to_plot : str
		String specifying the field to plot.
	fig : `figure`, optional
		A :class:`matplotlib.pyplot.figure`.
	ax : `axes`, optional
		An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
    fontsize : int
        The fontsize to be used. Default is 12.
    figsize : sequence
        Sequence representing the figure size. Default is [8,8].
        This argument is effective only if the figure is instantiated
        within the function, i.e., if both the :obj:`fig` and :obj:`ax`
        arguments are not given
    x_factor : float
        Scaling factor for the :math:`x`-axis. Default is 1.
    y_factor : float
        Scaling factor for the :math:`y`-axis. Default is 1.
    z_factor : float
        Scaling factor for the topography. Default is 1.
	draw_grid : bool
		:obj:`True` to draw the underlying grid, :obj:`False` otherwise.
		Default is :obj:`True`.
	cbar_on : bool
		:obj:`True` to show the color bar, :obj:`False` otherwise. Default is :obj:`True`.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided
		by Matplotlib, as well as the corresponding inverted versions, are available.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1,
		i.e., all ticks are displayed with the corresponding label.
	cbar_ticks_pos : str
		'center' to place the color bar ticks in the middle of the color intervals,
		anything else to place the ticks at the interfaces between color intervals.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers
		the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar
		covers the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the color bar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the color bar. By default,
		only the current axes are resized.

	Note
	----
	The arguments :obj:`state` and :obj:`field_to_plot` are dummy parameters,
	i.e., they get not used by the function. Nevertheless, they are retained
	in the function signature for compliancy with
	:class:`~tasmania.plot.plot_monitors.Plot3d`

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8, 8])
	x_factor         = kwargs.get('x_factor', 1.)
	y_factor         = kwargs.get('y_factor', 1.)
	z_factor     	 = kwargs.get('z_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', 'BrBG_r')
	cbar_on			 = kwargs.get('cbar_on', True)
	cbar_levels		 = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_ticks_pos	 = kwargs.get('cbar_ticks_pos', 'center')
	cbar_center		 = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label	 = kwargs.get('cbar_x_label', '')
	cbar_y_label	 = kwargs.get('cbar_y_label', '')
	cbar_title		 = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')
	cbar_ax			 = kwargs.get('cbar_ax', None)

	# Get figure and axes
	out_fig, out_ax = plot_utils.get_figure_and_axes(fig, ax, figsize=figsize,
													 fontsize=fontsize, projection='3d')

	# Extract axes and topography height
	nx, ny = grid.nx, grid.ny
	xv, yv = grid.x.values, grid.y.values
	xv = np.repeat(xv[:, np.newaxis], ny, axis=1)
	yv = np.repeat(np.reshape(yv[:, np.newaxis], (1, ny)), nx, axis=0)
	zv = grid.topography_height

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	xv *= x_factor
	yv *= y_factor
	zv *= z_factor

	# Create color bar for colormap
	field_min, field_max = np.amin(zv), np.amax(zv)
	if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center-field_min, field_max-cbar_center) \
			if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center-half_width, cbar_center+half_width
	color_levels = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint=True)
	if eq(color_levels[0], color_levels[-1]):
		color_levels = np.linspace(cbar_lb-1e-8, cbar_ub+1e-8, cbar_levels, endpoint=True)

	# Create colormap
	if cmap_name == 'BuRd':
		cm = plot_utils.reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the topography
	surf = out_ax.plot_surface(xv, yv, zv, cmap=cm, linewidth=.1)

	# Set the color bar
	if cbar_on:
		plot_utils.set_colorbar(out_fig, surf, color_levels,
								cbar_ticks_step=cbar_ticks_step,
								cbar_ticks_pos=cbar_ticks_pos, cbar_title=cbar_title,
								cbar_x_label=cbar_x_label, cbar_y_label=cbar_y_label,
								cbar_orientation=cbar_orientation, cbar_ax=cbar_ax)

	# Bring topography back to original dimensions
	zv /= z_factor

	return out_fig, out_ax
