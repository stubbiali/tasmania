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
	make_quiver_xy
	plot_quiver_xy
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tasmania.utils import plot_utils
from tasmania.utils.data_utils import get_numpy_arrays
from tasmania.utils.utils import equal_to as eq, smaller_or_equal_than as lt


def make_quiver_xy(grid, state, field_to_plot, z_level, fig, ax, **kwargs):
	"""
	Generate the quiver plot of a vector field at a cross section parallel
	to the :math:`xy`-plane.

	Parameters
	----------
	grid : grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str 
		String specifying the field to plot. This might be:

		* 'horizontal_velocity', for the horizontal velocity; the current object must \
			contain either:
		
			- `x_velocity` and `y_velocity`;
			- `x_velocity_at_u_locations` and `y_velocity_at_v_locations`;
			- `air_density`, `x_momentum`, and `y_momentum`;
			- `air_isentropic_density`, `x_momentum_isentropic`, and \
				`y_momentum_isentropic`.

	z_level : int 
		:math:`z`-level identifying the cross-section.
    fig : figure
        A :class:`matplotlib.pyplot.figure`.
    ax : axes
        An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments specifying different plotting settings.
		See :func:`~tasmania.plot.quiver_xy.plot_quiver_xy` for the complete list.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Extract, compute, or interpolate the field to plot
	if field_to_plot == 'horizontal_velocity':
		try:
			vx, vy = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'x_velocity', 'y_velocity')
		except KeyError:
			pass

		try:
			u, v = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'x_velocity_at_u_locations', 'y_velocity_at_v_locations')
			vx = 0.5 * (u[:-1, :] + u[1:, :])
			vy = 0.5 * (v[:, :-1] + v[:, 1:])
		except KeyError:
			pass

		try:
			r, ru, rv = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'air_density', 'x_momentum', 'y_momentum')
			vx, vy = ru / r, rv / r
		except KeyError:
			pass

		try:
			s, su, sv = get_numpy_arrays(state, (slice(0, None), slice(0, None), z_level),
				'air_isentropic_density', 'x_momentum_isentropic', 'y_momentum_isentropic')
			vx, vy = su / s, sv / s
		except KeyError:
			pass

		scalar = np.sqrt(vx ** 2 + vy ** 2)
	else:
		raise RuntimeError('Unknown field to plot.')

	# Shortcuts
	nx, ny = grid.nx, grid.ny
	ni, nj = scalar.shape

	# The underlying x-grid
	x  = grid.x.values[:] if ni == nx else grid.x_at_u_locations.values[:]
	xv = np.repeat(x[:, np.newaxis], nj, axis=1)

	# The underlying y-grid
	y  = grid.y.values[:] if nj == ny else grid.y_at_v_locations.values[:]
	yv = np.repeat(y[np.newaxis, :], ni, axis=0)

	# The topography height
	topo_ = np.copy(grid.topography_height)
	topo  = np.zeros((ni, nj), dtype=topo_.dtype)
	if ni == nx and nj == ny:
		topo[:, :] = topo_[:, :]
	elif ni == nx + 1 and nj == ny:
		topo[1:-1, :] = 0.5 * (topo_[:-1, :] + topo_[1:, :])
		topo[0, :], topo[-1, :] = topo[1, :], topo[-2, :]
	elif ni == nx and nj == ny + 1:
		topo[:, 1:-1] = 0.5 * (topo_[:, :-1] + topo_[:, 1:])
		topo[:, 0], topo[:, -1] = topo[:, 1], topo[:, -2]
	else:
		topo[1:-1, 1:-1] = 0.25 * (topo_[:-1, :-1] + topo_[1:, :-1] +
								  topo_[:-1, :1]  + topo_[1:, 1:])
		topo[0, 1:-1], topo[-1, 1:-1] = topo[1, 1:-1], topo[-2, 1:-1]
		topo[:, 0], topo[:, -1] = topo[:, 1], topo[:, -2]

	# Plot
	return plot_quiver_xy(xv, yv, vx, vy, topo, scalar, fig, ax, **kwargs)


def plot_quiver_xy(x, y, vx, vy, topography, scalar, fig, ax, **kwargs):
	"""
	Generate the quiver plot of a gridded vector field at a cross-section
	parallel to the :math:`xy`-plane.

	Parameters
	----------
	x : array_like
		2-D :class:`numpy.ndarray` representing the :math:`x`-coordinates
		of the grid points.
	y : array_like
		2-D :class:`numpy.ndarray` representing the :math:`y`-coordinates
		of the grid points.
	vx : array_like
		2-D :class:`numpy.ndarray` representing the :math:`x`-component
		of the field to plot.
	vy : array_like
		2-D :class:`numpy.ndarray` representing the :math:`y`-component
		of the field to plot.
	topography : array_like
		2-D :class:`numpy.ndarray` representing the underlying topography height.
	scalar : array_like
		:class:`numpy.ndarray` representing a scalar field associated with the vector field.
		The arrows will be colored based on the associated scalar value. 
		If :obj:`None`, the arrows will be colored based on their magnitude.
    fig : figure
        A :class:`matplotlib.pyplot.figure`.
    ax : axes
        An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Default is 12.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_step : int
		Maximum distance between the :math:`x`-index of a drawn point, and the
		:math:`x`-index of any of its neighbours. Default is 2, i.e., only half
		of the points will be drawn.
	y_factor : float
		Scaling factor for the :math:`y`-axis. Default is 1.
	y_step : int
		Maximum distance between the :math:`y`-index of a drawn point, and the
		:math:`y`-index of any of its neighbours. Default is 2, i.e., only half
		of the points will be drawn.
	scalar_bias : float
		Bias for the scalar field, so that the arrows will be colored based on
		:obj:`scalar - scalar_bias`. Default is 0.
	scalar_factor : float
		Scaling factor for the scalar field, so that the arrows will be colored based on 
		:obj:`scalar_factor * scalar`. If a bias is specified, then the arrows will be
		colored based on :obj:`scalar_factor * (scalar - scalar_bias)` are drawn.
		Default is 1.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided
		by Matplotlib, as well as the corresponding inverted versions, are available.
		If not specified, no color map will be used, and the arrows will draw black.
	cbar_on : bool
		:obj:`True` to show the color bar, :obj:`False` otherwise. Default is :obj:`True`.
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
	cbar_title : str
		Title for the color bar. Default is an empty string.
	cbar_orientation : str
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the color bar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the color bar. If no indices are given,
		only the current axes are resized.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 12)
	x_factor         = kwargs.get('x_factor', 1.)
	x_step           = kwargs.get('x_step', 2)
	y_factor         = kwargs.get('y_factor', 1.)
	y_step           = kwargs.get('y_step', 2)
	scalar_bias		 = kwargs.get('scalar_bias', 0.)
	scalar_factor    = kwargs.get('scalar_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', None)
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

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x *= x_factor
	y *= y_factor
	if scalar is not None:
		scalar -= scalar_bias
		scalar *= scalar_factor

	# Draw topography isolevels
	plt.contour(x, y, topography, colors='gray')

	if cmap_name is not None:
		# Create color bar for colormap
		if scalar is None:
			scalar = np.sqrt(vx ** 2 + vy ** 2)
		scalar_min, scalar_max = np.amin(scalar), np.amax(scalar)
		if cbar_center is None or not (lt(scalar_min, cbar_center) and lt(cbar_center, scalar_max)):
			cbar_lb, cbar_ub = scalar_min, scalar_max
		else:
			half_width = max(cbar_center-scalar_min, scalar_max-cbar_center) \
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
	else:
		cm = None

	# Generate quiver-plot
	if cm is None:
		plt.quiver(x[::x_step, ::y_step], y[::x_step, ::y_step],
					   vx[::x_step, ::y_step], vy[::x_step, ::y_step])
	else:	
		surf = plt.quiver(x[::x_step, ::y_step], y[::x_step, ::y_step],
					   	  vx[::x_step, ::y_step], vy[::x_step, ::y_step],
				   	   	  scalar[::x_step, ::y_step], cmap=cm)

	# Set the color bar
	if cm is not None and cbar_on:
		plot_utils.set_colorbar(fig, surf, color_levels,
								cbar_ticks_step=cbar_ticks_step,
								cbar_ticks_pos=cbar_ticks_pos, cbar_title=cbar_title,
								cbar_x_label=cbar_x_label, cbar_y_label=cbar_y_label,
								cbar_orientation=cbar_orientation, cbar_ax=cbar_ax)

	# Bring axes and field back to original units
	x      /= x_factor
	y 	   /= y_factor
	if scalar is not None:
		scalar /= scalar_factor
		scalar += scalar_bias

	return fig, ax
