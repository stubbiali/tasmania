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

from tasmania.namelist import datatype
import tasmania.utils.utils as utils


def make_contour_xz(grid, state, field_to_plot, y_level, fig, ax, **kwargs):
	"""
	Given an input model state, generate the contour plot of a specified field at
	a cross-section parallel to the :math:`xz`-plane.

	Parameters
	----------
	grid : grid
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str 
		String specifying the field to plot. This might be:

		* the name of a variable stored in the input model state;
		* 'vertical_velocity', for the vertical velocity of a two-dimensional, \
			steady-state, isentropic flow; the model state should contain the \
			following variables:

			- `air_isentropic_density`;
			- `x_momentum_isentropic`;
			- `height_on_interface_levels`.

	y_level : int 
		:math:`y`-index identifying the cross-section.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.
	**kwargs :
		Keyword arguments specifying different plotting settings.
		See :func:`~tasmania.plot.contour_xz.plot_contour_xz` for the complete list.

	Returns
	-------
	out_fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	out_ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.

	Raise
	-----
	ValueError :
		If neither the grid, nor the model state, contains `height` or
		`height_on_interface_levels`.
	"""
	# Extract, compute, or interpolate the field to plot
	if field_to_plot in state.keys():
		var = state[field_to_plot].values[:, y_level, :] 
	elif field_to_plot == 'vertical_velocity':
		assert grid.ny == 1, \
			'The input grid should consist of only one point in the y-direction.'
		assert y_level == 0, \
			'As the grid consists of only one point in the y-direction, y_level must be 0.'

		s, su, h = utils.get_numpy_arrays(state, (slice(0, None), y_level, slice(0, None)),
			'air_isentropic_density', 'x_momentum_isentropic', 'height_on_interface_levels')

		u = su / s
		h_mid  = 0.5 * (h[:, :-1] + h[:, 1:])
		h_mid_at_u_loc_ = 0.5 * (h_mid[:-1, :] + h_mid[1:, :])
		h_mid_at_u_loc  = np.concatenate((h_mid_at_u_loc_[0:1, :], h_mid_at_u_loc_,
										  h_mid_at_u_loc_[-1:, :]), axis=0)

		var = u * (h_mid_at_u_loc[1:, :] - h_mid_at_u_loc[:-1, :]) / grid.dx
	else:
		raise ValueError('Unknown field to plot {}.'.format(field_to_plot))

	# Shortcuts
	nx, nz = grid.nx, grid.nz
	ni, nk = var.shape

	# The underlying x-grid
	x  = grid.x[:] if ni == nx else grid.x_at_u_locations[:]
	xv = np.repeat(x[:, np.newaxis], nk, axis=1)

	# Extract the geometric height at the main or interface levels, and the topography
	try:
		z = utils.get_numpy_arrays(state, (slice(0, None), y_level, slice(0, None)),
							 	   'height_on_interface_levels')
		topo_ = z[:, -1]
	except KeyError:
		try:
			z = grid.height_on_interface_levels.values
			topo_ = z[:, -1]
		except AttributeError:
			try:
				z = utils.get_numpy_arrays(state, (slice(0, None), y_level, slice(0, None)),
									 	   'height')
				topo_ = grid.topography_height
			except KeyError:
				try:
					z = grid.height.values
					topo_ = grid.topography_height
				except AttributeError:
					print("""Neither the grid, nor the state, contains either ''height'' 
                             or ''height_on_interface_levels''.""")

	# Reshape the extracted height of the vertical levels
	if z.shape[1] < nk:
		raise ValueError("""As the field to plot is vertically staggered, 
							''height_on_interface_levels'' is needed.""")
	if z.shape[1] > nk:
		z = 0.5 * (z[:, :-1] + z[:, 1:])

	# The underlying z-grid
	zv = np.zeros((ni, nk), dtype=datatype)
	if ni == nx:
		zv[:, :] = z[:, :]
	else:
		zv[1:-1, :] = 0.5 * (z[:-1, :] + z[1:, :])
		zv[0, :], zv[-1, :] = zv[1, :], zv[-2, :]

	# Possibly reshape the underlying topography
	if ni == nx:
		topo = topo_
	else:
		topo = np.zeros((nx + 1), dtype=datatype)
		topo[1:-1] = 0.5 * (topo_[:-1] + topo_[1:])
		topo[0], topo[-1] = topo[1], topo[-2]

	# Plot and return figure object
	return plot_contour_xz(xv, zv, var, topo, fig, ax, **kwargs)


def plot_contour_xz(x, z, field, topography, fig, ax, **kwargs):
	"""
	Generate the contour plot of a gridded field at a cross-section
	parallel to the :math:`xz`-plane.

	Parameters
	----------
	x : array_like
		2-D :class:`numpy.ndarray` representing the :math:`x`-coordinates
		of the grid points.
	z : array_like
		2-D :class:`numpy.ndarray` representing the :math:`z`-coordinates
		of the grid points.
	field : array_like
		2-D :class:`numpy.ndarray` representing the field to plot.
	topography : array_like
		1-D :class:`numpy.ndarray` representing the underlying topography.
	fig : figure
		A :class:`matplotlib.pyplot.figure`.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used in the plot. Default is 16.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	z_factor : float
		Scaling factor for the :math:`z`-axis. Default is 1.
	field_bias : float
		Bias for the field, so that the contour lines for :obj:`field - field_bias`
		are drawn. Default is 0.
	field_factor : float
		Scaling factor for the field, so that the contour lines for
		:obj:`field_factor * field` are drawn. If a bias is specified, then the contour
		lines for :obj:`field_factor * (field - field_bias)` are drawn. Default is 1.
	draw_grid : bool
		:obj:`True` to draw the underlying grid, :obj:`False` otherwise.
		Default is :obj:`True`.

	Returns
	-------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	ax : axes
		The :class:`matplotlib.axes.Axes` enclosing the plot.
	"""
	# Shortcuts
	ni, nk = field.shape

	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 16)
	x_factor         = kwargs.get('x_factor', 1.)
	z_factor         = kwargs.get('z_factor', 1.)
	field_bias	 	 = kwargs.get('field_bias', 0.)
	field_factor	 = kwargs.get('field_factor', 1.)
	draw_grid  		 = kwargs.get('draw_grid', True)

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x          *= x_factor
	z          *= z_factor
	field      -= field_bias
	field	   *= field_factor
	topography *= z_factor

	# Plot the z-isolines
	if draw_grid:
		for k in range(nk):
			ax.plot(x[:, k], z[:, k], color='gray', linewidth=1)

	# Plot the topography
	ax.plot(x[:, -1], topography, color='black', linewidth=1)

	# Plot the field
	plt.contour(x, z, field, colors='black')

	# Bring axes and field back to original units
	x     /= x_factor
	z 	  /= z_factor
	field /= field_factor
	field += field_bias

	return fig, ax