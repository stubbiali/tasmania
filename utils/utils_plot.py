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
Plotting utilities.
"""
import matplotlib as mpl
import matplotlib.animation as manimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np

import tasmania.utils.utils as utils
from tasmania.utils.utils import smaller_than as lt

#
# Private utilities
#
def _reverse_colormap(cmap, name = None):
	"""
	Reverse a Matplotlib colormap.

	Parameters
	----------
	cmap : obj 
		The :class:`matplotlib.colors.LinearSegmentedColormap` to invert.
	name : `str`, optional 
		The name of the reversed colormap. By default, this is obtained by appending '_r' 
		to the name of the input colormap.

	Return
	------
	obj :
		The reversed :class:`matplotlib.colors.LinearSegmentedColormap`.

	References
	----------
	https://stackoverflow.com/questions/3279560/invert-colormap-in-matplotlib.
	"""
	keys = []
	reverse = []

	for key in cmap._segmentdata:
		# Extract the channel
		keys.append(key)
		channel = cmap._segmentdata[key]

		# Reverse the channel
		data = []
		for t in channel:
			data.append((1-t[0], t[2], t[1]))
		reverse.append(sorted(data))

	# Set the name for the reversed map
	if name is None:
		name = cmap.name + '_r'

	return LinearSegmentedColormap(name, dict(zip(keys, reverse)))

#
# Plotting utilities
#
def contour_xz(x, z, field, topography, **kwargs):
	"""
	Generate the contour plot of a gridded field at a cross-section parallel to the :math:`xz`-plane.

	Parameters
	----------
	x : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
	z : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`z`-grid.
	field : array_like
		Two-dimensional :class:`numpy.ndarray` representing the field to plot.
	topography : array_like
		One-dimensional :class:`numpy.ndarray` representing the underlying topography.
		
	Keyword arguments
	-----------------
	show : bool
		:obj:`True` if the plot should be showed, :obj:`False` otherwise. Default is :obj:`True`.
	destination : str
		String specifying the path to the location where the plot will be saved. Default is :obj:`None`, 
		meaning that the plot will not be saved. Note that the plot may be saved only if :data:`show` is 
		set to :obj:`False`.
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	z_label : str
		Label for the :math:`z`-axis. Default is 'z'.
	z_factor : float
		Scaling factor for the :math:`z`-axis. Default is 1.
	z_lim : sequence
		Sequence representing the interval of the :math:`z`-axis to visualize. 
		By default, the entire domain is shown.
	field_factor : float
		Scaling factor for the field. Default is 1.
	draw_z_isolines : bool
		:obj:`True` to draw the :math:`z`-isolines, :obj:`False` otherwise. Default is :obj:`True`.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Shortcuts
	ni, nk = field.shape

	# Get keyword arguments
	show             = kwargs.get('show', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	z_label          = kwargs.get('z_label', 'z')
	z_factor         = kwargs.get('z_factor', 1.)
	z_lim			 = kwargs.get('z_lim', None)
	field_factor     = kwargs.get('field_factor', 1.)
	draw_z_isolines  = kwargs.get('draw_z_isolines', True)
	text			 = kwargs.get('text', None)
	text_loc		 = kwargs.get('text_loc', 'upper right')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x          *= x_factor
	z          *= z_factor
	field      *= field_factor
	topography *= z_factor

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Plot the z-isolines
	if draw_z_isolines:
		for k in range(nk):
			ax.plot(x[:, k], z[:, k], color = 'gray', linewidth = 1)

	# Plot the topography
	ax.plot(x[:, -1], topography, color = 'black', linewidth = 1)

	# Plot the field
	surf = plt.contour(x, z, field, colors = 'black')

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = z_label, title = title)
	if x_lim is None:
		ax.set_xlim([x[0,0], x[-1,0]])
	else:
		ax.set_xlim(x_lim)
	if z_lim is not None:
		ax.set_ylim(z_lim)
	
	# Add text
	if text is not None:
		ax.add_artist(AnchoredText(text, loc = text_loc))

	# Show
	fig.tight_layout()
	if show or (destination is None):
		plt.show()
	else:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def contourf_xy(x, y, topography, field, **kwargs):
	"""
	Generate the contourf plot of a field at a cross-section parallel to the :math:`xy`-plane.

	Parameters
	----------
	x : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
	y : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`y`-grid.
	topography : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying topography height.
	field : array_like
		Two-dimensional :class:`numpy.ndarray` representing the field to plot.
		
	Keyword arguments
	-----------------
	show : bool
		:obj:`True` if the plot should be showed, :obj:`False` otherwise. Default is :obj:`True`.
	destination : str
		String specify the path to the location where the plot will be saved. Default is :obj:`None`, 
		meaning that the plot will not be saved. Note that the plot may be saved only if :data:`show` 
		is set to :obj:`False`.
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	y_label : str
		Label for the :math:`y`-axis. Default is 'y'.
	y_factor : float
		Scaling factor for the :math:`y`-axis. Default is 1.
	y_lim : sequence
		Sequence representing the interval of the :math:`y`-axis to visualize. 
		By default, the entire domain is shown.
	field_factor : float
		Scaling factor for the field. Default is 1.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib, 
		as well as the corresponding inverted versions, are available.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e., 
		all ticks are displayed with the corresponding label.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str 
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Get keyword arguments
	show             = kwargs.get('show', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	y_label          = kwargs.get('y_label', 'y')
	y_factor         = kwargs.get('y_factor', 1.)
	y_lim			 = kwargs.get('y_lim', None)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')
	text			 = kwargs.get('text', None)
	text_loc		 = kwargs.get('text_loc', 'upper right')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x *= x_factor
	y *= y_factor
	field *= field_factor

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Draw topography isolevels
	cs = plt.contour(x, y, topography, colors = 'gray')

	# Create color bar for colormap
	field_min, field_max = np.amin(field), np.amax(field)
	if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center - field_min, field_max - cbar_center) if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
	color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

	# Create colormap
	if cmap_name == 'BuRd':
		cm = _reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the field
	plt.contourf(x, y, field, color_scale, cmap = cm)

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = y_label, title = title)
	if x_lim is not None:
		ax.set_xlim(x_lim)
	if y_lim is not None:
		ax.set_ylim(y_lim)
	
	# Add text
	if text is not None:
		ax.add_artist(AnchoredText(text, loc = text_loc))
	
	# Set colorbar
	cb = plt.colorbar(orientation = cbar_orientation)
	cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
	cb.ax.set_title(cbar_title)
	cb.ax.set_xlabel(cbar_x_label)
	cb.ax.set_ylabel(cbar_y_label)

	# Show
	fig.tight_layout()
	if show or (destination is None):
		plt.show()
	else:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def contourf_xz(x, z, field, topography, **kwargs):
	"""
	Generate the contourf plot of a gridded field at a cross-section parallel to the :math:`xz`-plane.

	Parameters
	----------
	x : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
	z : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`z`-grid.
	field : array_like
		Two-dimensional :class:`numpy.ndarray` representing the field to plot.
	topography : array_like
		One-dimensional :class:`numpy.ndarray` representing the underlying topography.
		
	Keyword arguments
	-----------------
	show : bool
		:obj:`True` if the plot should be showed, :obj:`False` otherwise. Default is :obj:`True`.
	destination : str
		String specifying the path to the location where the plot will be saved. Default is :obj:`None`, 
		meaning that the plot will not be saved. Note that the plot may be saved only if :data:`show` 
		is set to :obj:`False`.
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	z_label : str
		Label for the :math:`z`-axis. Default is 'z'.
	z_factor : float
		Scaling factor for the :math:`z`-axis. Default is 1.
	z_lim : sequence
		Sequence representing the interval of the :math:`z`-axis to visualize. 
		By default, the entire domain is shown.
	field_factor : float
		Scaling factor for the field. Default is 1.
	draw_z_isolines : bool
		:obj:`True` to draw the :math:`z`-isolines, :obj:`False` otherwise. Default is :obj:`True`.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib, 
		as well as the corresponding inverted versions, are available.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e., 
		all ticks are displayed with the corresponding label.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str 
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Shortcuts
	ni, nk = field.shape

	# Get keyword arguments
	show             = kwargs.get('show', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	z_label          = kwargs.get('z_label', 'z')
	z_factor         = kwargs.get('z_factor', 1.)
	z_lim			 = kwargs.get('z_lim', None)
	field_factor     = kwargs.get('field_factor', 1.)
	draw_z_isolines  = kwargs.get('draw_z_isolines', True)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')
	text			 = kwargs.get('text', None)
	text_loc		 = kwargs.get('text_loc', 'upper right')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x          *= x_factor
	z          *= z_factor
	field      *= field_factor
	topography *= z_factor

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Plot the z-isolines
	if draw_z_isolines:
		for k in range(nk):
			ax.plot(x[:, k], z[:, k], color = 'gray', linewidth = 1)

	# Plot the topography
	ax.plot(x[:, -1], topography, color = 'black', linewidth = 1)

	# Determine color scale for colormap
	field_min, field_max = np.amin(field), np.amax(field)
	if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center - field_min, field_max - cbar_center) if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
	color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

	# Create colormap
	if cmap_name == 'BuRd':
		cm = _reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the field
	surf = plt.contourf(x, z, field, color_scale, cmap = cm)

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = z_label)
	ax.set_title(title, loc = 'left', fontsize = fontsize - 1)
	if x_lim is None:
		ax.set_xlim([x[0,0], x[-1,0]])
	else:
		ax.set_xlim(x_lim)
	if z_lim is not None:
		ax.set_ylim(z_lim)
	
	# Set colorbar
	cb = plt.colorbar(orientation = cbar_orientation)
	cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
	cb.ax.set_title(cbar_title)
	cb.ax.set_xlabel(cbar_x_label)
	cb.ax.set_ylabel(cbar_y_label)

	# Add text
	if text is not None:
		ax.add_artist(AnchoredText(text, loc = text_loc))

	# Show
	fig.tight_layout()
	if show:
		plt.show()
	elif destination is not None:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def quiver_xy(x, y, topography, vx, vy, scalar = None, **kwargs):
	"""
	Generate the quiver plot of a gridded vectorial field at a cross-section parallel to the :math:`xy`-plane.

	Parameters
	----------
	x : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
	y : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`y`-grid.
	topography : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying topography height.
	vx : array_like
		:class:`numpy.ndarray` representing the :math:`x`-component of the field to plot.
	vy : array_like
		:class:`numpy.ndarray` representing the :math:`y`-component of the field to plot.
	scalar : `array_like`, optional
		:class:`numpy.ndarray` representing a scalar field associated with the vectorial field.
		The arrows will be colored based on the associated scalar value. 
		If not specified, the arrows will be colored based on their magnitude.
		
	Keyword arguments
	-----------------
	show : bool
		:obj:`True` if the plot should be showed, :obj:`False` otherwise. Default is :obj:`True`.
	destination : str
		String specify the path to the location where the plot will be saved. Default is :obj:`None`, 
		meaning that the plot will not be saved. Note that the plot may be saved only if :data:`show` 
		is set to :obj:`False`.
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	x_step : int
		Maximum distance between the :math:`x`-index of a drawn point, and the :math:`x`-index of any 
		of its neighbours. Default is 2, i.e., only half of the points will be drawn.
	y_label : str
		Label for the :math:`y`-axis. Default is 'y'.
	y_factor : float
		Scaling factor for the :math:`y`-axis. Default is 1.
	y_lim : sequence
		Sequence representing the interval of the :math:`y`-axis to visualize. 
		By default, the entire domain is shown.
	y_step : int
		Maximum distance between the :math:`y`-index of a drawn point, and the :math:`y`-index of any 
		of its neighbours. Default is 2, i.e., only half of the points will be drawn.
	field_factor : float
		Scaling factor for the field. Default is 1.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib, 
		as well as the corresponding inverted versions, are available. If not specified, no color map 
		will be used, and the arrows will draw black.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e., 
		all ticks are displayed with the corresponding label.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str 
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Shortcuts
	nx, ny = grid.nx, grid.ny
	ni, nj = scalar.shape

	# Get keyword arguments
	show            = kwargs.get('show', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	x_step           = kwargs.get('x_step', 2)
	y_label          = kwargs.get('y_label', 'y')
	y_factor         = kwargs.get('y_factor', 1.)
	y_lim			 = kwargs.get('y_lim', None)
	y_step           = kwargs.get('y_step', 2)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', None)
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x *= x_factor
	y *= y_factor
	scalar *= field_factor

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Draw topography isolevels
	plt.contour(x, y, topography, colors = 'gray')

	if cmap_name is not None:
		# Create color bar for colormap
		if scalar is None:
			scalar = np.sqrt(vx ** 2 + vy ** 2)
		scalar_min, scalar_max = np.amin(scalar), np.amax(scalar)
		if cbar_center is None or not (lt(scalar_min, cbar_center) and lt(cbar_center, scalar_max)):
			cbar_lb, cbar_ub = scalar_min, scalar_max
		else:
			half_width = max(cbar_center - scalar_min, scalar_max - cbar_center) if cbar_half_width is None else cbar_half_width
			cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
		color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

		# Create colormap
		if cmap_name == 'BuRd':
			cm = _reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
		else:
			cm = plt.get_cmap(cmap_name)

	# Generate quiver-plot
	if cmap_name is None:
		q = plt.quiver(x[::x_step, ::y_step], y[::x_step, ::y_step], vx[::x_step, ::y_step], vy[::x_step, ::y_step]) 
	else:	
		q = plt.quiver(x[::x_step, ::y_step], y[::x_step, ::y_step], vx[::x_step, ::y_step], vy[::x_step, ::y_step], 
				   	   scalar[::x_step, ::y_step], cmap = cm)

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = y_label, title = title)
	if x_lim is not None:
		ax.set_xlim(x_lim)
	if y_lim is not None:
		ax.set_ylim(y_lim)
	
	# Set colorbar
	if cmap_name is not None:
		cb = plt.colorbar(orientation = cbar_orientation)
		cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
		cb.ax.set_title(cbar_title)
		cb.ax.set_xlabel(cbar_x_label)
		cb.ax.set_ylabel(cbar_y_label)

	# Show
	fig.tight_layout()
	if show or (destination is None):
		plt.show()
	else:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def quiver_xz(x, z, topography, vx, vz, scalar = None, **kwargs):
	"""
	Generate the quiver plot of a gridded vectorial field at a cross-section parallel to the :math:`xz`-plane.

	Parameters
	----------
	x : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
	z : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`z`-grid.
	topography : array_like
		One-dimensional :class:`numpy.ndarray` representing the underlying topography height.
	vx : array_like
		:class:`numpy.ndarray` representing the :math:`x`-component of the field to plot.
	vz : array_like
		:class:`numpy.ndarray` representing the :math:`z`-component of the field to plot.
	scalar : `array_like`, optional
		:class:`numpy.ndarray` representing a scalar field associated with the vectorial field.
		The arrows will be colored based on the associated scalar value. 
		If not specified, the arrows will be colored based on their magnitude.
		
	Keyword arguments
	-----------------
	show : bool
		:obj:`True` if the plot should be showed, :obj:`False` otherwise. Default is :obj:`True`.
	destination : str
		String specify the path to the location where the plot will be saved. Default is :obj:`None`, 
		meaning that the plot will not be saved. Note that the plot may be saved only if :data:`show` 
		is set to :obj:`False`.
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	x_step : int
		Maximum distance between the :math:`x`-index of a drawn point, and the :math:`x`-index of any 
		of its neighbours. Default is 2, i.e., only half of the points will be drawn.
	z_label : str
		Label for the :math:`z`-axis. Default is 'z'.
	z_factor : float
		Scaling factor for the :math:`z`-axis. Default is 1.
	z_lim : sequence
		Sequence representing the interval of the :math:`z`-axis to visualize. 
		By default, the entire domain is shown.
	z_step : int
		Maximum distance between the :math:`z`-index of a drawn point, and the :math:`z`-index of any 
		of its neighbours. Default is 2, i.e., only half of the points will be drawn.
	field_factor : float
		Scaling factor for the field. Default is 1.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib, 
		as well as the corresponding inverted versions, are available. If not specified, no color map 
		will be used, and the arrows will draw black.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e., 
		all ticks are displayed with the corresponding label.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str 
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Get keyword arguments
	show             = kwargs.get('show', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-quiver')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	x_step           = kwargs.get('x_step', 2)
	z_label          = kwargs.get('z_label', 'y')
	z_factor         = kwargs.get('z_factor', 1.)
	z_lim			 = kwargs.get('z_lim', None)
	z_step           = kwargs.get('z_step', 2)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', None)
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')

	# Shortcuts
	ni, nj = scalar.shape

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x          *= x_factor
	z          *= z_factor
	topography *= z_factor
	scalar     *= field_factor

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Draw topography
	plt.plot(x[:,-1], topography, color = 'black')

	if cmap_name is not None:
		# Create color bar for colormap
		if scalar is None:
			scalar = np.sqrt(vx ** 2 + vz ** 2)
		scalar_min, scalar_max = np.nanmin(scalar), np.nanmax(scalar)
		if cbar_center is None or not (lt(scalar_min, cbar_center) and lt(cbar_center, scalar_max)):
			cbar_lb, cbar_ub = scalar_min, scalar_max
		else:
			half_width = max(cbar_center - scalar_min, scalar_max - cbar_center) if cbar_half_width is None else cbar_half_width
			cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
		color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

		# Create colormap
		if cmap_name == 'BuRd':
			cm = _reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
		else:
			cm = plt.get_cmap(cmap_name)

	# Generate quiver-plot
	if cmap_name is None:
		q = plt.quiver(x[::x_step, ::z_step], z[::x_step, ::z_step], vx[::x_step, ::z_step], vz[::x_step, ::z_step]) 
	else:	
		q = plt.quiver(x[::x_step, ::z_step], z[::x_step, ::z_step], vx[::x_step, ::z_step], vz[::x_step, ::z_step], 
				   	   scalar[::x_step, ::z_step], cmap = cm)

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = z_label)
	ax.set_title(title, loc = 'left', fontsize = fontsize - 1)
	if x_lim is not None:
		ax.set_xlim(x_lim)
	if z_lim is not None:
		ax.set_ylim(z_lim)
	
	# Set colorbar
	if cmap_name is not None:
		cb = plt.colorbar(orientation = cbar_orientation)
		cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
		cb.ax.set_title(cbar_title)
		cb.ax.set_xlabel(cbar_x_label)
		cb.ax.set_ylabel(cbar_y_label)

	# Show
	fig.tight_layout()
	if show or (destination is None):
		plt.show()
	else:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

def streamplot_xz(x, z, u, w, color, topography, **kwargs):
	"""
	Generate the streamplot of a gridded vector field at a cross-section parallel to the :math:`xz`-plane.

	Parameters
	----------
	x : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
	z : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`z`-grid.
	u : array_like
		Two-dimensional :class:`numpy.ndarray` representing the :math:`x`-velocity.
	w : array_like
		Two-dimensional :class:`numpy.ndarray` representing the :math:`z`-velocity.
	color : array_like
		Two-dimensional :class:`numpy.ndarray` representing the streamlines color.
	topography : array_like
		One-dimensional :class:`numpy.ndarray` representing the underlying topography.
		
	Keyword arguments
	-----------------
	show : bool
		:obj:`True` if the plot should be showed, :obj:`False` otherwise. Default is :obj:`True`.
	destination : str
		String specifying the path to the location where the plot will be saved. Default is :obj:`None`, 
		meaning that the plot will not be saved. Note that the plot may be saved only if :data:`show` 
		is set to :obj:`False`.
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	z_label : str
		Label for the :math:`z`-axis. Default is 'z'.
	z_factor : float
		Scaling factor for the :math:`z`-axis. Default is 1.
	z_lim : sequence
		Sequence representing the interval of the :math:`z`-axis to visualize. 
		By default, the entire domain is shown.
	u_factor : float
		Scaling factor for the :math:`x`-velocity. Default is 1.
	w_factor : float
		Scaling factor for the :math:`z`-velocity. Default is 1.
	color_factor : float
		Scaling factor for the color field. Default is 1.
	draw_z_isolines : bool
		:obj:`True` to draw the :math:`z`-isolines, :obj:`False` otherwise. Default is :obj:`False`.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib, 
		as well as the corresponding inverted versions, are available.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e., 
		all ticks are displayed with the corresponding label.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str 
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Shortcuts
	nk, ni = u.shape

	# Get keyword arguments
	show             = kwargs.get('show', True)
	destination      = kwargs.get('destination', None)
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '$xz$-contourf')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	z_label          = kwargs.get('z_label', 'z')
	z_factor         = kwargs.get('z_factor', 1.)
	z_lim			 = kwargs.get('z_lim', None)
	u_factor     	 = kwargs.get('u_factor', 1.)
	w_factor     	 = kwargs.get('w_factor', 1.)
	color_factor     = kwargs.get('color_factor', 1.)
	draw_z_isolines  = kwargs.get('draw_z_isolines', True)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')
	text			 = kwargs.get('text', None)
	text_loc		 = kwargs.get('text_loc', 'upper right')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x          *= x_factor
	z          *= z_factor
	u          *= u_factor
	w          *= w_factor
	color      *= color_factor
	topography *= z_factor

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Plot the topography
	ax.plot(x, topography, color = 'black', linewidth = 1)

	# Determine color scale for colormap
	color_min, color_max = np.nanmin(color), np.nanmax(color)
	if cbar_center is None or not (lt(color_min, cbar_center) and lt(cbar_center, color_max)):
		cbar_lb, cbar_ub = color_min, color_max
	else:
		half_width = max(cbar_center - color_min, color_max - cbar_center) if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
	color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

	# Create colormap
	if cmap_name == 'BuRd':
		cm = _reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the streamlines
	strm = plt.streamplot(x, z, u, w, color = color, cmap = cm)

	# Set plot settings
	ax.set(xlabel = x_label, ylabel = z_label)
	ax.set_title(title, loc = 'left', fontsize = fontsize - 1)
	if x_lim is None:
		ax.set_xlim([x[0], x[-1]])
	else:
		ax.set_xlim(x_lim)
	if z_lim is not None:
		ax.set_ylim(z_lim)
	
	# Set colorbar
	cb = plt.colorbar(strm.lines, orientation = cbar_orientation)
	cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
	cb.ax.set_title(cbar_title)
	cb.ax.set_xlabel(cbar_x_label)
	cb.ax.set_ylabel(cbar_y_label)

	# Add text
	if text is not None:
		ax.add_artist(AnchoredText(text, loc = text_loc))

	# Show
	fig.tight_layout()
	if show:
		plt.show()
	elif destination is not None:
		plt.savefig(destination + '.eps', format = 'eps', dpi = 1000)

#
# Animation utilities
#
def animation_contourf_xz(destination, time, x, z, field, topography, **kwargs):
	"""
	Generate an animation showing the time evolution of the contourfs of a field at a cross-section 
	parallel to the :math:`xz`-plane.

	Parameters
	----------
	destination : str
		String specifying the path to the location where the movie will be saved. 
		Note that the string should include the extension as well.
	time : array_like
		Array of :class:`datetime.datetime`\s representing the time instants of the frames.
	x : array_like
		Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
		This is assumed to be time-independent.
	z : array_like
		Three-dimensional :class:`numpy.ndarray` representing the underlying :math:`z`-grid.
		It is assumed that:
		
		* the first array axis represents :math:`x`;
		* the second array axis represents the vertical coordinate;
		* the third array axis represents the time.

	field : array_like
		Three-dimensional :class:`numpy.ndarray` representing the field to plot.
		It is assumed that:
		
		* the first array axis represents :math:`x`;
		* the second array axis represents the vertical coordinate;
		* the third array axis represents the time.

	topography : `array_like`, optional
		Two-dimensional :class:`numpy.ndarray` representing the underlying topography.
		It is assumed that:
		
		* the first array axis represents :math:`x`;
		* the second array axis represents the time.
		
	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	z_label : str
		Label for the :math:`z`-axis. Default is 'z'.
	z_factor : float
		Scaling factor for the :math:`z`-axis. Default is 1.
	z_lim : sequence
		Sequence representing the interval of the :math:`z`-axis to visualize. 
		By default, the entire domain is shown.
	field_factor : float
		Scaling factor for the field. Default is 1.
	draw_z_isolines : bool
		:obj:`True` to draw the :math:`z`-isolines, :obj:`False` otherwise. Default is :obj:`True`.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib, 
		as well as the corresponding inverted versions, are available.
	cbar_levels : int
		Number of levels for the color bar. Default is 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e., 
		all ticks are displayed with the corresponding label.
	cbar_center : float
		Center of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field over time.
	cbar_half_width : float
		Half-width of the range covered by the color bar. By default, the color bar covers the spectrum 
		ranging from the minimum to the maximum assumed by the field over time.
	cbar_x_label : str
		Label for the horizontal axis of the color bar. Default is an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Default is an empty string.
	cbar_orientation : str 
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	fps : int
		Frames per second. Default is 15.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Shortcuts
	ni, nk, nt = field.shape

	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	z_label          = kwargs.get('z_label', 'z')
	z_factor         = kwargs.get('z_factor', 1.)
	z_lim			 = kwargs.get('z_lim', None)
	field_factor     = kwargs.get('field_factor', 1.)
	draw_z_isolines  = kwargs.get('draw_z_isolines', True)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
	cbar_levels      = kwargs.get('cbar_levels', 14)
	cbar_ticks_step  = kwargs.get('cbar_ticks_step', 1)
	cbar_center      = kwargs.get('cbar_center', None)
	cbar_half_width  = kwargs.get('cbar_half_width', None)
	cbar_x_label     = kwargs.get('cbar_x_label', '')
	cbar_y_label     = kwargs.get('cbar_y_label', '')
	cbar_title       = kwargs.get('cbar_title', '')
	cbar_orientation = kwargs.get('cbar_orientation', 'vertical')
	fps				 = kwargs.get('fps', 15)
	text			 = kwargs.get('text', None)
	text_loc		 = kwargs.get('text_loc', 'upper right')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Instantiate writer class
	ffmpeg_writer = manimation.writers['ffmpeg']
	metadata = {'title': ''}
	writer = ffmpeg_writer(fps = fps, metadata = metadata)

	with writer.saving(fig, destination, nt):
		# Rescale the axes and the field for visualization purposes
		x          *= x_factor
		z          *= z_factor
		field      *= field_factor
		topography *= z_factor

		# Create the color bar for the colormap
		field_min, field_max = np.amin(field), np.amax(field)
		if cbar_center is None or not (lt(field_min, cbar_center) and lt(cbar_center, field_max)):
			cbar_lb, cbar_ub = field_min, field_max
		else:
			half_width = max(cbar_center - field_min, field_max - cbar_center) if cbar_half_width is None \
						 else cbar_half_width
			cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width 
		color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint = True)

		# Create the colormap
		if cmap_name == 'BuRd':
			cm = _reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
		else:
			cm = plt.get_cmap(cmap_name)

		for n in range(nt):
			# Clean the canvas
			ax.cla()

			# Plot the z-isolines
			if draw_z_isolines:
				for k in range(nk):
					ax.plot(x[:, k], z[:, k, n], color = 'gray', linewidth = 1)

			# Plot the topography
			ax.plot(x[:, -1], topography[:, n], color = 'black', linewidth = 1)

			# Plot the field
			surf = plt.contourf(x, z[:, :, n], field[:, :, n], color_scale, cmap = cm)
		
			# Set plot settings
			ax.set(xlabel = x_label, ylabel = z_label)
			if x_lim is None:
				ax.set_xlim([x[0,0], x[-1,0]])
			else:
				ax.set_xlim(x_lim)
			if z_lim is not None:
				ax.set_ylim(z_lim)

			if n == 0:
				# Set colorbar
				cb = plt.colorbar(orientation = cbar_orientation)
				cb.set_ticks(0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step])
				if cbar_title is not None:
					cb.ax.set_title(cbar_title)
				if cbar_x_label is not None:
					cb.ax.set_xlabel(cbar_x_label)
				if cbar_y_label is not None:
					cb.ax.set_ylabel(cbar_y_label)

			# Add text
			if text is not None:
				ax.add_artist(AnchoredText(text, loc = text_loc))

			# Add time
			plt.title(title, loc = 'left', fontsize = fontsize - 1)
			plt.title(str(utils.convert_datetime64_to_datetime(time[n]) - utils.convert_datetime64_to_datetime(time[0])), 
				      loc = 'right', fontsize = fontsize - 1)

			# Let the writer grab the frame
			writer.grab_frame()

def animation_profile_x(time, x, field, destination, **kwargs):
	"""
	Generate an animation showing the time evolution of a field along a cross line orthogonal 
	to the :math:`yz`-plane.

	Parameters
	----------
	time : array_like 
		Array of :class:`datetime.datetime`\s representing the time instants of the frames.
	x : array_like
		One-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
	field : array_like
		Two-dimensional :class:`numpy.ndarray` representing the field to plot.
		It is assumed that:
		
		* the first array axis represents :math:`x`;
		* the second array axis represents the time.

	destination : str
		String specifying the path to the location where the movie will be saved. 
		Note that the string should include the extension as well.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	y_label : str
		Label for the :math:`y`-axis. Default is 'y'.
	y_factor : float
		Scaling factor for the field. Default is 1.
	y_lim : sequence
		Sequence representing the interval of the :math:`y`-axis to visualize. 
		By default, the entire domain is shown.
	color : str
		String specifying the color line. Default is 'blue'.
	linewidth : float
		The linewidth. Default is 1.
	grid_on : bool
		:obj:`True` to draw the grid, :obj:`False` otherwise. Default is :obj:`True`.
	fps : int
		Frames per second. Default is 15.
	text : str
		Text to be added to the figure as anchored text. By default, no extra text is shown.
	text_loc : str
		String specifying the location where the text box should be placed. Default is 'upper right'; 
		please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
	"""
	# Shortcuts
	ni, nt = field.shape

	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 12)
	figsize			 = kwargs.get('figsize', [8,8])
	title            = kwargs.get('title', '')
	x_label          = kwargs.get('x_label', 'x')
	x_factor         = kwargs.get('x_factor', 1.)
	x_lim			 = kwargs.get('x_lim', None)
	y_label          = kwargs.get('y_label', 'y')
	y_factor         = kwargs.get('y_factor', 1.)
	y_lim			 = kwargs.get('y_lim', None)
	color			 = kwargs.get('color', 'blue')
	linewidth        = kwargs.get('linewidth', 1.)
	grid_on     	 = kwargs.get('grid_on', True)
	fps				 = kwargs.get('fps', 15)
	text			 = kwargs.get('text', None)
	text_loc		 = kwargs.get('text_loc', 'upper right')

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Instantiate writer class
	ffmpeg_writer = manimation.writers['ffmpeg']
	metadata = {'title': ''}
	writer = ffmpeg_writer(fps = fps, metadata = metadata)

	with writer.saving(fig, destination, nt):
		# Rescale the x-axis and the field for visualization purposes
		x *= x_factor
		field *= y_factor

		for n in range(nt):
			# Clean the canvas
			ax.cla()

			# Plot the field
			plt.plot(x, field[:,n], color = color, linewidth = linewidth)
		
			# Set plot settings
			ax.set(xlabel = x_label, ylabel = y_label)
			if x_lim is not None:
				ax.set_xlim(x_lim)
			else:
				ax.set_xlim([x[0], x[-1]])
			if y_lim is not None:
				ax.set_ylim(y_lim)
			else:
				ax.set_ylim([field.min(), field.max()])
			if grid_on:
				ax.grid()

			# Add text
			if text is not None:
				ax.add_artist(AnchoredText(text, loc = text_loc))

			# Add time
			plt.title(title, loc = 'left', fontsize = fontsize - 1)
			plt.title(str(utils.convert_datetime64_to_datetime(time[n]) - utils.convert_datetime64_to_datetime(time[0])), 
					  loc = 'right', fontsize = fontsize - 1)

			# Let the writer grab the frame
			writer.grab_frame()

def animation_profile_x_comparison(time, x, field, destination, **kwargs):
	"""
	Generate an animation showing the time evolution of one or more fields along a cross line orthogonal 
	to the :math:`yz`-plane.

	Parameters
	----------
	time : array_like 
		Array of :class:`datetime.datetime`\s representing the time instants of the frames.
	x : list
		Two-dimensional :class:`numpy.ndarray` storing the :math:`x`-grids underlying each field.
		It is assumed that:

		* the fields are concatenated along the first array axis;
		* the second array axis represents :math:`x`.

	field : array_like
		Three-dimensional :class:`numpy.ndarray` storing the fields to plot.
		It is assumed that:
		
		* the fields are concatenated along the first array axis;
		* the second array axis represents :math:`x`;
		* the third array axis represents the time.

	destination : str
		String specifying the path to the location where the movie will be saved. 
		Note that the string should include the extension as well.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Default is 12.
	figsize : sequence
		Sequence representing the figure size. Default is [8,8].
	title : str
		The figure title. Default is an empty string.
	x_label : str
		Label for the :math:`x`-axis. Default is 'x'.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Default is 1.
	x_lim : sequence
		Sequence representing the interval of the :math:`x`-axis to visualize. 
		By default, the entire domain is shown.
	y_label : str
		Label for the :math:`y`-axis. Default is 'y'.
	y_lim : sequence
		Sequence representing the interval of the :math:`y`-axis to visualize. 
		By default, the entire domain is shown.
	field_factor : list
		List storing the scaling factor for each field. By default, all scaling factors are assumed to be 1.
	color : list
		List of strings specifying the line color for each field. The default sequence of colors is: 'blue', 'red', 'green', 'black'.
	linestyle : list
		List of strings specifying the line style for each field. The default line style is '-'.
	linewidth : list
		List of floats representing the line width for each field. The default line width is 1.
	grid_on : bool
		:obj:`True` to draw the grid, :obj:`False` otherwise. Default is :obj:`True`.
	fps : int
		Frames per second. Default is 15.
	legend : list
		List gathering the legend entries for each field. Default is 'field1', 'field2', etc.
	legend_loc : str
		String specifying the location where the legend box should be placed. Default is 'best'; 
		please see :func:`matplotlib.pyplot.legend` for all the available options.
	"""
	# Shortcuts
	nf = len(x)
	nt = field[0].shape[1]

	# Get keyword arguments
	fontsize    	= kwargs.get('fontsize', 12)
	figsize			= kwargs.get('figsize', [8,8])
	title       	= kwargs.get('title', '')
	x_label     	= kwargs.get('x_label', 'x')
	x_factor    	= kwargs.get('x_factor', 1.)
	x_lim			= kwargs.get('x_lim', None)
	y_label     	= kwargs.get('y_label', 'y')
	y_lim			= kwargs.get('y_lim', None)
	field_factor	= kwargs.get('field_factor', [1.] * nf)
	color			= kwargs.get('color', ['blue', 'red', 'green', 'black'])
	linestyle  		= kwargs.get('linestyle', ['-'] * nf)
	linewidth  		= kwargs.get('linewidth', [1.] * nf)
	grid_on     	= kwargs.get('grid_on', True)
	fps				= kwargs.get('fps', 15)
	legend			= kwargs.get('legend', ['field1', 'field2', 'field3', 'field4'])
	legend_loc      = kwargs.get('legend_loc', 'best')

	# Rescale the x-axis and the fields for visualization purposes
	for m in range(nf):
		x[m][:]       *= x_factor
		field[m][:,:] *= field_factor[m]

	# Global settings
	mpl.rcParams['font.size'] = fontsize

	# Instantiate figure and axis objects
	fig, ax = plt.subplots(figsize = figsize)

	# Instantiate writer class
	ffmpeg_writer = manimation.writers['ffmpeg']
	metadata = {'title': ''}
	writer = ffmpeg_writer(fps = fps, metadata = metadata)

	with writer.saving(fig, destination, nt):
		for n in range(nt):
			# Clean the canvas
			ax.cla()

			# Plot the fields
			for m in range(nf):
				plt.plot(x[m][:], field[m][:,n], color = color[m], linestyle = linestyle[m], linewidth = linewidth[m], label = legend[m])
			ax.legend(loc = legend_loc)
		
			# Set axis labels
			ax.set(xlabel = x_label, ylabel = y_label)

			# Draw grid
			if grid_on:
				ax.grid()

			# Set x-axis limit
			if x_lim is not None:
				ax.set_xlim(x_lim)
			else:
				ax.set_xlim([x[0][:].min(), x[0][:].max()])

			# Set y-axis limit
			if y_lim is not None:
				ax.set_ylim(y_lim)
			else:
				ax.set_ylim([field.min(), field.max()])

			# Add title
			plt.title(title, loc = 'left', fontsize = fontsize - 1)
			plt.title(str(utils.convert_datetime64_to_datetime(time[n]) - utils.convert_datetime64_to_datetime(time[0])), 
					  loc = 'right', fontsize = fontsize - 1)

			# Let the writer grab the frame
			writer.grab_frame()
