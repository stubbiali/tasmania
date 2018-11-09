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
	get_figure_and_axes
	set_figure_properties
	set_axes_properties
	reverse_colormap
	set_colorbar
	make_lineplot
	make_contour
	make_contourf
	make_quiver
"""
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np

from tasmania.plot.utils import smaller_than as lt, equal_to as eq


text_locations = {
	'upper right'  : 1,
	'upper left'   : 2,
	'lower left'   : 3,
	'lower right'  : 4,
	'right'        : 5,
	'center left'  : 6,
	'center right' : 7,
	'lower center' : 8,
	'upper center' : 9,
	'center'       : 10,
}


def get_figure_and_axes(fig=None, ax=None, default_fig=None,
						nrows=1, ncols=1, index=1, **kwargs):
	"""
	Get a :class:`matplotlib.pyplot.figure` object and a :class:`matplotlib.axes.Axes` 
	object, with the latter embedded in the former. The returned values are determined 
	as follows.

	* If both :obj:`fig` and :obj:`ax` arguments are passed in:

		- if :obj:`ax` is embedded in :obj:`fig`, return :obj:`fig` and :obj:`ax`;
		- otherwise, return the figure which encloses :obj:`ax`, and :obj:`ax` itself.

	* If :obj:`fig` is provided but :obj:`ax` is not:

		- if :obj:`fig` contains some subplots, return :obj:`fig` and the axes of the
			first subplot it contains;
		- otherwise, add a subplot to :obj:`fig` in position (1,1,1) and return :obj:`fig`
			and the subplot axes.

	* If :obj:`ax` is provided but :obj:`fig` is not, return the figure which encloses 
		:obj:`ax`, and :obj:`ax` itself.

	* If neither :obj:`fig` nor :obj:`ax` are passed in:

		- if :obj:`default_fig` is not given, instantiate a new pair of figure and axes;
		- if :obj:`default_fig` is provided and it  contains some subplots, return 
			:obj:`default_fig` and the axes of the first subplot it contains;
		- if :obj:`default_fig` is provided but is does not contain any subplot, add a 
			subplot to :obj:`default_fig` in position (1,1,1) and return :obj:`default_fig`
			and the subplot axes.

	Parameters
	----------
	fig : `figure`, optional
		A :class:`matplotlib.pyplot.figure` object.
	ax : `axes`, optional
		An instance of :class:`matplotlib.axes.Axes`.
	default_fig : `figure`, optional
		A :class:`matplotlib.pyplot.figure` object.
	nrows : `int`, optional
		Number of rows of the subplot grid. Defaults to 1.
		Only effective if :obj:`ax` is not given.
	ncols : `int`, optional
		Number of columns of the subplot grid. Defaults to 1.
		Only effective if :obj:`ax` is not given.
	index : `int`, optional
		Axes' index in the subplot grid. Defaults to 1.
		Only effective if :obj:`ax` is not given.

	Keyword arguments
	-----------------
	figsize : `tuple`, optional
		The size which the output figure should have. Defaults to (7, 7_
		This argument is effective only if the figure is created within the function.
	fontsize : `int`, optional
		Font size for the output figure and axes. Defaults to 12.
		This argument is effective only if the figure is created within the function.
	projection : `str`, optional
		The axes projection. Defaults to :obj:`None`.
		This argument is effective only if the figure is created within the function.

	Returns
	-------
	out_fig : figure
		A :class:`matplotlib.pyplot.figure` object.
	out_ax : axes
		An instance of :class:`matplotlib.axes.Axes`.
	"""
	figsize    = kwargs.get('figsize', (7, 7))
	fontsize   = kwargs.get('fontsize', 12)
	projection = kwargs.get('projection', None)

	rcParams['font.size'] = fontsize

	if (fig is not None) and (ax is not None):
		try:
			if ax not in fig.get_axes():
				import warnings
				warnings.warn("""Input axes do not belong to the input figure,
								 so consider the figure which the axes belong to.""",
							  RuntimeWarning)
				out_fig, out_ax = ax.get_figure(), ax
			else:
				out_fig, out_ax = fig, ax
		except AttributeError:
			import warnings
			warnings.warn("""Input argument ''fig'' does not seem to be a figure, 
							 so consider the figure the axes belong to.""", RuntimeWarning)
			out_fig, out_ax = ax.get_figure(), ax

	if (fig is not None) and (ax is None):
		try:
			out_fig = fig
			out_ax = fig.get_axes()[index] if len(fig.get_axes()) > index \
					 else fig.add_subplot(nrows, ncols, index, projection=projection)
		except AttributeError:
			import warnings
			warnings.warn("""Input argument ''fig'' does not seem to be a figure, 
							 hence a proper figure object is created.""", RuntimeWarning)
			out_fig = plt.figure(figsize=figsize)
			out_ax = out_fig.add_subplot(nrows, ncols, index, projection=projection)

	if (fig is None) and (ax is not None):
		out_fig, out_ax = ax.get_figure(), ax

	if (fig is None) and (ax is None):
		if default_fig is None:
			out_fig = plt.figure(figsize=figsize)
			out_ax = out_fig.add_subplot(nrows, ncols, index, projection=projection)
		else:
			try:
				out_fig = default_fig
				out_ax = out_fig.get_axes()[index] if len(out_fig.get_axes()) > index \
						 else out_fig.add_subplot(nrows, ncols, index, projection=projection)
			except AttributeError:
				import warnings
				warnings.warn("""The argument ''default_fig'' does not actually seem 
								 to be a figure, hence a proper figure object is created.""",
							  RuntimeWarning)
				out_fig = plt.figure(figsize=figsize)
				out_ax = out_fig.add_subplot(nrows, ncols, index, projection=projection)

	return out_fig, out_ax


def set_figure_properties(fig, **kwargs):
	"""
	An utility to ease the configuration of a :class:`~matplotlib.pyplot.figure`.

	Parameters
	----------
	fig : figure
		A :class:`~matplotlib.pyplot.figure`.

	Keyword arguments
	-----------------
	fontsize : `int`, optional
		Font size to use for the plot titles, and axes ticks and labels.
		Defaults to 12.
	tight_layout : `bool`, optional
        :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
        Defaults to :obj:`True`.
	suptitle : `str`, optional
		The figure title. Defaults to an empty string.
	"""
	fontsize     = kwargs.get('fontsize', 12)
	tight_layout = kwargs.get('tight_layout', True)
	suptitle     = kwargs.get('suptitle', '')

	rcParams['font.size'] = fontsize

	if tight_layout:
		fig.tight_layout()

	if suptitle != '':
		fig.suptitle(suptitle, fontsize=fontsize+1)


def set_axes_properties(ax, **kwargs):
	"""
	An utility to ease the configuration of an :class:`~matplotlib.axes.Axes` object.

	Parameters
	----------
	ax : axes
		Instance of :class:`matplotlib.axes.Axes` enclosing the plot.

	Keyword arguments
	-----------------
	fontsize : `int`, optional
		Font size to use for the plot titles, and axes ticks and labels.
		Defaults to 12.
	title_center : `str`, optional
		Text to use for the axes center title. Defaults to an empty string.
	title_left : `str`, optional
		Text to use for the axes left title. Defaults to an empty string.
	title_right : `str`, optional
		Text to use for the axes right title. Defaults to an empty string.
	x_label : `str`, optional
		Text to use for the label of the x-axis. Defaults to an empty string.
	x_lim : `tuple`, optional
		Data limits for the x-axis. Defaults to :obj:`None`, i.e., the data limits
		will be left unchanged.
	invert_xaxis : `bool`, optional
		TODO
	x_scale : `str`, optional
		TODO
	x_ticks : `list of float`, optional
		List of x-axis ticks location.
	x_ticklabels : `list of str`, optional
		List of x-axis ticks labels.
	xaxis_minor_ticks_visible : `bool`, optional
		TODO
	xaxis_visible : `bool`, optional
		:obj:`False` to make the x-axis invisible. Defaults to :obj:`True`.
	y_label : `str`, optional
		Text to use for the label of the y-axis. Defaults to an empty string.
	y_lim : `tuple`, optional
		Data limits for the y-axis. Defaults to :obj:`None`, i.e., the data limits
		will be left unchanged.
	invert_yaxis : `bool`, optional
		TODO
	y_scale : `str`, optional
		TODO
	y_ticks : `list of float`, optional
		List of y-axis ticks location.
	y_ticklabels : `list of str`, optional
		List of y-axis ticks labels.
	yaxis_minor_ticks_visible : `bool`, optional
		TODO
	yaxis_visible : `bool`, optional
		:obj:`False` to make the y-axis invisible. Defaults to :obj:`True`.
	z_label : `str`, optional
		Text to use for the label of the z-axis. Defaults to an empty string.
	z_lim : `tuple`, optional
		Data limits for the z-axis. Defaults to :obj:`None`, i.e., the data limits
		will be left unchanged.
	invert_zaxis : `bool`, optional
		TODO
	z_scale : `str`, optional
		TODO
	z_ticks : `list of float`, optional
		List of z-axis ticks location.
	z_ticklabels : `list of str`, optional
		List of z-axis ticks labels.
	zaxis_minor_ticks_visible : `bool`, optional
		TODO
	zaxis_visible : `bool`, optional
		:obj:`False` to make the z-axis invisible. Defaults to :obj:`True`.
	legend_on : `bool`, optional
		:obj:`True` to show the legend, :obj:`False` otherwise. Defaults to :obj:`False`.
	legend_loc : `str`, optional
		String specifying the location where the legend box should be placed.
		Defaults to 'best'; please see :func:`matplotlib.pyplot.legend` for all
		the available options.
	legend_framealpha : `float`, optional
		TODO
	text : str
		Text to be added to the figure as anchored text. Defaults to :obj:`None`,
		and no text box is shown.
	text_loc : str
		String specifying the location where the text box should be placed.
		Defaults to 'upper right'; please see :class:`matplotlib.offsetbox.AnchoredText`
		for all the available options.
	grid_on : `bool`, optional
		:obj:`True` to show the legend, :obj:`False` otherwise. Defaults to :obj:`False`.
	grid_properties : `dict`, optional
		TODO
	"""
	fontsize                  = kwargs.get('fontsize', 12)
	title_center              = kwargs.get('title_center', '')
	title_left                = kwargs.get('title_left', '')
	title_right               = kwargs.get('title_right', '')
	x_label                   = kwargs.get('x_label', '')
	x_lim                     = kwargs.get('x_lim', None)
	invert_xaxis              = kwargs.get('invert_xaxis', False)
	x_scale                   = kwargs.get('x_scale', None)
	x_ticks                   = kwargs.get('x_ticks', None)
	x_ticklabels              = kwargs.get('x_ticklabels', None)
	xaxis_minor_ticks_visible = kwargs.get('xaxis_minor_ticks_visible', False)
	xaxis_visible             = kwargs.get('xaxis_visible', True)
	y_label                   = kwargs.get('y_label', '')
	y_lim                     = kwargs.get('y_lim', None)
	invert_yaxis              = kwargs.get('invert_yaxis', False)
	y_scale                   = kwargs.get('y_scale', None)
	y_ticks                   = kwargs.get('y_ticks', None)
	y_ticklabels              = kwargs.get('y_ticklabels', None)
	yaxis_minor_ticks_visible = kwargs.get('yaxis_minor_ticks_visible', False)
	yaxis_visible             = kwargs.get('yaxis_visible', True)
	z_label                   = kwargs.get('z_label', '')
	z_lim                     = kwargs.get('z_lim', None)
	invert_zaxis              = kwargs.get('invert_zaxis', False)
	z_scale                   = kwargs.get('z_scale', None)
	z_ticks                   = kwargs.get('z_ticks', None)
	z_ticklabels              = kwargs.get('z_ticklabels', None)
	zaxis_minor_ticks_visible = kwargs.get('zaxis_minor_ticks_visible', False)
	zaxis_visible             = kwargs.get('zaxis_visible', True)
	legend_on                 = kwargs.get('legend_on', False)
	legend_loc                = kwargs.get('legend_loc', 'best')
	legend_framealpha         = kwargs.get('legend_framealpha', 0.5)
	text                      = kwargs.get('text', None)
	text_loc                  = kwargs.get('text_loc', '')
	grid_on                   = kwargs.get('grid_on', False)
	grid_properties           = kwargs.get('grid_properties', None)

	rcParams['font.size'] = fontsize

	if ax.get_title(loc='center') == '':
		ax.set_title(title_center, loc='center', fontsize=rcParams['font.size']-1)
	if ax.get_title(loc='left') == '':
		ax.set_title(title_left, loc='left', fontsize=rcParams['font.size']-1)
	if ax.get_title(loc='right') == '':
		ax.set_title(title_right, loc='right', fontsize=rcParams['font.size']-1)

	if ax.get_xlabel() == '':
		ax.set(xlabel=x_label)
	if ax.get_ylabel() == '':
		ax.set(ylabel=y_label)
	try:
		if ax.get_zlabel() == '':
			ax.set(zlabel=z_label)
	except AttributeError:
		if z_label != '':
			import warnings
			warnings.warn('The plot is not three-dimensional, therefore the '
					  	   'argument ''z_label'' is disregarded.', RuntimeWarning)
		else:
			pass

	if x_lim is not None:
		ax.set_xlim(x_lim)
	if y_lim is not None:
		ax.set_ylim(y_lim)
	try:
		if z_lim is not None:
			ax.set_zlim(z_lim)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''z_lim'' is disregarded.', RuntimeWarning)

	if invert_xaxis:
		ax.invert_xaxis()
	if invert_yaxis:
		ax.invert_yaxis()
	try:
		if invert_zaxis:
			ax.invert_zaxis()
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''invert_zaxis'' is disregarded.', RuntimeWarning)

	if x_scale is not None:
		ax.set_xscale(x_scale)
	if y_scale is not None:
		ax.set_yscale(y_scale)
	try:
		if z_scale is not None:
			ax.set_zscale(z_scale)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''z_scale'' is disregarded.', RuntimeWarning)

	if x_ticks is not None:
		ax.get_xaxis().set_ticks(x_ticks)
	if y_ticks is not None:
		ax.get_yaxis().set_ticks(y_ticks)
	try:
		if z_ticks is not None:
			ax.get_zaxis().set_ticks(z_ticks)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''z_ticks'' is disregarded.', RuntimeWarning)

	if x_ticklabels is not None:
		ax.get_xaxis().set_ticklabels(x_ticklabels)
	if y_ticklabels is not None:
		ax.get_yaxis().set_ticklabels(y_ticklabels)
	try:
		if z_ticklabels is not None:
			ax.get_zaxis().set_ticklabels(z_ticklabels)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''z_ticklabels'' is disregarded.', RuntimeWarning)

	if not xaxis_minor_ticks_visible:
		ax.get_xaxis().set_tick_params(which='minor', size=0)
		ax.get_xaxis().set_tick_params(which='minor', width=0)
	if not yaxis_minor_ticks_visible:
		ax.get_yaxis().set_tick_params(which='minor', size=0)
		ax.get_yaxis().set_tick_params(which='minor', width=0)
	try:
		if not zaxis_minor_ticks_visible:
			ax.get_zaxis().set_tick_params(which='minor', size=0)
			ax.get_zaxis().set_tick_params(which='minor', width=0)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''zaxis_minor_ticks_visible'' is disregarded.',
					  RuntimeWarning)

	if not xaxis_visible:
		ax.get_xaxis().set_visible(False)
	if not yaxis_visible:
		ax.get_yaxis().set_visible(False)
	try:
		if not zaxis_visible:
			ax.get_zaxis().set_visible(False)
	except AttributeError:
		import warnings
		warnings.warn('The plot is not three-dimensional, therefore the '
					  'argument ''zaxis_visible'' is disregarded.', RuntimeWarning)

	if legend_on:
		ax.legend(loc=legend_loc, framealpha=legend_framealpha)

	if text is not None:
		ax.add_artist(AnchoredText(text, loc=text_locations[text_loc]))

	if grid_on:
		g_props = grid_properties if grid_properties is not None else {}
		ax.grid(True, **g_props)


def reverse_colormap(cmap, name=None):
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


def set_colorbar(fig, mappable, color_levels, *, cbar_ticks_step=1, cbar_ticks_pos='center',
				 cbar_title='', cbar_x_label='', cbar_y_label='',
				 cbar_orientation='vertical', cbar_ax=None):
	"""
	An utility to ease the configuration of the colorbar in Matplotlib plots.

	Parameters
	----------
	fig : figure
		The :class:`matplotlib.pyplot.figure` containing the plot.
	mappable : mappable
		The mappable, i.e., the image, which the colorbar applies
	color_levels : array_like
		1-D array of the levels corresponding to the colorbar colors.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the colorbar. Defaults to 1,
		i.e., all ticks are displayed with the corresponding label.
	cbar_ticks_pos : str
		'center' to place the color bar ticks in the middle of the color intervals,
		anything else to place the ticks at the interfaces between color intervals.
	cbar_x_label : str
		Label for the horizontal axis of the colorbar. Defaults to an empty string.
	cbar_y_label : str
		Label for the vertical axis of the colorbar. Defaults to an empty string.
	cbar_title : str
		Title for the colorbar. Defaults to an empty string.
	cbar_orientation : str
		Orientation of the colorbar. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the colorbar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the colorbar. If no indices are given,
		only the current axes are resized.
	"""
	if cbar_ax is None:
		cb = plt.colorbar(mappable, orientation=cbar_orientation)
	else:
		try:
			axes = fig.get_axes()
			cb = plt.colorbar(mappable, orientation=cbar_orientation,
                              ax=[axes[i] for i in cbar_ax])
		except TypeError:
			# cbar_ax is not iterable
			cb = plt.colorbar(mappable, orientation=cbar_orientation)
		except IndexError:
			# cbar_ax contains an index which exceeds the number of axes in the figure
			cb = plt.colorbar(mappable, orientation=cbar_orientation)

	cb.ax.set_title(cbar_title)
	cb.ax.set_xlabel(cbar_x_label)
	cb.ax.set_ylabel(cbar_y_label)

	if cbar_ticks_pos == 'center':
		cb.set_ticks(0.5 * (color_levels[:-1] + color_levels[1:])[::cbar_ticks_step])
	else:
		cb.set_ticks(color_levels[::cbar_ticks_step])


def make_lineplot(x, y, ax, **kwargs):
	"""
	Utility plotting a line.

	Parameters
	----------
	x : array_like
		1-D :class:`numpy.ndarray` gathering the x-coordinates
		of the points to plot.
	y : array_like
		1-D :class:`numpy.ndarray` gathering the y-coordinates
		of the points to plot.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Defaults to 16.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Defaults to 1.
	y_factor : float
		Scaling factor for the field. Defaults to 1.
	linestyle : str
		String specifying the line style. The default line style is '-'.
	linewidth : float
		The line width. Defaults to 1.5.
	linecolor : str
		String specifying the line color. Defaults to 'blue'.
	marker : str
		TODO
	markersize : float
		TODO
	markeredgewidth : str
		TODO
	markerfacecolor : str
		TODO
	markeredgecolor : str
		TODO
	legend_label : str
		The legend label for the line. Defaults to an empty string.
	"""
	# Get keyword arguments
	fontsize     	= kwargs.get('fontsize', 16)
	x_factor     	= kwargs.get('x_factor', 1.)
	y_factor     	= kwargs.get('y_factor', 1.)
	linestyle    	= kwargs.get('linestyle', '-')
	linewidth    	= kwargs.get('linewidth', 1.5)
	linecolor	 	= kwargs.get('linecolor', 'blue')
	marker  		= kwargs.get('marker', None)
	markersize  	= kwargs.get('markersize', None)
	markeredgewidth = kwargs.get('markeredgewidth', None)
	markerfacecolor = kwargs.get('markerfacecolor', None)
	markeredgecolor = kwargs.get('markeredgecolor', None)
	legend_label 	= kwargs.get('legend_label', '')

	# Global settings
	rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x *= x_factor
	y *= y_factor

	# Plot the field
	if legend_label == '' or legend_label is None:
		ax.plot(x, y, color=linecolor, linestyle=linestyle, linewidth=linewidth,
				marker=marker, markersize=markersize, markeredgewidth=markeredgewidth,
				markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
	else:
		ax.plot(x, y, color=linecolor, linestyle=linestyle, linewidth=linewidth,
				marker=marker, markersize=markersize, markeredgewidth=markeredgewidth,
				markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor,
				label=legend_label)

	# Bring axes back to original units
	x /= x_factor
	y /= y_factor


def make_contour(x, y, field, ax, **kwargs):
	"""
	Generate a contour plot.

	Parameters
	----------
	x : array_like
		2-D :class:`numpy.ndarray` gathering the x-coordinates
		of the grid points.
	y : array_like
		2-D :class:`numpy.ndarray` gathering the y-coordinates
		of the grid points.
	field : array_like
		2-D :class:`numpy.ndarray` representing the field to plot.
	ax : axes
		An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used in the plot. Defaults to 16.
	x_factor : float
		Scaling factor for the x-axis. Defaults to 1.
	y_factor : float
		Scaling factor for the y-axis. Defaults to 1.
	field_bias : float
		Bias for the field, so that the contour lines for :obj:`field - field_bias`
		are drawn. Defaults to 0.
	field_factor : float
		Scaling factor for the field, so that the contour lines for
		:obj:`field_factor * field` are drawn. If a bias is specified, then the contour
		lines for :obj:`field_factor * (field - field_bias)` are drawn. Defaults to 1.
	alpha : float
		TODO
	colors : str, sequence[str]
		TODO
	draw_vertical_levels : bool
		:obj:`True` to draw the underlying grid, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	"""
	# Shortcuts
	ni, nk = field.shape

	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 16)
	x_factor         = kwargs.get('x_factor', 1.)
	y_factor         = kwargs.get('y_factor', 1.)
	field_bias	 	 = kwargs.get('field_bias', 0.)
	field_factor	 = kwargs.get('field_factor', 1.)
	alpha			 = kwargs.get('alpha', 1.0)
	colors			 = kwargs.get('colors', 'black')
	draw_grid 	 	 = kwargs.get('draw_vertical_levels', False)

	# Global settings
	rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x     *= x_factor
	y     *= y_factor
	field -= field_bias
	field *= field_factor

	# Plot the computational grid
	if draw_grid:
		for k in range(nk):
			ax.plot(x[:, k], y[:, k], color='gray', linewidth=1, alpha=0.5)

	# Plot the field
	plt.contour(x, y, field, colors=colors, alpha=alpha)

	# Bring axes and field back to original units
	x     /= x_factor
	y 	  /= y_factor
	field /= field_factor
	field += field_bias


def make_contourf(x, y, field, fig, ax, **kwargs):
	"""
	Generate a contourf plot.

	Parameters
	----------
	x : array_like
		2-D :class:`numpy.ndarray` gathering the x-coordinates
		of the grid points.
	y : array_like
		2-D :class:`numpy.ndarray` gathering the y-coordinates
		of the grid points.
	field : array_like
		2-D :class:`numpy.ndarray` representing the field to plot.
	fig : figure
        A :class:`matplotlib.pyplot.figure`.
	ax : axes
        An instance of :class:`matplotlib.axes.Axes`.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Defaults to 16.
	x_factor : float
		Scaling factor for the x-axis. Defaults to 1.
	y_factor : float
		Scaling factor for the y-axis. Defaults to 1.
	field_bias : float
		Bias for the field, so that the contourf plot for :obj:`field - field_bias`
		is generated. Defaults to 0.
	field_factor : float
		Scaling factor for the field, so that the contourf plot for
		:obj:`field_factor * field` is generated. If a bias is specified, then the
		contourf plot for :obj:`field_factor * (field - field_bias)` is generated.
		Defaults to 1.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided
		by Matplotlib, as well as the corresponding inverted versions, are available.
	cbar_on : bool
		:obj:`True` to show the color bar, :obj:`False` otherwise. Defaults to :obj:`True`.
	cbar_levels : int
		Number of levels for the color bar. Defaults to 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Defaults to 1,
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
		Label for the horizontal axis of the color bar. Defaults to an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Defaults to an empty string.
	cbar_title : str
		Title for the color bar. Defaults to an empty string.
	cbar_orientation : str
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the color bar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the color bar. If no indices are given,
		only the current axes are resized.
	draw_vertical_levels : bool
		:obj:`True` to draw the underlying grid, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	"""
	# Get keyword arguments
	fontsize         = kwargs.get('fontsize', 12)
	x_factor         = kwargs.get('x_factor', 1.)
	y_factor         = kwargs.get('y_factor', 1.)
	field_bias		 = kwargs.get('field_bias', 0.)
	field_factor     = kwargs.get('field_factor', 1.)
	cmap_name        = kwargs.get('cmap_name', 'RdYlBu')
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
	draw_grid 	 	 = kwargs.get('draw_vertical_levels', False)

	# Global settings
	rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x     *= x_factor
	y     *= y_factor
	field -= field_bias
	field *= field_factor

	# Create color bar for colormap
	field_min, field_max = np.amin(field), np.amax(field)
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
		cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	else:
		cm = plt.get_cmap(cmap_name)

	# Plot the computational grid
	if draw_grid:
		for k in range(x.shape[1]):
			ax.plot(x[:, k], y[:, k], color='gray', linewidth=1, alpha=0.5)

	# Plot the field
	surf = ax.contourf(x, y, field, color_levels, cmap=cm)

	# Set the color bar
	if cbar_on:
		set_colorbar(fig, surf, color_levels,
			  	     cbar_ticks_step=cbar_ticks_step,
					 cbar_ticks_pos=cbar_ticks_pos, cbar_title=cbar_title,
					 cbar_x_label=cbar_x_label, cbar_y_label=cbar_y_label,
					 cbar_orientation=cbar_orientation, cbar_ax=cbar_ax)

	# Bring axes and field back to original units
	x     /= x_factor
	y 	  /= y_factor
	field /= field_factor
	field += field_bias


def make_quiver(x, y, vx, vy, scalar, fig, ax, **kwargs):
	"""
	Generate the quiver plot of a gridded vector field at a cross-section
	parallel to a coordinate plane.

	Parameters
	----------
	x : array_like
		2-D :class:`numpy.ndarray` gathering the x-coordinates
		of the grid points.
	y : array_like
		2-D :class:`numpy.ndarray` gathering the y-coordinates
		of the grid points.
	vx : array_like
		2-D :class:`numpy.ndarray` representing the x-component
		of the field to plot.
	vy : array_like
		2-D :class:`numpy.ndarray` representing the y-component
		of the field to plot.
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
		The fontsize to be used. Defaults to 12.
	x_factor : float
		Scaling factor for the :math:`x`-axis. Defaults to 1.
	x_step : int
		Maximum distance between the :math:`x`-index of a drawn point, and the
		:math:`x`-index of any of its neighbours. Defaults to 2, i.e., only half
		of the points will be drawn.
	y_factor : float
		Scaling factor for the :math:`y`-axis. Defaults to 1.
	y_step : int
		Maximum distance between the :math:`y`-index of a drawn point, and the
		:math:`y`-index of any of its neighbours. Defaults to 2, i.e., only half
		of the points will be drawn.
	scalar_bias : float
		Bias for the scalar field, so that the arrows will be colored based on
		:obj:`scalar - scalar_bias`. Defaults to 0.
	scalar_factor : float
		Scaling factor for the scalar field, so that the arrows will be colored based on
		:obj:`scalar_factor * scalar`. If a bias is specified, then the arrows will be
		colored based on :obj:`scalar_factor * (scalar - scalar_bias)` are drawn.
		Defaults to 1.
	arrow_scale : float
		TODO
	arrow_scale_units : float
		TODO
	arrow_headwidth : float
		TODO
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided
		by Matplotlib, as well as the corresponding inverted versions, are available.
		If not specified, no color map will be used, and the arrows will draw black.
	cbar_on : bool
		:obj:`True` to show the color bar, :obj:`False` otherwise. Defaults to :obj:`True`.
	cbar_levels : int
		Number of levels for the color bar. Defaults to 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the color bar. Defaults to 1,
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
		Label for the horizontal axis of the color bar. Defaults to an empty string.
	cbar_y_label : str
		Label for the vertical axis of the color bar. Defaults to an empty string.
	cbar_title : str
		Title for the color bar. Defaults to an empty string.
	cbar_orientation : str
		Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the color bar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the color bar. If no indices are given,
		only the current axes are resized.
	quiverkey_on : bool
		TODO
	quiverkey_loc : tuple
		TODO
	quiverkey_length : float
		TODO
	quiverkey_label : str
		TODO
	quiverkey_label_loc : str
		TODO
	quiverkey_fontproperties : dict
		TODO
	draw_vertical_levels : bool
		:obj:`True` to draw the underlying grid, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	"""
	# Get keyword arguments
	fontsize          	= kwargs.get('fontsize', 12)
	x_factor          	= kwargs.get('x_factor', 1.)
	x_step            	= kwargs.get('x_step', 2)
	y_factor          	= kwargs.get('y_factor', 1.)
	y_step            	= kwargs.get('y_step', 2)
	scalar_bias		  	= kwargs.get('scalar_bias', 0.)
	scalar_factor     	= kwargs.get('scalar_factor', 1.)
	arrow_scale		  	= kwargs.get('arrow_scale', None)
	arrow_scale_units 	= kwargs.get('arrow_scale_units', None)
	arrow_headwidth 	= kwargs.get('arrow_headwidth', 3.0)
	cmap_name         	= kwargs.get('cmap_name', None)
	cbar_on			  	= kwargs.get('cbar_on', True)
	cbar_levels		  	= kwargs.get('cbar_levels', 14)
	cbar_ticks_step   	= kwargs.get('cbar_ticks_step', 1)
	cbar_ticks_pos	  	= kwargs.get('cbar_ticks_pos', 'center')
	cbar_center		  	= kwargs.get('cbar_center', None)
	cbar_half_width   	= kwargs.get('cbar_half_width', None)
	cbar_x_label	  	= kwargs.get('cbar_x_label', '')
	cbar_y_label	  	= kwargs.get('cbar_y_label', '')
	cbar_title		  	= kwargs.get('cbar_title', '')
	cbar_orientation  	= kwargs.get('cbar_orientation', 'vertical')
	cbar_ax			  	= kwargs.get('cbar_ax', None)
	quiverkey_on	  	= kwargs.get('quiverkey_on', False)
	quiverkey_loc	  	= kwargs.get('quiverkey_loc', (1, 1))
	quiverkey_length	= kwargs.get('quiverkey_length', 1.0)
	quiverkey_label	  	= kwargs.get('quiverkey_label', '1 m s$^{-1}$')
	quiverkey_label_loc	= kwargs.get('quiverkey_label_loc', 'E')
	quiverkey_fontproperties = kwargs.get('quiverkey_fontproperties', {})
	draw_grid 	 	 	= kwargs.get('draw_vertical_levels', False)

	# Global settings
	rcParams['font.size'] = fontsize

	# Rescale the axes and the field for visualization purposes
	x *= x_factor
	y *= y_factor
	if scalar is not None:
		scalar -= scalar_bias
		scalar *= scalar_factor

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
			cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
		else:
			cm = plt.get_cmap(cmap_name)
	else:
		cm = None

	# Plot the computational grid
	if draw_grid:
		for k in range(x.shape[1]):
			ax.plot(x[:, k], y[:, k], color='gray', linewidth=1, alpha=0.5)

	# Generate quiver-plot
	if cm is None:
		q = ax.quiver(x[::x_step, ::y_step], y[::x_step, ::y_step],
					  vx[::x_step, ::y_step], vy[::x_step, ::y_step],
					  scale=arrow_scale, scale_units=arrow_scale_units,
					  headwidth=arrow_headwidth)
	else:
		q = ax.quiver(x[::x_step, ::y_step], y[::x_step, ::y_step],
					  vx[::x_step, ::y_step], vy[::x_step, ::y_step],
					  scalar[::x_step, ::y_step], cmap=cm,
					  scale=arrow_scale, scale_units=arrow_scale_units,
					  headwidth=arrow_headwidth)

	# Set the color bar
	if cm is not None and cbar_on:
		set_colorbar(fig, q, color_levels,
					 cbar_ticks_step=cbar_ticks_step,
					 cbar_ticks_pos=cbar_ticks_pos, cbar_title=cbar_title,
					 cbar_x_label=cbar_x_label, cbar_y_label=cbar_y_label,
					 cbar_orientation=cbar_orientation, cbar_ax=cbar_ax)

	# Set quiverkey
	if quiverkey_on:
		ax.quiverkey(q, quiverkey_loc[0], quiverkey_loc[1], quiverkey_length,
					 quiverkey_label, coordinates='axes', labelpos=quiverkey_label_loc,
					 fontproperties=quiverkey_fontproperties)

	# Bring axes and field back to original units
	x /= x_factor
	y /= y_factor
	if scalar is not None:
		scalar /= scalar_factor
		scalar += scalar_bias
