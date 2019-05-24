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
	make_rectangle
"""
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from tasmania.python.plot.utils import smaller_than as lt, equal_to as eq


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


def get_figure_and_axes(
	fig=None, ax=None, default_fig=None, nrows=1, ncols=1, index=1, **kwargs
):
	"""
	Get a :class:`matplotlib.figure.Figure` object and a :class:`matplotlib.axes.Axes`
	object, with the latter embedded in the former. The returned values are determined 
	as follows:

		* If both `fig` and `ax` arguments are passed in:

			- if `ax` is embedded in `fig`, return `fig` and `ax`;
			- otherwise, return the figure which encloses `ax`, and `ax` itself;

		* If `fig` is provided but `ax` is not:

			- if `fig` contains some subplots, return `fig` and the axes of the
				subplot in position (`nrows`, `ncols`, `index`) it contains;
			- otherwise, add a subplot to `fig` in position (`nrows`, `ncols`, `index`)
				and return `fig` and the subplot axes;

		* If `ax` is provided but `fig` is not, return the figure which encloses `ax`,
			and `ax` itself;

		* If neither `fig` nor `ax` are passed in:

			- if `default_fig` is not given, instantiate a new pair of figure and axes;
			- if `default_fig` is provided and it contains some subplots, return
				`default_fig` and the axes of the subplot in position
				(`nrows`, `ncols`, `index`) it contains;
			- if `default_fig` is provided but is does not contain any subplot, add a
				subplot to `default_fig` in position (1,1,1) and return `default_fig`
				and the subplot axes.

	Parameters
	----------
	fig : `matplotlib.figure.Figure`, optional
		The figure.
	ax : `matplotlib.axes.Axes`, optional
        The axes.
	default_fig : `matplotlib.figure.Figure`, optional
        The default figure.
	nrows : `int`, optional
		Number of rows of the subplot grid. Defaults to 1.
	ncols : `int`, optional
		Number of columns of the subplot grid. Defaults to 1.
	index : `int`, optional
		Axes' index in the subplot grid. Defaults to 1.

	Keyword arguments
	-----------------
	figsize : `tuple`, optional
		The size which the output figure should have. Defaults to (7, 7).
		This argument is effective only if the figure is created within the function.
	fontsize : `int`, optional
		Font size for the output figure and axes. Defaults to 12.
		This argument is effective only if the figure is created within the function.
	projection : `str`, optional
		The axes projection. Defaults to :obj:`None`.
		This argument is effective only if the figure is created within the function.

	Returns
	-------
	out_fig : matplotlib.figure.Figure
        The figure.
	out_ax : matplotlib.axes.Axes
        The axes.
	"""
	figsize    = kwargs.get('figsize', (7, 7))
	fontsize   = kwargs.get('fontsize', 12)
	projection = kwargs.get('projection', None)

	rcParams['font.size'] = fontsize

	if (fig is not None) and (ax is not None):
		try:
			if ax not in fig.get_axes():
				import warnings
				warnings.warn(
					"Input axes do not belong to the input figure, "
					"so the figure which the axes belong to is considered.",
					RuntimeWarning
				)

				out_fig, out_ax = ax.get_figure(), ax
			else:
				out_fig, out_ax = fig, ax
		except AttributeError:
			import warnings
			warnings.warn(
				"Input argument ''fig'' does not seem to be a matplotlib.figure.Figure, "
				"so the figure the axes belong to are considered.",
				RuntimeWarning
			)

			out_fig, out_ax = ax.get_figure(), ax

	if (fig is not None) and (ax is None):
		try:
			out_fig = fig
			out_ax = out_fig.add_subplot(nrows, ncols, index, projection=projection)
		except AttributeError:
			import warnings
			warnings.warn(
				"Input argument ''fig'' does not seem to be a matplotlib.figure.Figure, "
				"hence a proper matplotlib.figure.Figure object is created.",
				RuntimeWarning
			)

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
				out_ax = out_fig.add_subplot(nrows, ncols, index, projection=projection)
			except AttributeError:
				import warnings
				warnings.warn(
					"Input argument ''default_fig'' does not seem to be a "
					"matplotlib.figure.Figure, hence a proper matplotlib.figure.Figure " 
					"object is created.",
					RuntimeWarning
				)

				out_fig = plt.figure(figsize=figsize)
				out_ax = out_fig.add_subplot(nrows, ncols, index, projection=projection)

	return out_fig, out_ax


def set_figure_properties(fig, **kwargs):
	"""
	Ease the configuration of a :class:`matplotlib.figure.Figure`.

	Parameters
	----------
	fig : matplotlib.figure.Figure
		The figure.

	Keyword arguments
	-----------------
	fontsize : int
		Font size to use for the plot titles, and axes ticks and labels.
		Defaults to 12.
	tight_layout : bool
		:obj:`True` to fit the whole subplots into the figure area,
		:obj:`False` otherwise. Defaults to :obj:`True`.
	tight_layout_rect : tuple
		A rectangle (left, bottom, right, top) in the normalized figure
		coordinate that the whole subplots area (including labels) will
		fit into. Defaults to (0, 0, 1, 1).
	suptitle : str
		The figure title. Defaults to an empty string.
	"""
	fontsize     	  = kwargs.get('fontsize', 12)
	tight_layout 	  = kwargs.get('tight_layout', True)
	tight_layout_rect = kwargs.get('tight_layout_rect', (0, 0, 1, 1))
	suptitle     	  = kwargs.get('suptitle', '')

	rcParams['font.size'] = fontsize

	if tight_layout:
		fig.tight_layout(rect=tight_layout_rect)

	if suptitle != '':
		fig.suptitle(suptitle, fontsize=fontsize+1)


def set_axes_properties(ax, **kwargs):
	"""
	Ease the configuration of a :class:`matplotlib.axes.Axes` object.

	Parameters
	----------
	ax : matplotlib.axes.Axes
		The axes.

	Keyword arguments
	-----------------
	fontsize : int
		Font size to use for the plot titles, and axes ticks and labels.
		Defaults to 12.
	title_center : str
		The center title. Defaults to an empty string.
	title_left : str
		The left title. Defaults to an empty string.
	title_right : str
		The right title. Defaults to an empty string.
	x_label : str
		The x-axis label. Defaults to an empty string.
	x_labelcolor : str
		Color of the x-axis label. Defaults to 'black'.
	x_lim : tuple
		Data limits for the x-axis. Defaults to :obj:`None`, i.e., the data limits
		will be left unchanged.
	invert_xaxis : bool
		:obj:`True` to make to invert the x-axis, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	x_scale : str
		The x-axis scale. Defaults to 'linear'.
	x_ticks : sequence[float]
		Sequence of x-axis ticks location. Defaults to :obj:`None`.
	x_ticklabels : sequence[str]
		Sequence of x-axis ticks labels. Defaults to :obj:`None`.
	x_ticklabelcolor : str
		Color for the x-axis ticks labels. Defaults to 'black'.
	xaxis_minor_ticks_visible : bool
		:obj:`True` to show all ticks, either labelled or unlabelled,
		:obj:`False` to show only the labelled ticks. Defaults to :obj:`False`.
	xaxis_visible : bool
		:obj:`False` to make the x-axis invisible. Defaults to :obj:`True`.
	y_label : str
		The y-axis label. Defaults to an empty string.
	y_labelcolor : str
		Color of the y-axis label. Defaults to 'black'.
	y_lim : tuple
		Data limits for the y-axis. Defaults to :obj:`None`, i.e., the data limits
		will be left unchanged.
	invert_yaxis : bool
		:obj:`True` to make to invert the y-axis, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	y_scale : str
		The y-axis scale. Defaults to 'linear'.
	y_ticks : sequence[float]
		Sequence of y-axis ticks location. Defaults to :obj:`None`.
	y_ticklabels : sequence[str]
		Sequence of y-axis ticks labels. Defaults to :obj:`None`.
	y_ticklabelcolor : str
		Color for the y-axis ticks labels. Defaults to 'black'.
	yaxis_minor_ticks_visible : bool
		:obj:`True` to show all ticks, either labelled or unlabelled,
		:obj:`False` to show only the labelled ticks. Defaults to :obj:`False`.
	yaxis_visible : bool
		:obj:`False` to make the y-axis invisible. Defaults to :obj:`True`.
	z_label : str
		The z-axis label. Defaults to an empty string.
	z_labelcolor : str
		Color of the z-axis label. Defaults to 'black'.
	z_lim : tuple
		Data limits for the z-axis. Defaults to :obj:`None`, i.e., the data limits
		will be left unchanged.
	invert_zaxis : bool
		:obj:`True` to make to invert the z-axis, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	z_scale : str
		The z-axis scale. Defaults to 'linear'.
	z_ticks : sequence[float]
		Sequence of z-axis ticks location. Defaults to :obj:`None`.
	z_ticklabels : sequence[str]
		Sequence of z-axis ticks labels. Defaults to :obj:`None`.
	z_ticklabelcolor : str
		Color for the z-axis ticks labels. Defaults to 'black'.
	zaxis_minor_ticks_visible : bool
		:obj:`True` to show all ticks, either labelled or unlabelled,
		:obj:`False` to show only the labelled ticks. Defaults to :obj:`False`.
	zaxis_visible : bool
		:obj:`False` to make the z-axis invisible. Defaults to :obj:`True`.
	legend_on : bool
		:obj:`True` to show the legend, :obj:`False` otherwise. Defaults to :obj:`False`.
	legend_loc : str
		String specifying the location where the legend should be placed.
		Defaults to 'best'; please see :func:`matplotlib.pyplot.legend` for all
		the available options.
	legend_bbox_to_anchor : tuple
		4-items tuple defining the box used to place the legend. This is used in
		conjuction with `legend_loc` to allow arbitrary placement of the legend.
	legend_framealpha : float
		Legend transparency. It should be between 0 and 1; defaults to 0.5.
	legend_ncol : int
		Number of columns into which the legend labels should be arranged.
		Defaults to 1.
	text : str
		Text to be added to the figure as anchored text. Defaults to :obj:`None`,
		meaning that no text box is shown.
	text_loc : str
		String specifying the location where the text box should be placed.
		Defaults to 'upper right'; please see :class:`matplotlib.offsetbox.AnchoredText`
		for all the available options.
	grid_on : bool
		:obj:`True` to show the plot grid, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	grid_properties : dict
		Keyword arguments specifying various settings of the plot grid.
	"""
	fontsize                  = kwargs.get('fontsize', 12)
	# title
	title_center              = kwargs.get('title_center', '')
	title_left                = kwargs.get('title_left', '')
	title_right               = kwargs.get('title_right', '')
	# x-axis
	x_label                   = kwargs.get('x_label', '')
	x_labelcolor              = kwargs.get('x_labelcolor', 'black')
	x_lim                     = kwargs.get('x_lim', None)
	invert_xaxis              = kwargs.get('invert_xaxis', False)
	x_scale                   = kwargs.get('x_scale', 'linear')
	x_ticks                   = kwargs.get('x_ticks', None)
	x_ticklabels              = kwargs.get('x_ticklabels', None)
	x_ticklabelcolor          = kwargs.get('x_ticklabelcolor', 'black')
	xaxis_minor_ticks_visible = kwargs.get('xaxis_minor_ticks_visible', False)
	xaxis_visible             = kwargs.get('xaxis_visible', True)
	# y-axis
	y_label                   = kwargs.get('y_label', '')
	y_labelcolor              = kwargs.get('y_labelcolor', 'black')
	y_lim                     = kwargs.get('y_lim', None)
	invert_yaxis              = kwargs.get('invert_yaxis', False)
	y_scale                   = kwargs.get('y_scale', 'linear')
	y_ticks                   = kwargs.get('y_ticks', None)
	y_ticklabels              = kwargs.get('y_ticklabels', None)
	y_ticklabelcolor          = kwargs.get('y_ticklabelcolor', 'black')
	yaxis_minor_ticks_visible = kwargs.get('yaxis_minor_ticks_visible', False)
	yaxis_visible             = kwargs.get('yaxis_visible', True)
	# z-axis
	z_label                   = kwargs.get('z_label', '')
	z_labelcolor              = kwargs.get('z_labelcolor', 'black')
	z_lim                     = kwargs.get('z_lim', None)
	invert_zaxis              = kwargs.get('invert_zaxis', False)
	z_scale                   = kwargs.get('z_scale', 'linear')
	z_ticks                   = kwargs.get('z_ticks', None)
	z_ticklabels              = kwargs.get('z_ticklabels', None)
	z_ticklabelcolor          = kwargs.get('z_ticklabelcolor', 'black')
	zaxis_minor_ticks_visible = kwargs.get('zaxis_minor_ticks_visible', False)
	zaxis_visible             = kwargs.get('zaxis_visible', True)
	# legend
	legend_on                 = kwargs.get('legend_on', False)
	legend_loc                = kwargs.get('legend_loc', 'best')
	legend_bbox_to_anchor     = kwargs.get('legend_bbox_to_anchor', None)
	legend_framealpha         = kwargs.get('legend_framealpha', 0.5)
	legend_ncol				  = kwargs.get('legend_ncol', 1)
	# textbox
	text                      = kwargs.get('text', None)
	text_loc                  = kwargs.get('text_loc', '')
	# grid
	grid_on                   = kwargs.get('grid_on', False)
	grid_properties           = kwargs.get('grid_properties', None)

	rcParams['font.size'] = fontsize

	# plot titles
	if ax.get_title(loc='center') == '':
		ax.set_title(title_center, loc='center', fontsize=rcParams['font.size']-1)
	if ax.get_title(loc='left') == '':
		ax.set_title(title_left, loc='left', fontsize=rcParams['font.size']-1)
	if ax.get_title(loc='right') == '':
		ax.set_title(title_right, loc='right', fontsize=rcParams['font.size']-1)

	# axes labels
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
			warnings.warn(
				"The plot is not three-dimensional, therefore the "
				"argument ''z_label'' is disregarded.", RuntimeWarning
			)
		else:
			pass

	# axes labels
	if ax.get_xlabel() != '' and x_labelcolor != '':
		ax.xaxis.label.set_color(x_labelcolor)
	if ax.get_ylabel() != '' and y_labelcolor != '':
		ax.yaxis.label.set_color(y_labelcolor)
	try:
		if ax.get_zlabel() != '' and z_labelcolor != '':
			ax.zaxis.label.set_color(z_labelcolor)
	except AttributeError:
		if z_labelcolor != '':
			import warnings
			warnings.warn(
				"The plot is not three-dimensional, therefore the "
				"argument ''z_labelcolor'' is disregarded.", RuntimeWarning
			)
		else:
			pass

	# axes limits
	if x_lim is not None:
		ax.set_xlim(x_lim)
	if y_lim is not None:
		ax.set_ylim(y_lim)
	try:
		if z_lim is not None:
			ax.set_zlim(z_lim)
	except AttributeError:
		import warnings
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''z_lim'' is disregarded.", RuntimeWarning
		)

	# invert the axes
	if invert_xaxis:
		ax.invert_xaxis()
	if invert_yaxis:
		ax.invert_yaxis()
	try:
		if invert_zaxis:
			ax.invert_zaxis()
	except AttributeError:
		import warnings
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''invert_zaxis'' is disregarded.", RuntimeWarning
		)

	# axes scale
	if x_scale is not None:
		ax.set_xscale(x_scale)
	if y_scale is not None:
		ax.set_yscale(y_scale)
	try:
		if z_scale is not None:
			ax.set_zscale(z_scale)
	except AttributeError:
		import warnings
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''z_scale'' is disregarded.", RuntimeWarning
		)

	# axes ticks
	if x_ticks is not None:
		ax.get_xaxis().set_ticks(x_ticks)
	if y_ticks is not None:
		ax.get_yaxis().set_ticks(y_ticks)
	try:
		if z_ticks is not None:
			ax.get_zaxis().set_ticks(z_ticks)
	except AttributeError:
		import warnings
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''z_ticks'' is disregarded.", RuntimeWarning
		)

	# axes tick labels
	if x_ticklabels is not None:
		ax.get_xaxis().set_ticklabels(x_ticklabels)
	if y_ticklabels is not None:
		ax.get_yaxis().set_ticklabels(y_ticklabels)
	try:
		if z_ticklabels is not None:
			ax.get_zaxis().set_ticklabels(z_ticklabels)
	except AttributeError:
		import warnings
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''z_ticklabels'' is disregarded.", RuntimeWarning
		)

	# axes tick labels color
	if x_ticklabelcolor != '':
		ax.tick_params(axis='x', colors=x_ticklabelcolor)
	if y_ticklabelcolor != '':
		ax.tick_params(axis='y', colors=y_ticklabelcolor)
	try:
		if z_ticklabelcolor != '':
			ax.tick_params(axis='z', colors=z_ticklabelcolor)
	except AttributeError:
		import warnings
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''z_ticklabelcolor'' is disregarded.", RuntimeWarning
		)

	# unlabelled axes ticks
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
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''zaxis_minor_ticks_visible'' is disregarded.",
			RuntimeWarning
		)

	# axes visibility
	if not xaxis_visible:
		ax.get_xaxis().set_visible(False)
	if not yaxis_visible:
		ax.get_yaxis().set_visible(False)
	try:
		if not zaxis_visible:
			ax.get_zaxis().set_visible(False)
	except AttributeError:
		import warnings
		warnings.warn(
			"The plot is not three-dimensional, therefore the "
			"argument ''zaxis_visible'' is disregarded.", RuntimeWarning
		)

	# legend
	if legend_on:
		if legend_bbox_to_anchor is None:
			ax.legend(loc=legend_loc, framealpha=legend_framealpha, ncol=legend_ncol)
		else:
			ax.legend(
				loc=legend_loc, framealpha=legend_framealpha, ncol=legend_ncol,
				bbox_to_anchor=legend_bbox_to_anchor
			)

	# text box
	if text is not None:
		ax.add_artist(AnchoredText(text, loc=text_locations[text_loc]))

	# plot grid
	if grid_on:
		gps = grid_properties if grid_properties is not None else {}
		ax.grid(True, **gps)


def reverse_colormap(cmap, name=None):
	"""
	Reverse a Matplotlib colormap.

	Parameters
	----------
	cmap : matplotlib.colors.LinearSegmentedColormap
		The colormap to invert.
	name : `str`, optional
		The name of the reversed colormap. By default, this is obtained by
		appending '_r' to the name of the input colormap.

	Return
	------
	matplotlib.colors.LinearSegmentedColormap :
		The reversed colormap.

	References
	----------
	https://stackoverflow.com/questions/3279560/invert-colormap-in-matplotlib.
	"""
	keys = []
	reverse = []

	for key in cmap._segmentdata:
		# extract the channel
		keys.append(key)
		channel = cmap._segmentdata[key]

		# reverse the channel
		data = []
		for t in channel:
			data.append((1-t[0], t[2], t[1]))
		reverse.append(sorted(data))

	# set the name of the reversed map
	if name is None:
		name = cmap.name + '_r'

	return LinearSegmentedColormap(name, dict(zip(keys, reverse)))


def set_colorbar(
	fig, mappable, color_levels, *, cbar_ticks_step=1, cbar_ticks_pos='center',
	cbar_title='', cbar_x_label='', cbar_y_label='', cbar_orientation='vertical',
	cbar_ax=None
):
	"""
	Ease the configuration of the colorbar in Matplotlib plots.

	Parameters
	----------
	fig : matplotlib.figure.Figure
		The figure containing the plot.
	mappable : mappable
		The mappable, i.e., the image, which the colorbar applies
	color_levels : array_like
		1-D array of the levels corresponding to the colorbar colors.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the colorbar. Defaults to 1,
		i.e., all ticks are displayed with the corresponding label.
	cbar_ticks_pos : str
		'center' to place the colorbar ticks in the middle of the color intervals,
		anything else to place the ticks at the interfaces between color intervals.
	cbar_x_label : str
		Label for the horizontal axis of the colorbar. Defaults to an empty string.
	cbar_y_label : str
		Label for the vertical axis of the colorbar. Defaults to an empty string.
	cbar_title : str
		Colorbar title. Defaults to an empty string.
	cbar_orientation : str
		Colorbar orientation. Either 'vertical' (default) or 'horizontal'.
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
			cb = plt.colorbar(
				mappable, orientation=cbar_orientation, ax=[axes[i] for i in cbar_ax]
			)
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
	Plot a line.

	Parameters
	----------
	x : numpy.ndarray
		1-D array gathering the x-coordinates of the points to plot.
	y : numpy.ndarray
		1-D array gathering the y-coordinates of the points to plot.
	ax : matplotlib.axes.Axes
		The axes embodying the plot.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Defaults to 16.
	x_factor : float
		Scaling factor for the x-coordinates. Defaults to 1.
	y_factor : float
		Scaling factor for the y-coordinates. Defaults to 1.
	linestyle : str
		String specifying the line style. Defaults to '-'.
	linewidth : float
		The line width. Defaults to 1.5.
	linecolor : str
		String specifying the line color. Defaults to 'blue'.
	marker : str
		The shape of the markers. If not given, no markers will be displayed.
	markersize : float
		Marker size. Defaults to 5.
	markeredgewidth : str
		Marker edge width. Defaults to 1.5.
	markerfacecolor : str
		Marker face color. Defaults to 'blue'.
	markeredgecolor : str
		Marker edge color. Defaults to 'blue'.
	legend_label : str
		The legend label for the line. Defaults to an empty string.
	"""
	# get keyword arguments
	fontsize     	= kwargs.get('fontsize', 16)
	x_factor     	= kwargs.get('x_factor', 1.)
	y_factor     	= kwargs.get('y_factor', 1.)
	linestyle    	= kwargs.get('linestyle', '-')
	linewidth    	= kwargs.get('linewidth', 1.5)
	linecolor	 	= kwargs.get('linecolor', 'blue')
	marker  		= kwargs.get('marker', None)
	markersize  	= kwargs.get('markersize', 5)
	markeredgewidth = kwargs.get('markeredgewidth', 1.5)
	markerfacecolor = kwargs.get('markerfacecolor', 'blue')
	markeredgecolor = kwargs.get('markeredgecolor', 'blue')
	legend_label 	= kwargs.get('legend_label', '')

	# global settings
	rcParams['font.size'] = fontsize

	# rescale the axes for visualization purposes
	x *= x_factor
	y *= y_factor

	# plot
	if legend_label == '' or legend_label is None:
		ax.plot(
			x, y, color=linecolor, linestyle=linestyle, linewidth=linewidth,
			marker=marker, markersize=markersize, markeredgewidth=markeredgewidth,
			markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor
		)
	else:
		ax.plot(
			x, y, color=linecolor, linestyle=linestyle, linewidth=linewidth,
			marker=marker, markersize=markersize, markeredgewidth=markeredgewidth,
			markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor,
			label=legend_label
		)

	# bring axes back to original units
	x /= x_factor
	y /= y_factor


def make_contour(x, y, field, ax, **kwargs):
	"""
	Generate a contour plot.

	Parameters
	----------
	x : numpy.ndarray
		2-D array gathering the x-coordinates of the grid points.
	y : numpy.ndarray
		2-D array gathering the y-coordinates of the grid points.
	field : numpy.ndarray
		2-D array representing the field to plot.
	ax : matplotlib.axes.Axes
		The axes embodying the plot.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used in the plot. Defaults to 16.
	x_factor : float
		Scaling factor for the x-coordinates. Defaults to 1.
	y_factor : float
		Scaling factor for the y-coordinates. Defaults to 1.
	field_bias : float
		Bias for the field, so that the contour lines for `field - field_bias`
		are drawn. Defaults to 0.
	field_factor : float
		Scaling factor for the field, so that the contour lines for
		`field_factor * field` are drawn. If a bias is specified, then the contour
		lines for `field_factor * (field - field_bias)` are drawn. Defaults to 1.
	alpha : float
		Transparency of the contour lines. It should be between 0 and 1;
		defaults to 1.
	colors : str, sequence[str]
		Contour lines colors. Defaults to 'black'.
	draw_vertical_levels : bool
		:obj:`True` to draw the underlying vertical levels, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	"""
	# shortcuts
	ni, nk = field.shape

	# get keyword arguments
	fontsize     = kwargs.get('fontsize', 16)
	x_factor     = kwargs.get('x_factor', 1.)
	y_factor     = kwargs.get('y_factor', 1.)
	field_bias	 = kwargs.get('field_bias', 0.)
	field_factor = kwargs.get('field_factor', 1.)
	alpha		 = kwargs.get('alpha', 1.0)
	colors		 = kwargs.get('colors', 'black')
	draw_vlevels = kwargs.get('draw_vertical_levels', False)

	# global settings
	rcParams['font.size'] = fontsize

	# rescale the axes and the field for visualization purposes
	x     *= x_factor
	y     *= y_factor
	field -= field_bias
	field *= field_factor

	# plot the vertical levels
	if draw_vlevels:
		for k in range(nk):
			ax.plot(x[:, k], y[:, k], color='gray', linewidth=1, alpha=0.5)

	# plot the field
	if field.min() != field.max():
		ax.contour(x, y, field, colors=colors, alpha=alpha)

	# bring axes and field back to original units
	x     /= x_factor
	y 	  /= y_factor
	field /= field_factor
	field += field_bias


def make_contourf(x, y, field, fig, ax, **kwargs):
	"""
	Generate a contourf plot.

	Parameters
	----------
	x : numpy.ndarray
		2-D array gathering the x-coordinates of the grid points.
	y : numpy.ndarray
		2-D array gathering the y-coordinates of the grid points.
	field : numpy.ndarray
		2-D array representing the field to plot.
	fig : matplotlib.figure.Figure
		The figure embodying the figure.
	ax : matplotlib.axes.Axes
		The axes embodying the plot.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Defaults to 16.
	x_factor : float
		Scaling factor for the x-coordinates. Defaults to 1.
	y_factor : float
		Scaling factor for the y-coordinates. Defaults to 1.
	field_bias : float
		Bias for the field, so that the contourf plot for `field - field_bias`
		is generated. Defaults to 0.
	field_factor : float
		Scaling factor for the field, so that the contourf plot for
		`field_factor * field` is generated. If a bias is specified, then the
		contourf plot for `field_factor * (field - field_bias)` is generated.
		Defaults to 1.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps provided
		by Matplotlib, as well as the corresponding inverted versions, are available.
	cbar_on : bool
		:obj:`True` to show the colorbar, :obj:`False` otherwise.
		Defaults to :obj:`True`.
	cbar_levels : int
		Number of levels for the colorbar. Defaults to 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the colorbar.
		Defaults to 1, i.e., all ticks are displayed with the corresponding label.
	cbar_ticks_pos : str
		'center' to place the colorbar ticks in the middle of the color intervals,
		anything else to place the ticks at the interfaces between color intervals.
	cbar_center : float
		Center of the range covered by the colorbar. By default, the colorbar covers
		the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the colorbar. By default, the colorbar
		covers the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the colorbar. Defaults to an empty string.
	cbar_y_label : str
		Label for the vertical axis of the colorbar. Defaults to an empty string.
	cbar_title : str
		Colorbar title. Defaults to an empty string.
	cbar_orientation : str
		Colorbar orientation. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the colorbar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the colorbar. If no indices are given,
		only the current axes are resized.
	draw_vertical_levels : bool
		:obj:`True` to draw the underlying vertical levels, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	"""
	# get keyword arguments
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
	draw_vlevels 	 	 = kwargs.get('draw_vertical_levels', False)

	# global settings
	rcParams['font.size'] = fontsize

	# rescale the axes and the field for visualization purposes
	x     *= x_factor
	y     *= y_factor
	field -= field_bias
	field *= field_factor

	# create colorbar for colormap
	field_min, field_max = np.amin(field), np.amax(field)
	if cbar_center is None:
		cbar_lb, cbar_ub = field_min, field_max
	else:
		half_width = max(cbar_center-field_min, field_max-cbar_center) \
			if cbar_half_width is None else cbar_half_width
		cbar_lb, cbar_ub = cbar_center-half_width, cbar_center+half_width
	color_levels = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint=True)
	if eq(color_levels[0], color_levels[-1]):
		color_levels = np.linspace(cbar_lb-1e-8, cbar_ub+1e-8, cbar_levels, endpoint=True)

	# create colormap
	if cmap_name == 'BuRd':
		cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
	elif cmap_name == 'BuYlRd':
		cm = reverse_colormap(plt.get_cmap('RdYlBu'), 'BuYlRd')
	elif cmap_name == 'CMRmap_r':
		cm = reverse_colormap(plt.get_cmap('CMRmap'), 'CMRmap_r')
	else:
		cm = plt.get_cmap(cmap_name)

	# plot the vertical levels
	if draw_vlevels:
		for k in range(x.shape[1]):
			ax.plot(x[:, k], y[:, k], color='gray', linewidth=1, alpha=0.5)

	# plot the field
	surf = ax.contourf(x, y, field, color_levels, cmap=cm, extend="both")

	# set the colorbar
	if cbar_on:
		set_colorbar(
			fig, surf, color_levels,
			cbar_ticks_step=cbar_ticks_step, cbar_ticks_pos=cbar_ticks_pos,
			cbar_title=cbar_title, cbar_x_label=cbar_x_label, cbar_y_label=cbar_y_label,
			cbar_orientation=cbar_orientation, cbar_ax=cbar_ax
		)

	# bring axes and field back to original units
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
	x : numpy.ndarray
		2-D array gathering the x-coordinates of the grid points.
	y : numpy.ndarray
		2-D array gathering the y-coordinates of the grid points.
	vx : numpy.ndarray
		2-D array representing the x-component of the vector field to plot.
	vy : numpy.ndarray
		2-D array representing the y-component of the vector field to plot.
	scalar : numpy.ndarray
		2-D array representing a scalar field associated with the vector field.
		The arrows will be colored based on the associated scalar value.
		If :obj:`None`, the arrows will be colored based on their magnitude.
	fig : matplotlib.figure.Figure
		The figure encapsulating the plot.
	ax : matplotlib.axes.Axes
		The axes embodying the plo.

	Keyword arguments
	-----------------
	fontsize : int
		The fontsize to be used. Defaults to 12.
	x_factor : float
		Scaling factor for the x-coordinates. Defaults to 1.
	x_step : int
		Distance between the x-index of a given drawn point, and the
		x-index of its closest drawn neighbour. Defaults to 2, i.e.,
		only half of the points will be drawn.
	y_factor : float
		Scaling factor for the y-coordinates. Defaults to 1.
	y_step : int
		Distance between the y-index of a given drawn point, and the
		y-index of its closest drawn neighbour. Defaults to 2, i.e.,
		only half of the points will be drawn.
	scalar_bias : float
		Bias for the scalar field, so that the arrows will be colored
		based on `scalar - scalar_bias`. Defaults to 0.
	scalar_factor : float
		Scaling factor for the scalar field, so that the arrows will be
		colored based on `scalar_factor * scalar`. If a bias is specified,
		then the arrows will be colored based on
		`scalar_factor * (scalar - scalar_bias)` are drawn. Defaults to 1.
	arrow_scale : float
		Arrows scale factor.
	arrow_scale_units : str
		Arrows scale units.
	arrow_headwidth : float
		Arrows head width.
	cmap_name : str
		Name of the Matplotlib's color map to be used. All the color maps
		provided by Matplotlib, as well as the corresponding inverted versions,
		are available. If not specified, no color map will be used, and the
		arrows will draw black.
	cbar_on : bool
		:obj:`True` to show the colorbar, :obj:`False` otherwise.
		Defaults to :obj:`True`.
	cbar_levels : int
		Number of levels for the colorbar. Defaults to 14.
	cbar_ticks_step : int
		Distance between two consecutive labelled ticks of the colorbar.
		Defaults to 1, i.e., all ticks are displayed with the corresponding label.
	cbar_ticks_pos : str
		'center' to place the colorbar ticks in the middle of the color intervals,
		anything else to place the ticks at the interfaces between color intervals.
	cbar_center : float
		Center of the range covered by the colorbar. By default, the colorbar covers
		the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_half_width : float
		Half-width of the range covered by the colorbar. By default, the colorbar
		covers the spectrum ranging from the minimum to the maximum assumed by the field.
	cbar_x_label : str
		Label for the horizontal axis of the colorbar. Defaults to an empty string.
	cbar_y_label : str
		Label for the vertical axis of the colorbar. Defaults to an empty string.
	cbar_title : str
		Colorbar title. Defaults to an empty string.
	cbar_orientation : str
		Colorbar orientation. Either 'vertical' (default) or 'horizontal'.
	cbar_ax : tuple
		Indices of the figure axes from which space for the colorbar axes
		is stolen. If multiple indices are given, the corresponding axes are
		all evenly resized to make room for the colorbar. If no indices are given,
		only the current axes are resized.
	quiverkey_on : bool
		:obj:`True` to show the quiver key box, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	quiverkey_loc : tuple
		2-item tuple specifying the location in normalized figure coordinates
		of the quiver key box.
	quiverkey_length : float
		Quiver key length.
	quiverkey_label : str
		Quiver key label. Defaults to '1 m s$^{-1}$'.
	quiverkey_label_loc : str
		Quiver key label location. Either 'N', 'S', 'W' or 'E'.
	quiverkey_color : str
		Quiver key color. Defaults to 'black'.
	quiverkey_fontproperties : dict
		Dictionary specifying font settings for the quiver key label.
	draw_vertical_levels : bool
		:obj:`True` to draw the underlying vertical levels, :obj:`False` otherwise.
		Defaults to :obj:`False`.
	"""
	# get keyword arguments
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
	quiverkey_color		= kwargs.get('quiverkey_color', None)
	quiverkey_fontprops = kwargs.get('quiverkey_fontproperties', {})
	draw_vlevels 	 	= kwargs.get('draw_vertical_levels', False)

	# global settings
	rcParams['font.size'] = fontsize

	# rescale the axes and the field for visualization purposes
	x *= x_factor
	y *= y_factor
	if scalar is not None:
		scalar -= scalar_bias
		scalar *= scalar_factor

	if cmap_name is not None:
		# create colorbar for colormap
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

		# create colormap
		if cmap_name == 'BuRd':
			cm = reverse_colormap(plt.get_cmap('RdBu'), 'BuRd')
		elif cmap_name == 'BuYlRd':
			cm = reverse_colormap(plt.get_cmap('RdYlBu'), 'BuYlRd')
		elif cmap_name == 'CMRmap_r':
			cm = reverse_colormap(plt.get_cmap('CMRmap'), 'CMRmap_r')
		else:
			cm = plt.get_cmap(cmap_name)
	else:
		cm = None

	# plot the vertical levels
	if draw_vlevels:
		for k in range(x.shape[1]):
			ax.plot(x[:, k], y[:, k], color='gray', linewidth=1, alpha=0.5)

	# generate quiver-plot
	if cm is None:
		q = ax.quiver(
			x[::x_step, ::y_step], y[::x_step, ::y_step],
			vx[::x_step, ::y_step], vy[::x_step, ::y_step],
			scale=arrow_scale, scale_units=arrow_scale_units,
			headwidth=arrow_headwidth
		)
	else:
		q = ax.quiver(
			x[::x_step, ::y_step], y[::x_step, ::y_step],
			vx[::x_step, ::y_step], vy[::x_step, ::y_step],
			scalar[::x_step, ::y_step], cmap=cm,
			scale=arrow_scale, scale_units=arrow_scale_units,
			headwidth=arrow_headwidth
		)

	# set the colorbar
	if cm is not None and cbar_on:
		set_colorbar(
			fig, q, color_levels,
			cbar_ticks_step=cbar_ticks_step,
			cbar_ticks_pos=cbar_ticks_pos, cbar_title=cbar_title,
			cbar_x_label=cbar_x_label, cbar_y_label=cbar_y_label,
			cbar_orientation=cbar_orientation, cbar_ax=cbar_ax
		)

	# set quiver key
	if quiverkey_on:
		qk = ax.quiverkey(
			q, quiverkey_loc[0], quiverkey_loc[1],
			quiverkey_length, quiverkey_label,
			coordinates='axes', labelpos=quiverkey_label_loc,
			fontproperties=quiverkey_fontprops,
		)
		if quiverkey_color is not None:
			qk.text.set_backgroundcolor(quiverkey_color)

	# bring axes and field back to original units
	x /= x_factor
	y /= y_factor
	if scalar is not None:
		scalar /= scalar_factor
		scalar += scalar_bias


def make_circle(ax, **kwargs):
	"""
	Draw a circle.

	Parameters
	----------
	ax : matplotlib.axes.Axes
		The axes embodying the plot.

	Keyword arguments
	-----------------
	xy : tuple
		2-item tuple storing the coordinates of the circle center.
		Defaults to (0.0, 0.0).
	radius : float
		The circle radius.
	linewidth : float
		Edge width. Defaults to 1.0.
	edgecolor : str
		Edge color. Defaults to 'black'.
	facecolor : str
		Face color. Defaults to 'white'.
	"""
	xy 		  = kwargs.get('xy', (0.0, 0.0))
	radius 	  = kwargs.get('radius', 1.0)
	linewidth = kwargs.get('linewidth', 1.0)
	edgecolor = kwargs.get('edgecolor', 'black')
	facecolor = kwargs.get('facecolor', 'white')

	circ = patches.Circle(
		xy, radius, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor
	)
	ax.add_patch(circ)


def make_rectangle(ax, **kwargs):
	"""
	Draw a rectangle.

	Parameters
	----------
	ax : matplotlib.axes.Axes
		The axes embodying the plot.

	Keyword arguments
	-----------------
	xy : tuple
		2-item tuple storing the coordinates of the bottom left
		corner of the rectangle.
	width : float
		Rectangle width. Defaults to 1.0.
	height : float
		Rectangle height. Defaults to 1.0.
	angle : float
		Rotation angle. Defaults to 0.0.
	linewidth : float
		Edge width. Defaults to 1.0.
	edgecolor : str
		Edge color. Defaults to 'black'.
	facecolor : str
		Face color. Defaults to 'white'.
	"""
	xy 		  = kwargs.get('xy', (0.0, 0.0))
	width 	  = kwargs.get('width', 1.0)
	height 	  = kwargs.get('height', 1.0)
	angle 	  = kwargs.get('angle', 0.0)
	linewidth = kwargs.get('linewidth', 1.0)
	edgecolor = kwargs.get('edgecolor', 'black')
	facecolor = kwargs.get('facecolor', 'white')

	rect = patches.Rectangle(
		xy, width, height, angle=angle, linewidth=linewidth,
		edgecolor=edgecolor, facecolor=facecolor
	)
	ax.add_patch(rect)
