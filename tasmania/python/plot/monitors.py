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
This module contain:
	get_time
	Plot(Monitor)
	PlotComposite
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
import os
from sympl import Monitor

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.plot_utils import \
	get_figure_and_axes, set_axes_properties, set_figure_properties
from tasmania.python.plot.utils import assert_sequence


SequenceType = (tuple, list)


def get_time(states):
	for level0 in states:
		if isinstance(level0, dict):  # level0 is a state dictionary
			if 'time' in level0:
				return level0['time']
		else:  # level0 is a collection of state dictionaries
			for level1 in level0:
				if 'time' in level1:
					return level1['time']

	raise ValueError('No state dictionary contains the key ''time''.')


class Plot(Monitor):
	"""
	A :class:`sympl.Monitor` for visualization purposes, generating a
	plot by nicely overlapping distinct plots drawn by one or multiple
	:class:`tasmania.Drawer`\s.

	Warning
	-------
	No consistency/coherency controls are performed on the list of artists.
	For instance, the composer does not check that all the artists use
	the same units for the axes. Ultimately, it is up to the user to ensure
	that everything is coherent.

	Attributes
	----------
	interactive : bool
		:obj:`True` if interactive plotting is enabled,
		:obj:`False` otherwise.
	figure_properties : dict
		Keyword arguments specifying settings for the
		:class:`~matplotlib.figure.Figure` containing the plot.
	axes_properties : dict
		Keyword arguments specifying settings for the
		:class:`~matplotlib.axes.Axes` enclosing the plot.
	"""
	def __init__(
		self, *drawers, interactive=True, print_time=None, init_time=None,
		figure_properties=None, axes_properties=None
	):
		"""
		Parameters
		----------
		drawers : tasmania.Drawer
			The drawer(s) actually drawing the plot(s).
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		print_time : `str`, optional
			String specifying if time should be printed above the plot,
			flush with the right edge. Available options are:

				* 'elapsed', to print the time elapsed since `init_time`;
				* 'absolute', to print the absolute time of the snapshot;
				* anything else, not to print anything.

			Defaults to :obj:`None`.
		init_time : `datetime`, optional
			The initial time of the simulation. Only effective if `print_time`
			is 'elapsed'. If not specified, the elapsed time is calculated
			with respect to the first passed state.
		figure_properties : `dict`, optional
			Keyword arguments specifying settings for the figure containing
			the plot. To be broadcast to
			:func:`~tasmania.get_figure_and_axes_properties`
			and :func:`~tasmania.set_figure_properties`.
		axes_properties : `dict`, optional
			Keyword arguments specifying settings for the axes enclosing
			the plot. To be broadcast to
			:func:`~tasmania.get_figure_and_axes_properties`
			and :func:`~tasmania.set_axes_properties`.
		"""
		assert_sequence(drawers, reftype=Drawer)
		self._artists = drawers

		self.interactive = interactive

		self._ptime = print_time
		self._itime = init_time

		self.figure_properties = {} if figure_properties is None else figure_properties
		self.axes_properties = {} if axes_properties is None else axes_properties

		self._figure = None

	@property
	def artists(self):
		"""
		Returns
		-------
		tuple :
			The artists.
		"""
		return self._artists

	@property
	def figure(self):
		"""
		Returns
		-------
		matplotlib.figure.Figure :
			The figure used and *owned* by this object.
		"""
		self._set_figure()
		return self._figure

	def store(self, *states, fig=None, ax=None, save_dest=None, show=False):
		"""
		Use the input state(s) to update the plot.

		Parameters
		----------
		states : dict, sequence[dict]
			A model state dictionary, or a sequence of model state
			dictionaries, feeding the artists. This means that the
			i-th item in the sequence will be forwarded to the i-th artist.
		fig : `matplotlib.figure.Figure`, optional
			The figure which should contain the plot.
			If not given, the internal figure is used.
		ax : `matplotlib.axes.Axes`, optional
			The axes which should enclose the plot.
			If not given, the internal axes are used.
		save_dest : `str`, optional
			Path under which the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on,
			:obj:`True` to show the figure, :obj:`False` otherwise.
			Defaults to :obj:`False`.

		Return
		------
		out_fig : matplotlib.figure.Figure
			The figure containing the plot.
		out_ax : matplotlib.axes.Axes
			The axes enclosing the plot.
		"""
		assert_sequence(states, reflen=len(self._artists), reftype=dict)

		# set the private _figure attribute
		self._set_figure(fig)

		# retrieve figure and axes
		out_fig, out_ax = get_figure_and_axes(
			fig, ax, default_fig=self._figure,
			**self.figure_properties,
			**{
				key: value for key, value in self.axes_properties.items()
				if key not in self.figure_properties
			},
		)

		# if needed, clean the canvas
		if ax is None:
			out_ax.cla()

		# save initial time
		if self._ptime == 'elapsed' and self._itime is None:
			self._itime = get_time(states)

		# let the drawers draw
		for drawer, state in zip(self._artists, states):
			drawer(state, out_fig, out_ax)

		# set axes properties
		if self.axes_properties != {}:
			time = get_time(states)

			if self._ptime == 'elapsed':
				time_str = str(time - self._itime)
			elif self._ptime == 'absolute':
				time_str = str(time)
			else:
				time_str = None

			if time_str is not None:
				self.axes_properties['title_right'] = time_str

			set_axes_properties(out_ax, **self.axes_properties)

		# if figure is not provided, set figure properties
		if fig is None and self.figure_properties != {}:
			set_figure_properties(out_fig, **self.figure_properties)

		# save
		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		# show
		if fig is None:
			if self.interactive:
				out_fig.canvas.draw()
				plt.show(block=False)
			elif show:
				plt.show()

		return out_fig, out_ax

	def _set_figure(self, fig=None):
		"""
		Set the private attribute representing the figure
		*owned* by this object.
		"""
		if fig is not None:
			self._figure = None
			return

		fontsize = self.figure_properties.get('fontsize', 12)
		figsize  = self.figure_properties.get('figsize', (7, 7))

		if self.interactive:
			plt.ion()
			if self._figure is not None:
				rcParams['font.size'] = fontsize
				self._figure = plt.figure(figsize=figsize)
		else:
			plt.ioff()
			rcParams['font.size'] = fontsize
			self._figure = \
				plt.figure(figsize=figsize) if self._figure is None \
				else self._figure


class PlotComposite:
	"""
	This class creates a visualization consisting of different subplots,
	with each subplot generated by a :class:`~tasmania.Plot`.

	Attributes
	----------
    figure_properties : dict
		Keyword arguments specifying settings for the figure containing
		the plot. To be broadcast to
		:func:`~tasmania.get_figure_and_axes_properties`
		and :func:`~tasmania.set_figure_properties`.
	"""
	def __init__(
		self, *artists, nrows=1, ncols=1, interactive=True, 
		print_time=None, init_time=None, figure_properties=None
	):
		"""
		Parameters
		----------
		artists : sequence
			The artists, each generating a single subplot.
			With respect to the subplot grid, row-major ordering is assumed.
		nrows : `int`, optional
			Number of rows of the subplot grid. Defaults to 1.
		ncols : `int`, optional
			Number of columns of the subplot grid. Defaults to 1.
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		print_time : `str`, optional
			String specifying if time should be printed as suptitle.
			Available options are:

				* 'elapsed', to print the time elapsed since `init_time`;
				* 'absolute', to print the absolute time of the snapshot;
				* anything else, not to print anything.

			Defaults to :obj:`None`.
		init_time : `datetime`, optional
			The initial time of the simulation. Only effective if `print_time`
			is 'elapsed'. If not specified, the elapsed time is calculated
			with respect to the first passed state.
    	figure_properties : `dict`, optional
			Keyword arguments specifying settings for the figure containing
			the plot. To be broadcast to
			:func:`~tasmania.get_figure_and_axes_properties`
			and :func:`~tasmania.set_figure_properties`.
		"""
		# check input artists list
		assert_sequence(artists, reftype=Plot)

		# store input arguments as private attributes
		self._artists 	  = artists
		self._nrows	  	  = nrows
		self._ncols	  	  = ncols
		self._interactive = interactive
		self._ptime       = print_time
		self._itime       = init_time

		# store input arguments as public attributes
		self.figure_properties = \
			{} if figure_properties is None else figure_properties

		# initialize the figure attribute
		self._figure = None

	@property
	def artists(self):
		"""
		Returns
		-------
		tuple :
			The artists.
		"""
		return self._artists

	@property
	def figure(self):
		"""
		Returns
		-------
		matplotlib.figure.Figure :
			The figure used and *owned* by this object.
		"""
		self._set_figure()
		return self._figure

	@property
	def interactive(self):
		"""
		Returns
		-------
		bool :
			:obj:`True` if interactive model is enabled, :obj:`False` otherwise.
		"""
		return self._interactive

	@interactive.setter
	def interactive(self, value):
		"""
		Switch interactive mode on/off.

		Parameters
		----------
		value : bool
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
		"""
		self._interactive = value
		for artist in self.artists:
			artist.interactive = value

	def store(self, *states, fig=None, save_dest=None, show=False):
		"""
		Use the input states to update the plot.

		Parameters
		----------
		states : sequence
			Sequence whose items can be:

			 	* dictionary states, or
			 	* a sequence of dictionary states.

			The i-th item will be forwarded to the i-th artist.
		fig : `matplotlib.figure.Figure`, optional
			The figure encapsulating all the subplots.
		save_dest : `str`, optional
			Path to the location where the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on,
			:obj:`True` to show the figure, :obj:`False` otherwise.
			Defaults :obj:`False`.

		Return
		------
		matplotlib.figure.Figure
			The figure encapsulating all the subplots.
		"""
		# assert the list of states
		assert_sequence(
			states, reflen=len(self.artists), reftype=SequenceType + (dict, )
		)

		# set the figure attribute, if needed
		self._set_figure(fig)

		# set output figure
		out_fig = fig if fig is not None else self._figure

		# clear the canvas, if needed
		if fig is None:
			out_fig.clear()

		# get the initial time
		if self._ptime == 'elapsed' and self._itime is None:
			self._itime = get_time(states)

		# generate all the subplot axes
		axes = []
		for i in range(len(self.artists)):
			out_fig, ax = get_figure_and_axes(
				out_fig, nrows=self._nrows, ncols=self._ncols,
				index=i+1, **self.figure_properties,
				**{
					key: val for key, val in self.artists[i].axes_properties.items()
					if key not in self.figure_properties
				},
			)
			axes.append(ax)

		for i in range(len(self.artists)):
			artist = self.artists[i]
			state = states[i]
			ax = axes[i]

			if isinstance(state, dict):
				out_fig, _ = artist.store(state, fig=out_fig, ax=ax, show=False)
			else:
				out_fig, _ = artist.store(*state, fig=out_fig, ax=ax, show=False)

		time = get_time(states)

		if self._ptime == 'elapsed':
			time_str = str(time - self._itime)
		elif self._ptime == 'absolute':
			time_str = str(time)
		else:
			time_str = None

		if time_str is not None:
			self.figure_properties['suptitle'] = time_str

		# set figure properties
		set_figure_properties(out_fig, **self.figure_properties)

		# save
		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		# show
		if fig is None:
			if self.interactive:
				out_fig.canvas.draw()
				plt.show(block=False)
			elif show:
				plt.show()

		return out_fig

	def _set_figure(self, fig=None):
		"""
		Set the private attribute representing the figure
		*owned* by this object.
		"""
		if fig is not None:
			self._figure = None
			return

		fontsize = self.figure_properties.get('fontsize', 12)
		figsize  = self.figure_properties.get('figsize', (7, 7))

		if self.interactive:
			plt.ion()
			if self._figure is None:
				rcParams['font.size'] = fontsize
				self._figure = plt.figure(figsize=figsize)
		else:
			plt.ioff()
			rcParams['font.size'] = fontsize
			self._figure = plt.figure(figsize=figsize)
