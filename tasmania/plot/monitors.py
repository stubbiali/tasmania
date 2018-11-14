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
	Plot(Monitor)
	PlotComposite
"""
from matplotlib import rcParams
import matplotlib.pyplot as plt
import os
from sympl import Monitor

from tasmania.plot.drawer import Drawer
from tasmania.plot.plot_utils import get_figure_and_axes, \
									 set_axes_properties, set_figure_properties
from tasmania.plot.utils import assert_sequence


SequenceType = (tuple, list)


class Plot(Monitor):
	"""
	A :class:`sympl.Monitor` for visualization purposes, generating a
	plot by nicely overlapping distinct plots created by one or multiple
	:class:`~tasmania.plot.drawer.Drawer`\s.

	Warning
	-------
	No consistency controls are performed on the list of artists.
	For instance, the composer does not check that all the artists use
	the same units for the axes. Ultimately, it is up to the user to ensure
	that everything is coherent.

	Attributes
	----------
    interactive : bool
        :obj:`True` to enable interactive plotting, :obj:`False` otherwise.
    figure_properties : dict
		Keyword arguments specifying properties for the
		:class:`~matplotlib.pyplot.figure` containing the plot,
		and which are forwarded and thereby set by
		:func:`~tasmania.plot.plot_utils.get_figure_and_axes_properties`
		and :func:`~tasmania.plot.plot_utils.set_figure_properties`.
    axes_properties : dict
		Keyword arguments specifying properties for the
		:class:`~matplotlib.axes.Axes` enclosing the plot,
		and which are forwarded and thereby set by
		:func:`~tasmania.plot.plot_utils.get_figure_and_axes_properties`
		and :func:`~tasmania.plot.plot_utils.set_axes_properties`.
	"""
	def __init__(self, drawers, interactive=True,
				 figure_properties=None, axes_properties=None):
		"""
		Parameters
		----------
		drawers : obj
			A :class:`~tasmania.plot.drawer.Drawer` or a sequence of
			:class:`~tasmania.plot.drawer.Drawer`\s, actually drawing the plot(s).
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Defaults to :obj:`True`.
    	figure_properties : `dict`, optional
			Keyword arguments specifying properties for the
			:class:`~matplotlib.pyplot.figure` containing the plot,
			and which are forwarded and thereby set by
			:func:`~tasmania.plot.plot_utils.get_figure_and_axes_properties`.
			and :func:`~tasmania.plot.plot_utils.set_figure_properties`.
			Defaults to :obj:`None`.
    	axes_properties : `dict`, optional
			Keyword arguments specifying properties for the
			:class:`~matplotlib.axes.Axes` enclosing the plot,
			and which are forwarded and thereby set by
			:func:`~tasmania.plot.plot_utils.set_axes_properties`.
			Defaults to :obj:`None`.
		"""
		artists = (drawers, ) if not isinstance(drawers, SequenceType) else drawers
		assert_sequence(artists, reftype=Drawer)
		self._artists = artists

		self.interactive = interactive

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
		figure :
			The :class:`matplotlib.pyplot.figure` used and *owned* by this object.
		"""
		self._set_figure()
		return self._figure

	def store(self, states, fig=None, ax=None, save_dest=None, show=False):
		"""
		Use the input state(s) to update the plot.

		Parameters
		----------
		states : dict, sequence[dict]
			A model state dictionary, or a sequence of model state dictionaries,
			feeding the artists. This means that the i-th input state will be
			forwarded to the i-th artist.
		fig : `figure`, optional
			The :class:`matplotlib.pyplot.figure` which should contain the plot.
			If not given, the internal :class:`matplotlib.pyplot.figure` is used.
		ax : `axes`, optional
			The :class:`matplotlib.axes.Axes` which should enclosing the plot.
			If not given, the internal :class:`matplotlib.axes.Axes` are used.
		save_dest : `str`, optional
			Path under which the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on,
			:obj:`True` to show the figure, :obj:`False` otherwise.
			Default is :obj:`False`.

		Return
		------
		out_fig : figure
			The :class:`matplotlib.pyplot.figure` containing the plot.
		out_ax : axes
			The :class:`matplotlib.axes.Axes` object enclosing the plot.
		"""
		states_list = (states, ) if not isinstance(states, SequenceType) else states
		assert_sequence(states_list, reflen=len(self._artists), reftype=dict)

		# Set the private _figure attribute
		self._set_figure(fig)

		# Retrieve figure and axes
		out_fig, out_ax = get_figure_and_axes(
			fig, ax, default_fig=self._figure,
			**self.figure_properties,
			**{key: value for key, value in self.axes_properties.items()
			   if key not in self.figure_properties},
		)

		# If needed, clean the canvas
		if ax is None:
			out_ax.cla()

		# Let the drawers draw
		for drawer, state in zip(self._artists, states_list):
			drawer(state, out_fig, out_ax)

		# Set plot-independent properties
		if self.axes_properties != {}:
			set_axes_properties(out_ax, **self.axes_properties)
		if self.figure_properties != {}:
			set_figure_properties(out_fig, **self.figure_properties)

		# Save
		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		# Show
		if fig is None:
			if self.interactive:
				out_fig.canvas.draw()
				plt.show(block=False)
			elif show:
				plt.show()

		return out_fig, out_ax

	def _set_figure(self, fig=None):
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
			self._figure = plt.figure(figsize=figsize) if self._figure is None \
						   else self._figure


class PlotComposite:
	"""
	This class creates a visualization consisting of different subplots,
	with each subplot generated by a :class:`~tasmania.plot.monitors.Plot`.

	Attributes
	----------
    figure_properties : dict
		Keyword arguments specifying properties for the
		:class:`~matplotlib.pyplot.figure` containing the plot,
		and which are forwarded and thereby set by
		:func:`~tasmania.plot.plot_utils.get_figure_and_axes_properties`
		and :func:`~tasmania.plot.plot_utils.set_figure_properties`.
	"""
	def __init__(self, nrows, ncols, artists, interactive=True,
				 figure_properties=None):
		"""
		The constructor.

		Parameters
		----------
		nrows : int
			Number of rows of the subplot grid.
		ncols : int
			Number of columns of the subplot grid.
		artists : sequence
			Sequence of :class:`~tasmania.plot.monitors.Plot`\s generating
			the single subplots. With respect to the subplot grid,
			row-major ordering is assumed.
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Defaults to :obj:`True`.
    	figure_properties : `dict`, optional
			Keyword arguments specifying properties for the
			:class:`~matplotlib.pyplot.figure` containing the plot,
			and which are forwarded and thereby set by
			:func:`~tasmania.plot.plot_utils.get_figure_and_axes_properties`
			and :func:`~tasmania.plot.plot_utils.set_figure_properties`.
			Defaults to :obj:`None`.
		"""
		# Check input artists list
		assert_sequence(artists, reftype=Plot)

		# Store input arguments as private attributes
		self._nrows	  = nrows
		self._ncols	  = ncols
		self._artists = artists
		self._interactive = interactive

		# Store input arguments as public attributes
		self.figure_properties = {} if figure_properties is None else figure_properties

		# Initialize the figure attribute
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
		figure :
			The :class:`matplotlib.pyplot.figure` used and *owned* by this object.
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
		"""
		self._interactive = value
		for artist in self.artists:
			artist.interactive = value

	def store(self, states, fig=None, save_dest=None, show=False):
		"""
		Use the input list of states to update the plot.

		Parameters
		----------
		states : list
			List of dictionary model states. :obj:`states[i]` is forwarded
			to the :obj:`i`-th artist.
		fig : `figure`, optional
			A :class:`matplotlib.pyplot.figure`.
		save_dest : `str`, optional
			Path to the location where the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on, :obj:`True` to show the figure,
			:obj:`False` otherwise. Default is :obj:`False`.

		Return
		------
		out_fig : figure
			The :class:`matplotlib.pyplot.figure` containing the plot.
		"""
		# Assert the list of states
		assert_sequence(states, reflen=len(self.artists), reftype=SequenceType + (dict, ))

		# Set the figure attribute, if needed
		self._set_figure(fig)

		# Set output figure
		out_fig = fig if fig is not None else self._figure

		# Clear the canvas, if needed
		if fig is None:
			out_fig.clear()

		# Generate all subplots
		index = 1
		for state, artist in zip(states, self.artists):
			out_fig, ax = get_figure_and_axes(
				out_fig, nrows=self._nrows, ncols=self._ncols,
				index=index, **self.figure_properties,
				**{key: val for key, val in artist.axes_properties.items()
				   if key not in self.figure_properties},
			)

			out_fig, _ = artist.store(state, fig=out_fig, ax=ax, show=False)

			index += 1

		# Set figure properties
		set_figure_properties(out_fig, **self.figure_properties)

		# Save
		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		# Show
		if fig is None:
			if self.interactive:
				out_fig.canvas.draw()
				plt.show(block=False)
			elif show:
				plt.show()

		return out_fig

	def _set_figure(self, fig=None):
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
