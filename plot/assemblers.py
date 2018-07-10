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
from matplotlib import rcParams
import matplotlib.pyplot as plt
import os

from tasmania.plot.plot_monitors import Plot1d, Plot2d, Plot3d
from tasmania.utils import plot_utils
from tasmania.utils import utils


class PlotsAssembler:
	"""
	This class creates a plot by merging the individual plots created by
	different artists. An artist may be either a
	:class:`~tasmania.plot.plot_monitors.Plot1d`,
	:class:`~tasmania.plot.plot_monitors.Plot2d`, or
	:class:`~tasmania.plot.plot_monitors.Plot3d` object.

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
    fontsize : int
        The fontsize to be used.
    figsize : tuple
        The size which the figure should have.
    tight_layout : bool
        :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
    plot_properties : dict
        Plot properties which are forwarded to and thereby set by
        :func:`~tasmania.utils.plot_utils.set_plot_properties`.
	"""
	def __init__(self, artists, interactive=True, fontsize=16, figsize=(8, 8),
				 tight_layout=True, plot_properties=None):
		"""
		The constructor.

		Parameters
		----------
		artists : list
			List of instances of either :class:`~tasmania.plot.plot_monitors.Plot1d` 
			or :class:`~tasmania.plot.plot_monitors.Plot1d`.
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise. 
			Default is :obj:`True`.
		fontsize : `int`, optional
			The fontsize to be used. Default is 16.
		figsize : `tuple`, optional
			The size which the figure should have. Default is (8, 8).
		tight_layout : `bool`, optional
            :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
            Default is :obj:`True`.
		plot_properties : `dict`, optional
			Plot properties which are forwarded to and thereby set by
			:func:`~tasmania.utils.plot_utils.set_plot_properties`.
			Default is :obj:`None`.
		"""
		# Assert the list of artists
		utils.assert_sequence(artists, reftype=(Plot1d, Plot2d, Plot3d))

		# Input arguments stored as private attributes
		self._artists = artists

		# Input arguments stored as public attributes
		self.interactive 	 = interactive
		self.fontsize   	 = fontsize
		self.figsize    	 = figsize
		self.tight_layout	 = tight_layout
		self.plot_properties = {} if not isinstance(plot_properties, dict) \
							   else plot_properties

		# Initialize the figure attribute
		self._figure = None

	@property
	def projection(self):
		"""
		Return the projection used by the artists.

		Return
		------
		str :
			The projection used by each artist.

		Raises
		------
		RuntimeError :
			If the artists use different projections.
		AttributeError :
			If the artists do not use any projection, i.e., if they are
			:class:`~tasmania.plot.plot_monitors.Plot1d` objects.
		"""
		# The internal list consists of only one artist
		if len(self._artists) == 1:
			return self._artists[0].projection

		# If the internal lists consists of Plot1d objects,
		# an AttributeError is raised here
		first_artist_projection = self._artists[0].projection

		for artist in self._artists[1:]:
			if artist.projection != first_artist_projection:
				raise RuntimeError('All artists should use the same projection.')

		return first_artist_projection

	def store(self, states, fig=None, ax=None, save_dest=None, show=False):
		"""
		Update the plot by invoking each artist on the corresponding state
		and properly merging the plots.

		Parameters
		----------
		states : list
			List of dictionary model states. :obj:`states[i]` is forwarded
			to the :obj:`i`-th artist.
		fig : `figure`, optional
			A :class:`matplotlib.pyplot.figure`.
		ax : `axes`, optional
			An instance of :class:`matplotlib.axes.Axes`.
		save_dest : `str`, optional
			Path to the location where the figure should be saved.
			The path should include the extension of the figure.
			If :obj:`None` or empty, the plot will not be saved.
		show : `bool`, optional
			When the non-interactive mode is switched on, :obj:`True` to show the figure, 
			:obj:`False` otherwise. Default is :obj:`False`.

		Note
		----
		Figure titles and axes labels set by an artist do not overwrite the titles and labels
		set by any previous artist.

		Return
		------
		out_fig : figure
			The :class:`matplotlib.pyplot.figure` containing the plot.
		out_ax : axes
			The :class:`matplotlib.axes.Axes` enclosing the plot.
		"""
		# Assert the list of states
		utils.assert_sequence(states, reftype=dict, reflen=len(self._artists))

		# Set the figure attribute, if necessary
		self._set_figure(fig)

		# Set the output figure and axes
		try:
			out_fig, out_ax = plot_utils.get_figure_and_axes(
				fig, ax, default_fig=self._figure,
				figsize=self.figsize, fontsize=self.fontsize,
				projection=self.projection)
		except AttributeError:  # The artists are Plot1d objects
			out_fig, out_ax = plot_utils.get_figure_and_axes(
				fig, ax, default_fig=self._figure,
				figsize=self.figsize, fontsize=self.fontsize)

		# Clean-up the canvas, if needed
		if ax is None:
			out_ax.cla()

		for artist_i, (artist, state) in enumerate(zip(self._artists, states)):
			# Get the current axes limits
			if artist_i > 0:
				xmin_now, xmax_now = out_ax.get_xlim()
				ymin_now, ymax_now = out_ax.get_ylim()

			# Call an artist from the list
			out_fig, out_ax = artist.store(state, fig=out_fig, ax=out_ax, show=False)

			# Get the new axes limits
			xmin_new, xmax_new = out_ax.get_xlim()
			ymin_new, ymax_new = out_ax.get_ylim()

			# Set the axes limits
			if artist_i > 0:
				out_ax.set_xlim([min(xmin_now, xmin_new), max(xmax_now, xmax_new)])
				out_ax.set_ylim([min(ymin_now, ymin_new), max(ymax_now, ymax_new)])
		
		# Set plot properties
		plot_utils.set_plot_properties(out_ax, **self.plot_properties)

		# Set the layout
		if fig is None and self.tight_layout:
			out_fig.tight_layout()

		# Save
		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		# Show
		if not self.interactive and show:
			plt.show()

		return out_fig, out_ax

	def _set_figure(self, fig=None):
		if fig is not None:
			self._figure = None
			return

		if self.interactive:
			plt.ion()
			if self._figure is not None:
				rcParams['font.size'] = self.fontsize
				self._figure = plt.figure(figsize=self.figsize)
		else:
			plt.ioff()
			rcParams['font.size'] = self.fontsize
			self._figure = plt.figure(figsize=self.figsize)


class SubplotsAssembler:
	"""
	This class creates a visualization consisting of different subplots,
	with each subplot generated by a
	:class:`~tasmania.plot.plot_monitors.Plot1d`,
	:class:`~tasmania.plot.plot_monitors.Plot2d`,
	:class:`~tasmania.plot.plot_monitors.Plot3d`, or
	:class:`~tasmania.plot.assemblers.PlotsAssembler` object.

	Attributes
	----------
    artists : sequence
        Sequence of :class:`~tasmania.plot.plot_monitors.Plot1d`,
        :class:`~tasmania.plot.plot_monitors.Plot2d`,
        :class:`~tasmania.plot.plot_monitors.Plot3d`, or
        :class:`~tasmania.plot.assemblers.PlotsAssembler` objects.
    interactive : bool
        :obj:`True` to enable interactive plotting, :obj:`False` otherwise.
    fontsize : int
        The fontsize to be used.
    figsize : tuple
        The size which the figure should have.
    tight_layout : bool
        :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
	"""
	def __init__(self, nrows, ncols, artists, interactive=True,
				 fontsize=16, figsize=(8, 8), tight_layout=True):
		"""
		The constructor.

		Parameters
		----------
		nrows : int
			Number of rows of the subplot grid.
		ncols : int
			Number of columns of the subplot grid.
		artists : sequence
			Sequence of :class:`~tasmania.plot.plot_monitors.Plot1d`,
			:class:`~tasmania.plot.plot_monitors.Plot2d`,
			:class:`~tasmania.plot.plot_monitors.Plot3d`, or
			:class:`~tasmania.plot.assemblers.PlotsAssembler` objects.
		interactive : `bool`, optional
			:obj:`True` to enable interactive plotting, :obj:`False` otherwise.
			Default is :obj:`True`.
		fontsize : `int`, optional
			The fontsize to be used. Default is 16.
		figsize : `tuple`, optional
			The size which the figure should have. Default is (8, 8).
		tight_layout : `bool`, optional
            :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
            Default is :obj:`True`.
		"""
		# Check input artists list
		utils.assert_sequence(artists, reflen=nrows*ncols,
							  reftype=(Plot1d, Plot2d, Plot3d, PlotsAssembler))

		# Store input arguments as private attributes
		self._nrows	= nrows
		self._ncols	= ncols

		# Store input arguments as public attributes
		self.artists      = artists
		self.interactive  = interactive
		self.fontsize     = fontsize
		self.figsize      = figsize
		self.tight_layout = tight_layout

		# Initialize the figure attribute
		self._figure = None

	def store(self, states, fig=None, save_dest=None, show=False):
		"""
		Update the plot with the given list of states.

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
		utils.assert_sequence(states, reflen=len(self.artists))

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
			rcParams['font.size'] = self.fontsize

			try:
				ax = plt.subplot(self._nrows, self._ncols, index,
								 projection=artist.projection)
			except AttributeError:  # The artist is a Plot1d object
				ax = plt.subplot(self._nrows, self._ncols, index)

			out_fig, _ = artist.store(state, fig=out_fig, ax=ax, show=False)

			index += 1

		# Set the layout
		if fig is None and self.tight_layout:
			out_fig.tight_layout()

		# Save
		if not (save_dest is None or save_dest == ''):
			_, ext = os.path.splitext(save_dest)
			plt.savefig(save_dest, format=ext[1:], dpi=1000)

		# Show
		if not self.interactive and show:
			plt.show()

		return out_fig

	def _set_figure(self, fig=None):
		if fig is not None:
			self._figure = None
			return

		if self.interactive:
			plt.ion()
			if self._figure is not None:
				rcParams['font.size'] = self.fontsize
				self._figure = plt.figure(figsize=self.figsize)
		else:
			plt.ioff()
			rcParams['font.size'] = self.fontsize
			self._figure = plt.figure(figsize=self.figsize)
