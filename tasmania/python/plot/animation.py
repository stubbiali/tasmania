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
	get_time
	Animation
"""
import matplotlib.animation as manimation

from tasmania.python.plot.monitors import PlotComposite


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


class Animation:
	"""
	This class creates an animation by leveraging a wrapped
	:class:`~tasmania.Plot` or :class:`~tasmania.PlotComposite`
	to generate the frames.
	"""
	def __init__(self, artist, print_time=None, init_time=None, fps=15):
		"""
		Parameters
		----------
		artist : `tasmania.Plot, tasmania.PlotComposite`
			The *artist* actually generating the frames.
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
		fps : `int`, optional
			Frames per second. Defaults to 15.
		"""
		# store input arguments as private attributes
		self._artist = artist
		self._ptime = print_time
		self._itime = init_time
		self._fps = fps

		# ensure the artist is in non-interactive mode
		self._artist.interactive = False

		# initialize the list of states
		self._states = []

	def store(self, *states):
		"""
		Append a new state (respectively, a list of states), to the list of
		states (resp., lists of states) stored in this object.

		Parameters
		----------
		states : dict or list
			A model state dictionary, or a list of model state dictionaries.
		"""
		self._states.append(states)

	def reset(self):
		"""
		Empty the list of stored states.
		"""
		self._states = []

	def run(self, save_dest):
		"""
		Generate the animation based on the list of states stored in this object.

		Parameters
		----------
		save_dest : str
			Path to the location where the movie should be saved.
			The path should include the format extension.
		"""
		nt = len(self._states)
		if nt == 0:
			import warnings
			warnings.warn(
				"This object does not contain any model state, "
				"so no movie will be created."
			)
			return

		# instantiate writer class
		ffmpeg_writer = manimation.writers['ffmpeg']
		metadata = {'title': ''}
		writer = ffmpeg_writer(fps=self._fps, metadata=metadata)

		# retrieve the figure object from the artist
		fig = self._artist.figure

		# save initial time
		if self._ptime == 'elapsed' and self._itime is None:
			self._itime = get_time(self._states[0])

		with writer.saving(fig, save_dest, nt):
			for n in range(nt):
				# clean the canvas
				fig.clear()

				# get current time
				if self._ptime in ('elapsed', 'absolute'):
					time = get_time(self._states[n])

				# get the string with the time
				if self._ptime == 'elapsed':
					time_str = str(time - self._itime)
				elif self._ptime == 'absolute':
					time_str = str(time)
				else:
					time_str = None

				# update artist(s)'s properties
				if time_str is not None:
					if isinstance(self._artist, PlotComposite):
						for subplot_artist in self._artist.artists:
							subplot_artist.axes_properties['title_right'] = time_str
					else:
						self._artist.axes_properties['title_right'] = time_str

				# create the frame
				_ = self._artist.store(*self._states[n], fig=fig, show=False)

				# let the writer grab the frame
				writer.grab_frame()
