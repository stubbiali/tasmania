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
	Animation
"""
import matplotlib.animation as manimation

from tasmania.python.plot.monitors import PlotComposite


class Animation:
	"""
	This class creates an animation by leveraging a wrapped
	:class:`~tasmania.Plot` or :class:`~tasmania.PlotComposite`
	to generate the frames.
	"""
	def __init__(self, artist, fps=15):
		"""
		Parameters
		----------
		artist : `tasmania.Plot, tasmania.PlotComposite`
			The *artist* actually generating the frames.
		fps : `int`, optional
			Frames per second. Defaults to 15.
		"""
		# store input arguments as private attributes
		self._artist = artist
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

		with writer.saving(fig, save_dest, nt):
			for n in range(nt):
				# clean the canvas
				fig.clear()

				# create the frame
				_ = self._artist.store(*self._states[n], fig=fig, show=False)

				# let the writer grab the frame
				writer.grab_frame()
