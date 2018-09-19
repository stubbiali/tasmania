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
import matplotlib.pyplot as plt

from tasmania.plot.assemblers import SubplotsAssembler
from tasmania.utils import utils


class Animation:
	"""
	This class creates an animation by sequentially invoking a wrapped
	:class:`~tasmania.plot.plot_monitors.Plot1d`,
	:class:`~tasmania.plot.plot_monitors.Plot2d`,
	:class:`~tasmania.plot.assemblers.PlotsOverlapper`,
	or :class:`~tasmania.plot.plot_monitors.SubplotsAssembler` object
	on a list of model states, and grabbing the so-generated frames.

	Attributes
	----------
    fontsize : int
        The fontsize to be used.
    figsize : tuple
        The size which the figure should have.
    tight_layout : bool
        :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
    print_time : str
        String specifying if time should be printed above the plot,
        flush with the right edge. Options are:

            * 'elapsed', to print the time elapsed from the first snapshot stored;
            * 'absolute', to print the absolute time of the snapshot.
            * anything else, not to print anything.

    fps : int
        Frames per second.
	"""
	def __init__(self, artist, fontsize=16, figsize=(8, 8), tight_layout=True,
				 print_time=None, fps=15):
		"""
		The constructor.

		Parameters
		----------
		artist : artist
			Instance of :class:`~tasmania.plot.plot_monitors.Plot1d`,
			:class:`~tasmania.plot.plot_monitors.Plot2d`,
			:class:`~tasmania.plot.assemblers.PlotsOverlapper`,
			or :class:`~tasmania.plot.assemblers.SubplotsAssembler`.
		fontsize : `int`, optional
			The fontsize to be used. Default is 16.
		figsize : `tuple`, optional
			The size which the figure should have. Default is (8, 8).
		tight_layout : `bool`, optional
            :obj:`True` to fit plot to the figure, :obj:`False` otherwise.
            Default is :obj:`True`.
		print_time : str
			String specifying if time should be printed above the plot,
			flush with the right edge. Options are:

				* 'elapsed', to print the time elapsed from the first snapshot stored;
				* 'absolute', to print the absolute time of the snapshot.
				* anything else, not to print anything.

			Default is :obj:`None`.
		fps : int
			Frames per second. Default is 15.
		"""
		# Store input arguments as private attributes
		self._artist 	 = artist

		# Store input arguments as public attributes
		self.fontsize	  = fontsize
		self.figsize	  = figsize
		self.tight_layout = tight_layout
		self.print_time	  = print_time
		self.fps		  = fps

		# Ensure the artist is in non-interactive mode
		self._artist.interactive = False

		# Initialize the list of states
		self._states = []

	def store(self, states):
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
			warnings.warn('This object does not contain any model state, so no movie will be created.')
			return

		# Instantiate writer class
		ffmpeg_writer = manimation.writers['ffmpeg']
		metadata = {'title': ''}
		writer = ffmpeg_writer(fps=self.fps, metadata=metadata)

		# Instantiate the figure object
		fig = plt.figure(figsize=self.figsize)

		# Save initial time
		try:
			init_time = self._states[0]['time']
		except TypeError:
			init_time = self._states[0][0]['time']

		with writer.saving(fig, save_dest, nt):
			for n in range(nt):
				# Clean the canvas
				fig.clear()

				# Get current time
				try:
					time = self._states[n]['time']
				except TypeError:
					time = self._states[n][0]['time']

				# Get the string with the time
				if self.print_time == 'elapsed':
					time_str = str(utils.convert_datetime64_to_datetime(time) -
								   utils.convert_datetime64_to_datetime(init_time))
				elif self.print_time == 'absolute':
					time_str = str(utils.convert_datetime64_to_datetime(time))
				else:
					time_str = None

				# Update artist(s)'s properties
				if time_str is not None:
					if isinstance(self._artist, SubplotsAssembler):
						for subplot_artist in self._artist.artists:
							subplot_artist.plot_properties['title_right'] = time_str
					else:
						self._artist.plot_properties['title_right'] = time_str

				# Create the frame
				fig, _ = self._artist.store(self._states[n], fig=fig, show=False)

				# Set the frame layout
				if self.tight_layout:
					fig.tight_layout()

				# Let the writer grab the frame
				writer.grab_frame()
