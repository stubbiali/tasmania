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
	Drawer
"""
import abc


class Drawer:
	"""
	This abstract base class represents a generic drawer.
	A *drawer* is a functor which uses the data grabbed
	from an input state dictionary (or a time-series of state
	dictionaries) to generate a specific plot. The figure and
	the axes encapsulating the plot should be provided, as well.

	Attributes
	----------
	properties : dict
		Dictionary whose keys are strings denoting plot-specific
		settings, and whose values specify values for those settings.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, properties=None):
		"""
		Parameters
		----------
		properties : `dict`, optional
			Dictionary whose keys are strings denoting plot-specific
			settings, and whose values specify values for those settings.
		"""
		self.properties = {} if properties is None else properties

	@abc.abstractmethod
	def __call__(self, state, fig, ax):
		"""
		Call operator generating the plot.

		Parameters
		----------
		state : dict, sequence[dict]
			Either a state or a sequence of states from which
			retrieving the data used to draw the plot.
			A state is a dictionary whose keys are strings denoting
			model variables, and values are :class:`sympl.DataArray`\s
			storing values for those variables.
		fig : matplotlib.figure.Figure
			The figure encapsulating the plot.
		ax : matplotlib.axes.Axes
			The axes encapsulating the plot
		"""
		pass
