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
import abc
import warnings

class Tendency:
	"""
	Abstract base class whose derived classes implement different parameterization schemes, providing tendencies.

	Note
	----
	All the derived classes should be model-agnostic.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		"""
		self._grid = grid

	@property
	def time_levels(self):
		"""
		Get the attribute representing the number of time levels the dynamical core relies on.
		
		Return
		------
		int :
			The number of time levels the dynamical core relies on.
		"""
		if self._time_levels is None:
			warn_msg = '''The attribute representing the number of time levels the underlying dynamical core relies on ''' \
					   '''has not been previously set, so it is tacitly assumed it is 1.''' \
					   '''If you want to manually set it, please use the ''time_levels'' property.'''
			warnings.warn(warn_msg, RuntimeWarning)
			self._time_levels = 1
		return self._time_levels

	@time_levels.setter
	def time_levels(self, value):
		"""
		Set the attribute representing the number of time levels the dynamical core relies on.

		Parameters
		----------
		value : int
			The number of time levels the dynamical core relies on.
		"""
		self._time_levels = value

	@abc.abstractmethod
	def __call__(self, dt, state):
		"""
		Entry-point method applying the parameterization scheme.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the timestep.
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the current state.

		Return
		------
		tendencies : obj
			:class:`~storages.grid_data.GridData` storing the output tendencies.
		diagnostics : obj
			:class:`~storages.grid_data.GridData` storing possible output diagnostics.
		"""

class TendencyMicrophysics(Tendency):
	"""
	Abstract base class whose derived classes implement different parameterization schemes providing 
	microphysical tendencies.
	"""
	pass
