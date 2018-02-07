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
import numpy as np
from sympl import Prognostic

class DynamicalCore(Prognostic):
	"""
	Abstract base class whose derived classes implement different dynamical cores.
	The class inherits :class:`sympl.Prognostic`.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid):
		"""
		Constructor.

		Parameters
		----------
			grid : obj 
				The underlying grid, as an instance of :class:`~grids.xyz_grid.XYZGrid` or one of its derived classes.
		"""
		self._grid = grid

	@abc.abstractmethod
	def __call__(self, dt, state):
		"""
		Call operator advancing the input state one step forward. 
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
			dt : obj
				:class:`datetime.timedelta` object representing the time step.
			state : obj 
				The current state, as an instance of :class:`~storages.grid_data.GridData` or one of its derived classes.

		Return
		------
			obj :
				The state at the next time level. This is of the same class of :data:`state`.
		"""

	@abc.abstractmethod
	def get_initial_state(self, *args):
		"""
		Get the initial state.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
			*args 
				The arguments depend on the specific dynamical core which the derived class implements.

		Return
		------
			obj :
				The initial state, as an instance of :class:`~storages.grid_data.GridData` or one of its derived classes.
		"""

	def update_topography(self, time):
		"""
		Update the underlying (time-dependent) topography.

		Parameters
		----------
			time : obj
				:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self._grid.update_topography(time)
