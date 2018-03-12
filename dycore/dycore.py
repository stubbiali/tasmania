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
from sympl import TimeStepper

class DynamicalCore(TimeStepper):
	"""
	Abstract base class whose derived classes implement different dynamical cores.
	The class inherits :class:`sympl.TimeStepper`.

	Attributes
	----------
	microphysics : obj
		Derived class of :class:`~parameterizations.microphysics.Microphysics` taking care of the cloud microphysics.
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
		# Set the underlying grid
		self._grid = grid

		# Initialize pointer to the object taking care of the microphysics
		self.microphysics = None

	@abc.abstractmethod
	def __call__(self, dt, state, diagnostics = None):
		"""
		Call operator advancing the input state one step forward. 
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` object representing the time step.
		state : obj 
			The current state, as an instance of :class:`~storages.grid_data.GridData` or one of its derived classes.
		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing the possibly required diagnostics. Default is :obj:`None`.

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
		*args :
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

	@staticmethod
	def factory(model, *args, **kwargs):
		"""
		Static method returning an instance of the derived class implementing the dynamical core specified by :obj:`model`.

		Parameters
		----------
		model : str
			String specifying the dynamical core to implement. Either:

			* 'isentropic', for the hydrostatic, isentropic dynamical core;
			* 'isentropic_isothermal', for the hydrostatic, isentropic, isothermal dynamical core.

		*args :
			Positional arguments to forward to the derived class.
		**kwargs :
			Keyword arguments to forward to the derived class.
			
		Return
		------
		obj :
			Instance of the derived class implementing the specified model.
		"""
		if model == 'isentropic':
			from dycore.dycore_isentropic import DynamicalCoreIsentropic
			return DynamicalCoreIsentropic(*args, **kwargs)
		elif model == 'isentropic_isothermal':
			from dycore.dycore_isentropic_isothermal import DynamicalCoreIsentropicIsothermal
			return DynamicalCoreIsentropicIsothermal(*args, **kwargs)
