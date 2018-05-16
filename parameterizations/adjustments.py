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
from sympl import Implicit

class Adjustment(Implicit):
	"""
	Abstract base class whose derived classes implement different physical adjustment schemes.
	The hierarchy lays on top of :class:`sympl.Implicit`.

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
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		"""
		self._grid = grid
		self._time_levels = None

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
			warn_msg = """The attribute representing the number of time levels the underlying dynamical core relies on """ \
					   """has not been previously set, so it is tacitly assumed it is 1. """ \
					   """If you want to manually set it, please use the ''time_levels'' property."""
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
	def __call__(self, state, dt):
		"""
		Entry-point method applying the parameterization scheme.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		state : obj
			:class:`~tasmania.storages.grid_data.GridData` or one of its derived classes representing the current state.
		dt : obj
			:class:`datetime.timedelta` representing the timestep.

		Return
		------
		diagnostics : obj
			:class:`~tasmania.storages.grid_data.GridData` storing possible output diagnostics.
		state_new : obj
			:class:`~tasmania.storages.grid_data.GridData` storing the output, adjusted state.
		"""

class AdjustmentMicrophysics(Adjustment):
	"""
	Abstract base class whose derived classes implement different parameterization schemes carrying out 
	microphysical adjustments. The model variables which get adjusted are:

	* the mass fraction of water vapor;
	* the mass fraction of cloud liquid water;
	* the mass fraction of precipitation water.

	The derived classes also compute the following diagnostics:

	* the raindrop fall speed ([:math:`m \, s^{-1}`]).
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, rain_evaporation_on, backend):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		rain_evaporation_on : bool
			:obj:`True` if the evaporation of raindrops should be taken into account, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		"""
		super().__init__(grid)
		self._rain_evaporation_on, self._backend = rain_evaporation_on, backend

	@abc.abstractmethod
	def get_raindrop_fall_velocity(self, state):
		"""
		Get the raindrop fall velocity.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		state : obj
			:class:`~tasmania.storages.grid_data.GridData` or one of its derived classes representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).
			
		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the raindrop fall velocity.
		"""

	@staticmethod
	def factory(micro_scheme, grid, rain_evaporation_on, backend, **kwargs):
		"""
		Static method returning an instance of the derived class implementing the microphysics scheme
		specified by :obj:`micro_scheme`.

		Parameters
		----------
		micro_scheme : str
			String specifying the microphysics parameterization scheme to implement. Either:

			* 'kessler_wrf', for the WRF version of the Kessler scheme;
			* 'kessler_wrf_saturation', for the WRF version of the Kessler scheme, performing only \
				the saturation adjustment.

		grid : obj
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		rain_evaporation_on : bool
			:obj:`True` if the evaporation of raindrops should be taken into account, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		**kwargs :
			Keyword arguments to be forwarded to the derived class.
		"""
		if micro_scheme == 'kessler_wrf':
			from tasmania.parameterizations.adjustment_microphysics_kessler_wrf \
				import AdjustmentMicrophysicsKesslerWRF
			return AdjustmentMicrophysicsKesslerWRF(grid, rain_evaporation_on, backend, **kwargs)
		elif micro_scheme == 'kessler_wrf_saturation':
			from tasmania.parameterizations.adjustment_microphysics_kessler_wrf_saturation \
				import AdjustmentMicrophysicsKesslerWRFSaturation
			return AdjustmentMicrophysicsKesslerWRFSaturation(grid, rain_evaporation_on, backend, **kwargs)
		else:
			raise ValueError('Unknown microphysics parameterization scheme.')
