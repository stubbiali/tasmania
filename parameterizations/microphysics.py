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

import gridtools as gt

class Microphysics(Diagnostic):
	"""
	Abstract base class whose derived classes implement different microphysics parameterization schemes.
	This class inherits :class:`sympl.Diagnostic`.
	The diagnostic quantities which are computed are:

	* the raindrop fall speed ([:math:`m \, s^{-1}`]);
	* the precipitation rate ([:math:`mm \, h^{-1}`]);
	* the source term for the potential temperature involving the latent heat of condensation ([:math:`K \, s^{-1}`]);
	* the source term representing evaporation of cloud droplets [:math:`s^{-1}`];
	* the source term representing auto-conversion of cloud droplets into rain droplets [:math:`s^{-1}`];
	* the source term representing accreation of rain droplets [:math:`s^{-1}`];
	* the source term representing the evaporation of rain droplets [:math:`s^{-1}`].
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, dycore_diagnostic):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		dycore_diagnostic : obj
			Object carrying out the diagnostic steps of the underlying dynamical core.
		"""
		self._grid, self._diagnostic = grid, dycore_diagnostic

	@abc.abstractmethod
	def __call__(self, state):
		"""
		Call operator computing microphysics diagnostics at the current time level.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		state : obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.

		Returns
		-------
		diagnostic : obj
			:class:`~storages.grid_data.GridData` collecting the diagnostics.
		"""

	@staticmethod
	def factory(micro_scheme, grid, dycore_diagnostic):
		"""
		Static method returning an instance of the derived class implementing the microphysics scheme
		specified by :obj:`micro_scheme`.

		Parameters
		----------
		micro_scheme : str
			String specifying the microphysics parameterization scheme to implement. Either:

			* 'kessler', for the Kessler scheme.

		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		dycore_diagnostic : obj
			Object carrying out the diagnostic steps of the underlying dynamical core.
		"""
		if micro_scheme == 'kessler':
			return MicrophysicsKessler(grid, dycore_diagnostic)
		else:
			raise ValueError('Unknown microphysics parameterization scheme.')


class MicrophysicsKessler(Microphysics):
	"""
	This class inherits :class:`~parameterizations.microphysics.Microphysics` to implement the Kessler scheme.
	"""
	def __init__(self, grid, dycore_diagnostic):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		dycore_diagnostic : obj
			Object carrying out the diagnostic steps of the underlying dynamical core.

		Note
		----
		To instantiate this class, one should prefer the static method 
		:meth:`~parameterizations.microphysics.Microphysics.factory` of :class:`~parameterizations.microphysics.Microphysics`.
		"""
		super().__init__(grid, dycore_diagnostic)

	def __call__(self, state):
		"""
		Call operator computing microphysics diagnostics at the current time level.

		Parameters
		----------
		state : obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.

		Returns
		-------
		diagnostic : obj
			:class:`~storages.grid_data.GridData` collecting the diagnostics.
		"""
		# Extract the mixing ratio of the water constituents
		qv  = state['water_vapor'].values[:, :, :, -1]
		qc  = state['cloud_water'].values[:, :, :, -1]
		qr  = state['precipitation_water'].values[:, :, :, -1]

		# Extract or compute the density field
		if state['density'] is not None:
			rho = state['density'].values[:, :, :, -1]
		else:
			rho = self._diagnostic.get_density(state)
		
		# Express density in units of grams per cubic centimeter
		rho_gcm3 = 1.e3 * rho

		# Extract density at the surface layer
		rho_gcm3_s = np.repeat(rho_gcm3[:, :, -1:], grid.nz, axis = 2)

		# Compute fall velocity
		C1 = 36.34
		C2 = .1346
		vt = C1 * (rho_gcm3 * qr) ** C2 * np.sqrt(rho_gcm3_s / rho_gcm3)

		# Compute auto-conversion source term
		k1 = .001
		a  = .0005
		Ar = k1 * np.maximum(qc - a, 0.)

		# Compute accretion source term
		k2 = 2.2
		C3 = 0.875
		Cr = k2 * qc * (qr ** C3)


		

