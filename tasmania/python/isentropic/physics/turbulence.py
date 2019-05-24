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
	IsentropicSmagorinsky
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.diagnostics import HorizontalVelocity
from tasmania.python.physics.turbulence import Smagorinsky2d

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float64


class IsentropicSmagorinsky(Smagorinsky2d):
	"""
	Implementation of the Smagorinsky turbulence model for the
	isentropic model. The conservative form of the governing
	equations is used.
	"""
	def __init__(
		self, domain, grid_type='numerical', smagorinsky_constant=0.18,
		backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		smagorinsky_constant : `float`, optional
			The Smagorinsky constant. Defaults to 0.18.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.python.physics.turbulence.Smagorinsky2d`.
		"""
		super().__init__(
			domain, grid_type, smagorinsky_constant, backend, dtype, **kwargs
		)

		self._hv = HorizontalVelocity(
			self.grid, staggering=False, backend=backend, dtype=dtype
		)

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._out_su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
		return {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'}
		}

	@property
	def tendency_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
		return {
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'}
		}

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		s = state['air_isentropic_density']
		su = state['x_momentum_isentropic']
		sv = state['y_momentum_isentropic']

		self._hv.get_velocity_components(s, su, sv, self._in_u, self._in_v)

		self._stencil.compute()

		self._hv.get_momenta(
			s, self._out_u_tnd, self._out_v_tnd, self._out_su_tnd, self._out_sv_tnd
		)

		tendencies = {
			'x_momentum_isentropic': self._out_su_tnd,
			'y_momentum_isentropic': self._out_sv_tnd
		}
		diagnostics = {}

		return tendencies, diagnostics
