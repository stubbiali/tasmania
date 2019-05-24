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
	Smagorinsky2d
"""
import numpy as np

import gridtools as gt
from tasmania.python.framework.base_components import TendencyComponent

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float64


class Smagorinsky2d(TendencyComponent):
	"""
	Implementation of the Smagorinsky turbulence model for a
	two-dimensional flow.

	References
	----------
	Rösler, M. (2015). *The Smagorinsky turbulence model.* Master thesis, \
		Freie Universität Berlin.
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
			:class:`tasmania.TendencyComponent`.
		"""
		super().__init__(domain, grid_type, **kwargs)

		self._cs = smagorinsky_constant

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		nb = max(2, self.horizontal_boundary.nb)

		self._in_u = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_v = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_u_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_v_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_u': self._in_u, 'in_v': self._in_v},
			outputs={'out_u_tnd': self._out_u_tnd, 'out_v_tnd': self._out_v_tnd},
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=backend
		)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-1'},
			'y_velocity': {'dims': dims, 'units': 'm s^-1'}
		}

	@property
	def tendency_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
		return {
			'x_velocity': {'dims': dims, 'units': 'm s^-2'},
			'y_velocity': {'dims': dims, 'units': 'm s^-2'}
		}

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		self._in_u[...] = state['x_velocity']
		self._in_v[...] = state['y_velocity']

		self._stencil.compute()

		tendencies = {'x_velocity': self._out_u_tnd, 'y_velocity': self._out_v_tnd}
		diagnostics = {}

		return tendencies, diagnostics

	def _stencil_defs(self, in_u, in_v):
		# shortcuts
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()
		cs = self._cs

		# indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary fields
		s00 = gt.Equation()
		s01 = gt.Equation()
		s11 = gt.Equation()
		nu = gt.Equation()

		# output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# computations
		s00[i, j] = (in_u[i+1, j] - in_u[i-1, j]) / (2.0 * dx)
		s01[i, j] = 0.5 * (
			(in_u[i, j+1] - in_u[i, j-1]) / (2.0 * dy) +
			(in_v[i+1, j] - in_v[i-1, j]) / (2.0 * dx)
		)
		s11[i, j] = (in_v[i, j+1] - in_v[i, j-1]) / (2.0 * dy)
		nu[i, j] = cs**2 * dx * dy * \
			(2.0 * (s00[i, j]**2 + 2.0 * s01[i, j]**2 + s11[i, j]**2))**0.5
		out_u_tnd[i, j] = 2.0 * (
			(nu[i+1, j] * s00[i+1, j] - nu[i-1, j] * s00[i-1, j]) / (2.0 * dx) +
			(nu[i, j+1] * s01[i, j+1] - nu[i, j-1] * s01[i, j-1]) / (2.0 * dy)
		)
		out_v_tnd[i, j] = 2.0 * (
			(nu[i+1, j] * s01[i+1, j] - nu[i-1, j] * s01[i-1, j]) / (2.0 * dx) +
			(nu[i, j+1] * s11[i, j+1] - nu[i, j-1] * s11[i, j-1]) / (2.0 * dy)
		)

		return out_u_tnd, out_v_tnd

