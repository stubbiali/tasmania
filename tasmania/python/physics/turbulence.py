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
from tasmania.python.utils.storage_utils import get_storage_descriptor

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float64


class Smagorinsky2d(TendencyComponent):
	"""
	Implementation of the Smagorinsky turbulence model for a
	two-dimensional flow.
	The class is instantiated over the *numerical* grid of the
	underlying domain.

	References
	----------
	Rösler, M. (2015). *The Smagorinsky turbulence model.* Master thesis, \
		Freie Universität Berlin.
	"""
	def __init__(
		self, domain, smagorinsky_constant=0.18, *,
		backend='numpy', backend_opts=None, build_info=None, dtype=datatype,
		exec_info=None, halo=None, rebuild=False, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		smagorinsky_constant : `float`, optional
			The Smagorinsky constant. Defaults to 0.18.
		backend : `str`, optional
			TODO
		backend_opts : `dict`, optional
			TODO
		build_info : `dict`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		exec_info : `dict`, optional
			TODO
		halo : `tuple`, optional
			TODO
		rebuild : `bool`, optional
			TODO
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.
		"""
		super().__init__(domain, 'numerical', **kwargs)

		self._cs = smagorinsky_constant
		self._exec_info = exec_info

		assert self.horizontal_boundary.nb >= 2, \
			'The number of boundary layers must be greater or equal than two.'

		self._nb = max(2, self.horizontal_boundary.nb)

		storage_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
		descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo)
		self._in_u = gt.storage.zeros(descriptor, backend=backend)
		self._in_v = gt.storage.zeros(descriptor, backend=backend)
		self._out_u_tnd = gt.storage.zeros(descriptor, backend=backend)
		self._out_v_tnd = gt.storage.zeros(descriptor, backend=backend)

		decorator = gt.stencil(
			backend, backend_opts=backend_opts, build_info=build_info,
			rebuild=rebuild
		)
		self._stencil = decorator(self._stencil_defs)

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
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		nb = self._nb
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()

		self._in_u.data[...] = state['x_velocity'][...]
		self._in_v.data[...] = state['y_velocity'][...]

		self._stencil(
			in_u=self._in_u, in_v=self._in_v, out_u_tnd=self._out_u_tnd,
			out_v_tnd=self._out_v_tnd, dx=dx, dy=dy, cs=self._cs,
			origin={'_all_': (nb, nb, 0)}, domain=(nx-2*nb, ny-2*nb, nz),
			exec_info=self._exec_info
		)

		tendencies = {
			'x_velocity': self._out_u_tnd.data,
			'y_velocity': self._out_v_tnd.data
		}
		diagnostics = {}

		return tendencies, diagnostics

	@staticmethod
	def _stencil_defs(
		in_u: gt.storage.f64_ijk_sd,
		in_v: gt.storage.f64_ijk_sd,
		out_u_tnd: gt.storage.f64_ijk_sd,
		out_v_tnd: gt.storage.f64_ijk_sd,
		*,
		dx: float,
		dy: float,
		cs: float
	):
		s00 = (in_u[+1, 0, 0] - in_u[-1, 0, 0]) / (2.0 * dx)
		s01 = 0.5 * (
			(in_u[0, +1, 0] - in_u[0, -1, 0]) / (2.0 * dy) +
			(in_v[+1, 0, 0] - in_v[-1, 0, 0]) / (2.0 * dx)
		)
		s11 = (in_v[0, +1, 0] - in_v[0, -1, 0]) / (2.0 * dy)
		nu = cs**2 * dx * dy * \
			(2.0 * (s00[0, 0, 0]**2 + 2.0 * s01[0, 0, 0]**2 + s11[0, 0, 0]**2))**0.5
		out_u_tnd = 2.0 * (
			(nu[+1, 0, 0] * s00[+1, 0, 0] - nu[-1, 0, 0] * s00[-1, 0, 0]) / (2.0 * dx) +
			(nu[0, +1, 0] * s01[0, +1, 0] - nu[0, -1, 0] * s01[0, -1, 0]) / (2.0 * dy)
		)
		out_v_tnd = 2.0 * (
			(nu[+1, 0, 0] * s01[+1, 0, 0] - nu[-1, 0, 0] * s01[-1, 0, 0]) / (2.0 * dx) +
			(nu[0, +1, 0] * s11[0, +1, 0] - nu[0, -1, 0] * s11[0, -1, 0]) / (2.0 * dy)
		)
