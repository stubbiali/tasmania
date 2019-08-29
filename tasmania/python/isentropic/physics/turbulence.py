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
from tasmania.python.utils.storage_utils import get_storage_descriptor

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float64


class IsentropicSmagorinsky(Smagorinsky2d):
	"""
	Implementation of the Smagorinsky turbulence model for the
	isentropic model. The conservative form of the governing
	equations is used.
	The class is instantiated over the *numerical* grid of the
	underlying domain.
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
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

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
			:class:`~tasmania.python.physics.turbulence.Smagorinsky2d`.
		"""
		super().__init__(
			domain, smagorinsky_constant, backend=backend,
			backend_opts=backend_opts, build_info=build_info, dtype=dtype,
			exec_info=exec_info, halo=halo, rebuild=rebuild, **kwargs
		)

		self._hv = HorizontalVelocity(
			self.grid, staggering=False, backend=backend, backend_opts=backend_opts,
			build_info=build_info, exec_info=exec_info, rebuild=True
		)

		storage_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
		descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo)
		self._in_s = gt.storage.zeros(descriptor, backend=backend)
		self._in_su = gt.storage.zeros(descriptor, backend=backend)
		self._in_sv = gt.storage.zeros(descriptor, backend=backend)
		self._out_su_tnd = gt.storage.zeros(descriptor, backend=backend)
		self._out_sv_tnd = gt.storage.zeros(descriptor, backend=backend)

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
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		nb = self._nb
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()

		self._in_s.data[:nx, :ny, :nz]  = state['air_isentropic_density']
		self._in_su.data[:nx, :ny, :nz] = state['x_momentum_isentropic']
		self._in_sv.data[:nx, :ny, :nz] = state['y_momentum_isentropic']

		self._hv.get_velocity_components(
			self._in_s, self._in_su, self._in_sv, self._in_u, self._in_v
		)

		self._stencil(
			in_u=self._in_u, in_v=self._in_v, out_u_tnd=self._out_u_tnd,
			out_v_tnd=self._out_v_tnd, dx=dx, dy=dy, cs=self._cs,
			origin={'_all_': (nb, nb, 0)}, domain=(nx-2*nb, ny-2*nb, nz),
			exec_info=self._exec_info
		)

		self._hv.get_momenta(
			self._in_s, self._out_u_tnd, self._out_v_tnd,
			self._out_su_tnd, self._out_sv_tnd
		)

		tendencies = {
			'x_momentum_isentropic': self._out_su_tnd.data[:nx, :ny, :nz],
			'y_momentum_isentropic': self._out_sv_tnd.data[:nx, :ny, :nz]
		}
		diagnostics = {}

		return tendencies, diagnostics
