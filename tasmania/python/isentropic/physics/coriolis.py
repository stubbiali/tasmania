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
	IsentropicConservativeCoriolis
"""
import numpy as np

import gridtools as gt
from tasmania.python.framework.base_components import TendencyComponent

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class IsentropicConservativeCoriolis(TendencyComponent):
	"""
	Calculate the Coriolis forcing term for the isentropic velocity momenta.
	"""
	def __init__(
		self, domain, grid_type='numerical', coriolis_parameter=None,
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

		coriolis_parameter : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing the Coriolis
			parameter, in units compatible with [rad s^-1].
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.TendencyComponent`.
		"""
		super().__init__(domain, grid_type, **kwargs)

		f = coriolis_parameter.to_units('rad s^-1').values.item() \
			if coriolis_parameter is not None else 1e-4
		self._f = gt.Global(f)

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._in_su = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_sv = np.zeros((nx, ny, nz), dtype=dtype)
		self._tnd_su = np.zeros((nx, ny, nz), dtype=dtype)
		self._tnd_sv = np.zeros((nx, ny, nz), dtype=dtype)

		nb = 0 if grid_type == 'physical' else self.horizontal_boundary.nb

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={'in_su': self._in_su, 'in_sv': self._in_sv},
			global_inputs={'f': self._f},
			outputs={'tnd_su': self._tnd_su, 'tnd_sv': self._tnd_sv},
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=backend
		)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		self._in_su[...] = state['x_momentum_isentropic'][...]
		self._in_sv[...] = state['y_momentum_isentropic'][...]

		self._stencil.compute()

		tendencies = {
			'x_momentum_isentropic': self._tnd_su,
			'y_momentum_isentropic': self._tnd_sv,
		}

		diagnostics = {}

		return tendencies, diagnostics

	@staticmethod
	def _stencil_defs(f, in_su, in_sv):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		tnd_su = gt.Equation()
		tnd_sv = gt.Equation()

		tnd_su[i, j] = f * in_sv[i, j]
		tnd_sv[i, j] = - f * in_su[i, j]

		return tnd_su, tnd_sv
