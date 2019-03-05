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
	ConservativeIsentropicCoriolis
"""
import numpy as np
from sympl import TendencyComponent

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class ConservativeIsentropicCoriolis(TendencyComponent):
	"""
	This class calculates the Coriolis forcing term for the
	isentropic velocity momenta.
	"""
	def __init__(self, grid, coriolis_parameter=None, dtype=datatype, **kwargs):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			TODO
		coriolis_parameter : `dataarray_like`, optional
			TODO
		dtype : `obj`, optional
			TODO
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.TendencyComponent`.
		"""
		self._grid = grid

		super().__init__(**kwargs)

		self._f = coriolis_parameter.to_units('rad s^-1').values.item() \
				  if coriolis_parameter is not None \
			  	  else 1e-4

		self._out_su = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
		self._out_sv = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)

	@property
	def input_properties(self):
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		g = self._grid
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
		self._out_su[3:-3, 3:-3, :] = self._f * state['y_momentum_isentropic'][3:-3, 3:-3, :]
		self._out_sv[3:-3, 3:-3, :] = - self._f * state['x_momentum_isentropic'][3:-3, 3:-3, :]

		tendencies = {
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}

		diagnostics = {}

		return tendencies, diagnostics
