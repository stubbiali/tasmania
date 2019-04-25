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
	IsentropicNonconservativePressureGradient
	IsentropicConservativePressureGradient
"""
import numpy as np

import gridtools as gt
from tasmania.python.framework.base_components import TendencyComponent

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class IsentropicNonconservativePressureGradient(TendencyComponent):
	"""
	Calculate the anti-gradient of the Montgomery potential, which provides
	tendencies for the x- and y-velocity in the isentropic system.
	The class is always instantiated over the numerical grid of the
	underlying domain.
	"""
	def __init__(
		self, domain, order, backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		order : int
			The order of the finite difference formula used to
			discretize the gradient of the Montgomery potential.
			Available options are:

				* '2', for a second-order centered formula;
				* '4', for a fourth-order centered formula.

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.
		"""
		#
		# call parent's constructor
		#
		super().__init__(domain, 'numerical', **kwargs)

		#
		# initialize the pointer to the underlying GT4Py stencil calculating
		# the pressure gradient, i.e., the anti-gradient of the Montgomery potential
		#
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		hb = self.horizontal_boundary

		if order == 2:
			nb = 1  # TODO: nb = 1 if hb.nb < 1 else hb.nb
			stencil_defs = self._stencil_second_order_defs
		elif order == 4:
			nb = 2  # TODO: nb = 2 if hb.nb < 2 else hb.nb
			stencil_defs = self._stencil_fourth_order_defs
		else:
			raise RuntimeError(
				"The order should be either 2 or 4, but {} given.".format(order)
			)

		assert nx > 2*nb, \
			"Number of grid points along the first horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, nx)
		assert ny > 2*nb, \
			"Number of grid points along the second horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, ny)

		self._in_mtg = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_u_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_v_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		self._stencil = gt.NGStencil(
			definitions_func=stencil_defs,
			inputs={'in_mtg': self._in_mtg},
			outputs={'out_u_tnd': self._out_u_tnd, 'out_v_tnd': self._out_v_tnd},
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=backend
		)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'x_velocity':
				{'dims': dims, 'units': 'm s^-2'},
			'y_velocity':
				{'dims': dims, 'units': 'm s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		# update the numpy arrays serving as stencil inputs
		self._in_mtg[...] = state['montgomery_potential']

		# run the stencil
		self._stencil.compute()

		# instantiate the return dictionary
		tendencies = {'x_velocity': self._out_u_tnd, 'y_velocity': self._out_v_tnd}

		return tendencies, {}

	def _stencil_second_order_defs(self, in_mtg):
		# shortcuts
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate the output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# define the computations
		out_u_tnd[i, j] = (- in_mtg[i+1, j] + in_mtg[i-1, j]) / (2. * dx)
		out_v_tnd[i, j] = (- in_mtg[i, j+1] + in_mtg[i, j-1]) / (2. * dy)

		return out_u_tnd, out_v_tnd

	def _stencil_fourth_order_defs(self, in_mtg):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate the output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# define the computations
		out_u_tnd[i, j] = \
			(- in_mtg[i-2, j] + 8. * in_mtg[i-1, j]
			 - 8. * in_mtg[i+1, j] + in_mtg[i+2, j]) / (12. * dx)
		out_v_tnd[i, j] = \
			(- in_mtg[i, j-2] + 8. * in_mtg[i, j-1]
			 - 8. * in_mtg[i, j+1] + in_mtg[i, j+2]) / (12. * dy)

		return out_u_tnd, out_v_tnd


class IsentropicConservativePressureGradient(TendencyComponent):
	"""
	Calculate the anti-gradient of the Montgomery potential, multiplied by
	the air isentropic density. This quantity provides tendencies for the
	x- and y-momentum in the isentropic system. The class is always instantiated
	over the numerical grid of the underlying domain.
	"""
	def __init__(
		self, domain, order, backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		order : int
			The order of the finite difference formula used to
			discretize the gradient of the Montgomery potential.
			Available options are:

				* '2', for a second-order centered formula;
				* '4', for a fourth-order centered formula.

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.
		"""
		#
		# call parent's constructor
		#
		super().__init__(domain, 'numerical', **kwargs)

		#
		# initialize the pointer to the underlying GT4Py stencil calculating
		# the pressure gradient, i.e., the anti-gradient of the Montgomery potential
		#
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		hb = self.horizontal_boundary

		if order == 2:
			nb = 1  # TODO: nb = 1 if hb.nb < 1 else hb.nb
			stencil_defs = self._stencil_second_order_defs
		elif order == 4:
			nb = 2  # TODO: nb = 2 if hb.nb < 2 else hb.nb
			stencil_defs = self._stencil_fourth_order_defs
		else:
			raise RuntimeError(
				"The order should be either 2 or 4, but {} given.".format(order)
			)

		assert nx > 2*nb, \
			"Number of grid points along the first horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, nx)
		assert ny > 2*nb, \
			"Number of grid points along the second horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, ny)

		self._in_s = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_mtg = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		self._stencil = gt.NGStencil(
			definitions_func=stencil_defs,
			inputs={'in_s': self._in_s, 'in_mtg': self._in_mtg},
			outputs={'out_su_tnd': self._out_su_tnd, 'out_sv_tnd': self._out_sv_tnd},
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=backend
		)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_isentropic_density':
				{'dims': dims, 'units': 'kg m^-2 K^-1'},
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'x_momentum_isentropic':
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic':
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		# update the numpy arrays serving as stencil inputs
		self._in_s[...] = state['air_isentropic_density']
		self._in_mtg[...] = state['montgomery_potential']

		# run the stencil
		self._stencil.compute()

		# instantiate the return dictionary
		tendencies = {
			'x_momentum_isentropic': self._out_su_tnd,
			'y_momentum_isentropic': self._out_sv_tnd,
		}

		return tendencies, {}

	def _stencil_second_order_defs(self, in_s, in_mtg):
		# shortcuts
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate the output fields
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# define the computations
		out_su_tnd[i, j] = in_s[i, j] * (- in_mtg[i+1, j] + in_mtg[i-1, j]) / (2. * dx)
		out_sv_tnd[i, j] = in_s[i, j] * (- in_mtg[i, j+1] + in_mtg[i, j-1]) / (2. * dy)

		return out_su_tnd, out_sv_tnd

	def _stencil_fourth_order_defs(self, in_s, in_mtg):
		# shortcuts
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate the output fields
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# define the computations
		out_su_tnd[i, j] = \
			in_s[i, j] * (
				- in_mtg[i-2, j] + 8. * in_mtg[i-1, j]
				- 8. * in_mtg[i+1, j] + in_mtg[i+2, j]
			) / (12. * dx)
		out_sv_tnd[i, j] = \
			in_s[i, j] * (
				- in_mtg[i, j-2] + 8. * in_mtg[i, j-1]
				- 8. * in_mtg[i, j+1] + in_mtg[i, j+2]
			) / (12. * dy)

		return out_su_tnd, out_sv_tnd
