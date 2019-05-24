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
	stage_minimum
	stage_second_order
	stage_fourth_order
	stage_weighted

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


def stage_minimum(a, b):
	return (a < b) * a + (a >= b) * b


def stage_second_order(i, j, dx, dy, mtg, u_tnd, v_tnd):
	u_tnd[i, j] = (- mtg[i+1, j] + mtg[i-1, j]) / (2. * dx)
	v_tnd[i, j] = (- mtg[i, j+1] + mtg[i, j-1]) / (2. * dy)


def stage_fourth_order(i, j, dx, dy, mtg, u_tnd, v_tnd):
	u_tnd[i, j] = \
		(- mtg[i-2, j] + 8. * mtg[i-1, j]
		 - 8. * mtg[i+1, j] + mtg[i+2, j]) / (12. * dx)
	v_tnd[i, j] = \
		(- mtg[i, j-2] + 8. * mtg[i, j-1]
		 - 8. * mtg[i, j+1] + mtg[i, j+2]) / (12. * dy)


def stage_weighted(i, j, k, dx, dy, eps, mtg, p, u_tnd, v_tnd):
	tmp_dp = gt.Equation()
	dpx = gt.Equation()
	dpy = gt.Equation()
	tmp_wgtx_c = gt.Equation()
	wgtx_c = gt.Equation()
	tmp_wgtx_n = gt.Equation()
	wgtx_n = gt.Equation()
	tmp_wgtx_s = gt.Equation()
	wgtx_s = gt.Equation()
	tmp_wgtx_e = gt.Equation()
	wgtx_e = gt.Equation()
	tmp_wgtx_w = gt.Equation()
	wgtx_w = gt.Equation()
	pgx_c = gt.Equation()
	pgx_n = gt.Equation()
	pgx_s = gt.Equation()
	pgx_e = gt.Equation()
	pgx_w = gt.Equation()
	pgx_avg = gt.Equation()
	tmp_wgty_c = gt.Equation()
	wgty_c = gt.Equation()
	tmp_wgty_n = gt.Equation()
	wgty_n = gt.Equation()
	tmp_wgty_s = gt.Equation()
	wgty_s = gt.Equation()
	tmp_wgty_e = gt.Equation()
	wgty_e = gt.Equation()
	tmp_wgty_w = gt.Equation()
	wgty_w = gt.Equation()
	pgy_n = gt.Equation()
	pgy_s = gt.Equation()
	pgy_e = gt.Equation()
	pgy_w = gt.Equation()
	pgy_c = gt.Equation()
	pgy_avg = gt.Equation()

	tmp_dp[k] = p[k+1] - p[k]
	dpx[i, j] = 0.5 * (tmp_dp[i-1, j] + tmp_dp[i, j])
	dpy[i, j] = 0.5 * (tmp_dp[i, j-1] + tmp_dp[i, j])

	tmp_wgtx_c[i, j] = stage_minimum(dpx[i, j], dpx[i+1, j])
	wgtx_c[i, j] = stage_minimum(tmp_wgtx_c[i, j], eps) / eps
	tmp_wgtx_w[i, j] = stage_minimum(dpx[i-1, j], dpx[i, j])
	wgtx_w[i, j] = stage_minimum(tmp_wgtx_w[i, j], eps)
	tmp_wgtx_e[i, j] = stage_minimum(dpx[i+1, j], dpx[i+2, j])
	wgtx_e[i, j] = stage_minimum(tmp_wgtx_e[i, j], eps)
	tmp_wgtx_s[i, j] = stage_minimum(dpx[i, j-1], dpx[i+1, j-1])
	wgtx_s[i, j] = stage_minimum(tmp_wgtx_s[i, j], eps)
	tmp_wgtx_n[i, j] = stage_minimum(dpx[i, j+1], dpx[i+1, j+1])
	wgtx_n[i, j] = stage_minimum(tmp_wgtx_n[i, j], eps)

	pgx_c[i, j] = mtg[i+1, j] - mtg[i-1, j]
	pgx_w[i, j] = mtg[i, j] - mtg[i-2, j]
	pgx_e[i, j] = mtg[i+2, j] - mtg[i, j]
	pgx_s[i, j] = mtg[i+1, j-1] - mtg[i-1, j-1]
	pgx_n[i, j] = mtg[i+1, j+1] - mtg[i-1, j+1]
	pgx_avg[i, j] = \
		(wgtx_w[i, j] * pgx_w[i, j] + wgtx_e[i, j] * pgx_e[i, j] +
		 wgtx_s[i, j] * pgx_s[i, j] + wgtx_n[i, j] * pgx_n[i, j]) / \
		(wgtx_w[i, j] + wgtx_e[i, j] + wgtx_s[i, j] + wgtx_n[i, j])

	tmp_wgty_c[i, j] = stage_minimum(dpy[i, j], dpy[i, j+1])
	wgty_c[i, j] = stage_minimum(tmp_wgty_c[i, j], eps) / eps
	tmp_wgty_w[i, j] = stage_minimum(dpy[i-1, j], dpy[i-1, j+1])
	wgty_w[i, j] = stage_minimum(tmp_wgty_w[i, j], eps)
	tmp_wgty_e[i, j] = stage_minimum(dpy[i+1, j], dpy[i+1, j+1])
	wgty_e[i, j] = stage_minimum(tmp_wgty_e[i, j], eps)
	tmp_wgty_s[i, j] = stage_minimum(dpy[i, j-1], dpy[i, j])
	wgty_s[i, j] = stage_minimum(tmp_wgty_s[i, j], eps)
	tmp_wgty_n[i, j] = stage_minimum(dpy[i, j+1], dpy[i, j+2])
	wgty_n[i, j] = stage_minimum(tmp_wgty_n[i, j], eps)

	pgy_c[i, j] = mtg[i, j+1] - mtg[i, j-1]
	pgy_w[i, j] = mtg[i-1, j+1] - mtg[i-1, j-1]
	pgy_e[i, j] = mtg[i+1, j+1] - mtg[i+1, j-1]
	pgy_s[i, j] = mtg[i, j] - mtg[i, j-2]
	pgy_n[i, j] = mtg[i, j+2] - mtg[i, j]
	pgy_avg[i, j] = \
		(wgty_w[i, j] * pgy_w[i, j] + wgty_e[i, j] * pgy_e[i, j] +
		 wgty_s[i, j] * pgy_s[i, j] + wgty_n[i, j] * pgy_n[i, j]) / \
		(wgty_w[i, j] + wgty_e[i, j] + wgty_s[i, j] + wgty_n[i, j])

	u_tnd[i, j] = \
		((wgtx_c[i, j] - 1.0) * pgx_avg[i, j] - wgtx_c[i, j] * pgx_c[i, j]) / (2.0 * dx)
	v_tnd[i, j] = \
		((wgty_c[i, j] - 1.0) * pgy_avg[i, j] - wgty_c[i, j] * pgy_c[i, j]) / (2.0 * dy)


class IsentropicNonconservativePressureGradient(TendencyComponent):
	"""
	Calculate the anti-gradient of the Montgomery potential, which provides
	tendencies for the x- and y-velocity in the isentropic system.
	The class is always instantiated over the numerical grid of the
	underlying domain.
	"""
	def __init__(
		self, domain, scheme, backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		scheme : str
			The scheme to be used to discretize the gradient of the
			Montgomery potential. Available options are:

				* 'second_order', for a second-order centered formula;
				* 'fourth_order', for a fourth-order centered formula;
				* 'pressure_thickness_weighted', for a second-order \
					pressure-thickness-weighted formula.

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.

		References
		----------
		Bleck, R., and L. T. Smith. (1990). A wind-driven isopycnic coordinate \
			model of the north and equatorial Atlantic ocean. I: Model development \
			and supporting experiments. *J. Geophys. Res.*, 95:3273-3285.
		"""
		hb = domain.horizontal_boundary

		if scheme == 'second_order':
			nb = 1  # TODO: nb = 1 if hb.nb < 1 else hb.nb
			stencil_defs = self._stencil_second_order_defs
		elif scheme == 'fourth_order':
			nb = 2  # TODO: nb = 2 if hb.nb < 2 else hb.nb
			stencil_defs = self._stencil_fourth_order_defs
		elif scheme == 'pressure_thickness_weighted':
			nb = 2  # TODO: nb = 2 if hb.nb < 2 else hb.nb
			stencil_defs = self._stencil_weighted_defs
			self._eps = 1e-6
		else:
			expected_schemes = (
				'second_order', 'fourth_order', 'pressure_thickness_weighted'
			)
			raise RuntimeError(
				"The order should be either {}, but {} given.".format(
					', '.join(expected_schemes), scheme
				)
			)

		self._scheme = scheme

		super().__init__(domain, 'numerical', **kwargs)

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		assert nx > 2*nb, \
			"Number of grid points along the first horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, nx)
		assert ny > 2*nb, \
			"Number of grid points along the second horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, ny)

		inputs = {}
		self._in_mtg = np.zeros((nx, ny, nz), dtype=dtype)
		inputs['in_mtg'] = self._in_mtg
		if scheme == 'pressure_thickness_weighted':
			self._in_p = np.zeros((nx, ny, nz+1), dtype=dtype)
			inputs['in_p'] = self._in_p

		outputs = {}
		self._out_u_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		outputs['out_u_tnd'] = self._out_u_tnd
		self._out_v_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		outputs['out_v_tnd'] = self._out_v_tnd

		self._stencil = gt.NGStencil(
			definitions_func=stencil_defs,
			inputs=inputs,
			outputs=outputs,
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
		if self._scheme == 'pressure_thickness_weighted':
			dims_z = (dims[0], dims[1], self.grid.z_on_interface_levels.dims[0])
			return_dict['air_pressure_on_interface_levels'] = \
				{'dims': dims_z, 'units': 'Pa'}

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
		if self._scheme == 'pressure_thickness_weighted':
			self._in_p[...] = state['air_pressure_on_interface_levels']

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
		stage_second_order(i, j, dx, dy, in_mtg, out_u_tnd, out_v_tnd)

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
		stage_fourth_order(i, j, dx, dy, in_mtg, out_u_tnd, out_v_tnd)

		return out_u_tnd, out_v_tnd

	def _stencil_weighted_defs(self, in_mtg, in_p):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		eps = self._eps

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)
		k = gt.Index(axis=2)

		# instantiate the output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# define the computations
		stage_weighted(i, j, k, dx, dy, eps, in_mtg, in_p, out_u_tnd, out_v_tnd)

		return out_u_tnd, out_v_tnd


class IsentropicConservativePressureGradient(TendencyComponent):
	"""
	Calculate the anti-gradient of the Montgomery potential, multiplied by
	the air isentropic density. This quantity provides tendencies for the
	x- and y-momentum in the isentropic system. The class is always instantiated
	over the numerical grid of the underlying domain.
	"""
	def __init__(
		self, domain, scheme, backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		scheme : str
			The scheme to be used to discretize the gradient of the
			Montgomery potential. Available options are:

				* 'second_order', for a second-order centered formula;
				* 'fourth_order', for a fourth-order centered formula;
				* 'pressure_thickness_weighted', for a second-order \
					pressure-thickness-weighted formula.

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.

		References
		----------
		Bleck, R., and L. T. Smith. (1990). A wind-driven isopycnic coordinate \
			model of the north and equatorial Atlantic ocean. I: Model development \
			and supporting experiments. *J. Geophys. Res.*, 95:3273-3285.
		"""
		hb = domain.horizontal_boundary

		if scheme == 'second_order':
			nb = 1  # TODO: nb = 1 if hb.nb < 1 else hb.nb
			stencil_defs = self._stencil_second_order_defs
		elif scheme == 'fourth_order':
			nb = 2  # TODO: nb = 2 if hb.nb < 2 else hb.nb
			stencil_defs = self._stencil_fourth_order_defs
		elif scheme == 'pressure_thickness_weighted':
			nb = 2  # TODO: nb = 2 if hb.nb < 2 else hb.nb
			stencil_defs = self._stencil_weighted_defs
			self._eps = 10
		else:
			expected_schemes = (
				'second_order', 'fourth_order', 'pressure_thickness_weighted'
			)
			raise RuntimeError(
				"The order should be either {}, but {} given.".format(
					', '.join(expected_schemes), scheme
				)
			)

		self._scheme = scheme

		super().__init__(domain, 'numerical', **kwargs)

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		assert nx > 2*nb, \
			"Number of grid points along the first horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, nx)
		assert ny > 2*nb, \
			"Number of grid points along the second horizontal dimensions " \
			"should be larger than {}, but {} given.".format(2*nb, ny)

		inputs = {}
		self._in_s = np.zeros((nx, ny, nz), dtype=dtype)
		inputs['in_s'] = self._in_s
		self._in_mtg = np.zeros((nx, ny, nz), dtype=dtype)
		inputs['in_mtg'] = self._in_mtg
		if scheme == 'pressure_thickness_weighted':
			self._in_p = np.zeros((nx, ny, nz+1), dtype=dtype)
			inputs['in_p'] = self._in_p

		outputs = {}
		self._out_su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		outputs['out_su_tnd'] = self._out_su_tnd
		self._out_sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		outputs['out_sv_tnd'] = self._out_sv_tnd

		self._stencil = gt.NGStencil(
			definitions_func=stencil_defs,
			inputs=inputs,
			outputs=outputs,
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
		if self._scheme == 'pressure_thickness_weighted':
			dims_z = (dims[0], dims[1], self.grid.z_on_interface_levels.dims[0])
			return_dict['air_pressure_on_interface_levels'] = \
				{'dims': dims_z, 'units': 'Pa'}

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
		if self._scheme == 'pressure_thickness_weighted':
			self._in_p[...] = state['air_pressure_on_interface_levels']

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

		# instantiate the temporary and output fields
		tmp_u_tnd = gt.Equation()
		tmp_v_tnd = gt.Equation()
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# define the computations
		stage_second_order(i, j, dx, dy, in_mtg, tmp_u_tnd, tmp_v_tnd)
		out_su_tnd[i, j] = in_s[i, j] * tmp_u_tnd[i, j]
		out_sv_tnd[i, j] = in_s[i, j] * tmp_v_tnd[i, j]

		return out_su_tnd, out_sv_tnd

	def _stencil_fourth_order_defs(self, in_s, in_mtg):
		# shortcuts
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate the temporary and output fields
		tmp_u_tnd = gt.Equation()
		tmp_v_tnd = gt.Equation()
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# define the computations
		stage_fourth_order(i, j, dx, dy, in_mtg, tmp_u_tnd, tmp_v_tnd)
		out_su_tnd[i, j] = in_s[i, j] * tmp_u_tnd[i, j]
		out_sv_tnd[i, j] = in_s[i, j] * tmp_v_tnd[i, j]

		return out_su_tnd, out_sv_tnd

	def _stencil_weighted_defs(self, in_s, in_mtg, in_p):
		# shortcuts
		dx = self.grid.dx.to_units('m').values.item()
		dy = self.grid.dy.to_units('m').values.item()
		eps = self._eps

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)
		k = gt.Index(axis=2)

		# instantiate the temporary and output fields
		tmp_u_tnd = gt.Equation()
		tmp_v_tnd = gt.Equation()
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# define the computations
		stage_weighted(i, j, k, dx, dy, eps, in_mtg, in_p, tmp_u_tnd, tmp_v_tnd)
		out_su_tnd[i, j] = in_s[i, j] * tmp_u_tnd[i, j]
		out_sv_tnd[i, j] = in_s[i, j] * tmp_v_tnd[i, j]

		return out_su_tnd, out_sv_tnd

