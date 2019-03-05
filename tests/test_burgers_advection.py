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
from hypothesis import \
	given, HealthCheck, reproduce_failure, settings, strategies as hyp_st
import numpy as np
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

import gridtools as gt
from tasmania.python.burgers.dynamics.advection import \
	BurgersAdvection, _FirstOrder, _SecondOrder, _ThirdOrder, \
	_FourthOrder, _FifthOrder


class WrappingStencil:
	def __init__(self, advection, backend):
		self.call_func = advection.__call__
		self.nb = advection.extent
		self.backend = backend
		self.stencil = None

	def __call__(self, dx, dy, u, v):
		mi, mj, mk = u.shape

		self.dx = gt.Global()
		self.dy = gt.Global()
		self.adv_u_x = np.zeros_like(u, dtype=u.dtype)
		self.adv_u_y = np.zeros_like(u, dtype=u.dtype)
		self.adv_v_x = np.zeros_like(u, dtype=u.dtype)
		self.adv_v_y = np.zeros_like(u, dtype=u.dtype)

		self.stencil = gt.NGStencil(
			definitions_func=self.stencil_defs,
			inputs={'u': u, 'v': v},
			global_inputs={'dx': self.dx, 'dy': self.dy},
			outputs={
				'adv_u_x': self.adv_u_x, 'adv_u_y': self.adv_u_y,
				'adv_v_x': self.adv_v_x, 'adv_v_y': self.adv_v_y
			},
			domain=gt.domain.Rectangle(
				(self.nb, self.nb, 0), (mi-self.nb-1, mj-self.nb-1, mk-1)),
			mode=self.backend,
		)

		self.dx.value = dx
		self.dy.value = dy

		self.stencil.compute()

	def stencil_defs(self, dx, dy, u, v):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		return self.call_func(i, j, dx, dy, u, v)


def first_order_advection(dx, dy, u, v, phi):
	adv_x = np.zeros_like(phi, dtype=phi.dtype)
	adv_x[1:-1, :, :] = u[1:-1, :, :] / (2.0 * dx) * (phi[2:, :, :] - phi[:-2, :, :]) - \
		np.abs(u)[1:-1, :, :] / (2.0 * dx) * (
			phi[2:, :, :] - 2.0 * phi[1:-1, :, :] + phi[:-2, :, :]
		)
	adv_y = np.zeros_like(phi, dtype=phi.dtype)
	adv_y[:, 1:-1, :] = v[:, 1:-1, :] / (2.0 * dy) * (phi[:, 2:, :] - phi[:, :-2, :]) - \
		np.abs(v)[:, 1:-1, :] / (2.0 * dy) * (
			phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]
		)
	return adv_x, adv_y


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_first_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*_FirstOrder.extent+1, 40),
			yaxis_length=(2*_FirstOrder.extent+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
	state = data.draw(utils.st_burgers_state(grid), label='state')
	backend = data.draw(utils.st_one_of(conf.backend))

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	# ========================================
	# test interface
	# ========================================
	i = gt.Index()
	j = gt.Index()

	u_eq = gt.Equation()
	v_eq = gt.Equation()

	adv = BurgersAdvection.factory('first_order')

	assert isinstance(adv, BurgersAdvection)
	assert isinstance(adv, _FirstOrder)

	out = adv(i, j, dx, dy, u_eq, v_eq)

	assert len(out) == 4
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'adv_u_x'
	assert out[1].get_name() == 'adv_u_y'
	assert out[2].get_name() == 'adv_v_x'
	assert out[3].get_name() == 'adv_v_y'

	# ========================================
	# test numerics
	# ========================================
	ws = WrappingStencil(adv, backend)

	ws(dx, dy, u, v)

	adv_u_x, adv_u_y = first_order_advection(dx, dy, u, v, u)
	adv_v_x, adv_v_y = first_order_advection(dx, dy, u, v, v)

	assert np.allclose(adv_u_x[1:-1, 1:-1, :], ws.adv_u_x[1:-1, 1:-1, :])
	assert np.allclose(adv_u_y[1:-1, 1:-1, :], ws.adv_u_y[1:-1, 1:-1, :])
	assert np.allclose(adv_v_x[1:-1, 1:-1, :], ws.adv_v_x[1:-1, 1:-1, :])
	assert np.allclose(adv_v_y[1:-1, 1:-1, :], ws.adv_v_y[1:-1, 1:-1, :])


def second_order_advection(dx, dy, u, v, phi):
	adv_x = np.zeros_like(phi, dtype=phi.dtype)
	adv_x[1:-1, :, :] = u[1:-1, :, :] / (2.0 * dx) * (phi[2:, :, :] - phi[:-2, :, :])
	adv_y = np.zeros_like(phi, dtype=phi.dtype)
	adv_y[:, 1:-1, :] = v[:, 1:-1, :] / (2.0 * dy) * (phi[:, 2:, :] - phi[:, :-2, :])
	return adv_x, adv_y


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_second_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*_SecondOrder.extent+1, 40),
			yaxis_length=(2*_SecondOrder.extent+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
	state = data.draw(utils.st_burgers_state(grid), label='state')
	backend = data.draw(utils.st_one_of(conf.backend))

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	# ========================================
	# test interface
	# ========================================
	i = gt.Index()
	j = gt.Index()

	u_eq = gt.Equation()
	v_eq = gt.Equation()

	adv = BurgersAdvection.factory('second_order')

	assert isinstance(adv, BurgersAdvection)
	assert isinstance(adv, _SecondOrder)

	out = adv(i, j, dx, dy, u_eq, v_eq)

	assert len(out) == 4
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'adv_u_x'
	assert out[1].get_name() == 'adv_u_y'
	assert out[2].get_name() == 'adv_v_x'
	assert out[3].get_name() == 'adv_v_y'

	# ========================================
	# test numerics
	# ========================================
	ws = WrappingStencil(adv, backend)

	ws(dx, dy, u, v)

	adv_u_x, adv_u_y = second_order_advection(dx, dy, u, v, u)
	adv_v_x, adv_v_y = second_order_advection(dx, dy, u, v, v)

	assert np.allclose(adv_u_x[1:-1, 1:-1, :], ws.adv_u_x[1:-1, 1:-1, :])
	assert np.allclose(adv_u_y[1:-1, 1:-1, :], ws.adv_u_y[1:-1, 1:-1, :])
	assert np.allclose(adv_v_x[1:-1, 1:-1, :], ws.adv_v_x[1:-1, 1:-1, :])
	assert np.allclose(adv_v_y[1:-1, 1:-1, :], ws.adv_v_y[1:-1, 1:-1, :])


def third_order_advection(dx, dy, u, v, phi):
	adv_x = np.zeros_like(phi, dtype=phi.dtype)
	adv_x[2:-2, :, :] = u[2:-2, :, :] / (12.0 * dx) * (
			8.0 * (phi[3:-1, :, :] - phi[1:-3, :, :])
			- (phi[4:, :, :] - phi[:-4, :, :])
		) \
		+ np.abs(u)[2:-2, :, :] / (12.0 * dx) * (
			(phi[4:, :, :] + phi[:-4, :, :])
			- 4.0 * (phi[3:-1, :, :] + phi[1:-3, :, :])
			+ 6.0 * phi[2:-2, :, :]
		)
	adv_y = np.zeros_like(phi, dtype=phi.dtype)
	adv_y[:, 2:-2, :] = v[:, 2:-2, :] / (12.0 * dy) * (
			8.0 * (phi[:, 3:-1, :] - phi[:, 1:-3, :])
			- (phi[:, 4:, :] - phi[:, :-4, :])
		) \
		+ np.abs(v)[:, 2:-2, :] / (12.0 * dy) * (
			(phi[:, 4:, :] + phi[:, :-4, :])
			- 4.0 * (phi[:, 3:-1, :] + phi[:, 1:-3, :])
			+ 6.0 * phi[:, 2:-2, :]
		)
	return adv_x, adv_y


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_third_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*_ThirdOrder.extent+1, 40),
			yaxis_length=(2*_ThirdOrder.extent+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
	state = data.draw(utils.st_burgers_state(grid), label='state')
	backend = data.draw(utils.st_one_of(conf.backend))

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	# ========================================
	# test interface
	# ========================================
	i = gt.Index()
	j = gt.Index()

	u_eq = gt.Equation()
	v_eq = gt.Equation()

	adv = BurgersAdvection.factory('third_order')

	assert isinstance(adv, BurgersAdvection)
	assert isinstance(adv, _ThirdOrder)

	out = adv(i, j, dx, dy, u_eq, v_eq)

	assert len(out) == 4
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'adv_u_x'
	assert out[1].get_name() == 'adv_u_y'
	assert out[2].get_name() == 'adv_v_x'
	assert out[3].get_name() == 'adv_v_y'

	# ========================================
	# test numerics
	# ========================================
	ws = WrappingStencil(adv, backend)

	ws(dx, dy, u, v)

	adv_u_x, adv_u_y = third_order_advection(dx, dy, u, v, u)
	adv_v_x, adv_v_y = third_order_advection(dx, dy, u, v, v)

	assert np.allclose(adv_u_x[2:-2, 2:-2, :], ws.adv_u_x[2:-2, 2:-2, :])
	assert np.allclose(adv_u_y[2:-2, 2:-2, :], ws.adv_u_y[2:-2, 2:-2, :])
	assert np.allclose(adv_v_x[2:-2, 2:-2, :], ws.adv_v_x[2:-2, 2:-2, :])
	assert np.allclose(adv_v_y[2:-2, 2:-2, :], ws.adv_v_y[2:-2, 2:-2, :])


def fourth_order_advection(dx, dy, u, v, phi):
	adv_x = np.zeros_like(phi, dtype=phi.dtype)
	adv_x[2:-2, :, :] = u[2:-2, :, :] / (12.0 * dx) * (
			8.0 * (phi[3:-1, :, :] - phi[1:-3, :, :])
			- (phi[4:, :, :] - phi[:-4, :, :])
		)
	adv_y = np.zeros_like(phi, dtype=phi.dtype)
	adv_y[:, 2:-2, :] = v[:, 2:-2, :] / (12.0 * dy) * (
			8.0 * (phi[:, 3:-1, :] - phi[:, 1:-3, :])
			- (phi[:, 4:, :] - phi[:, :-4, :])
		)
	return adv_x, adv_y


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_fourth_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*_FourthOrder.extent+1, 40),
			yaxis_length=(2*_FourthOrder.extent+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
	state = data.draw(utils.st_burgers_state(grid), label='state')
	backend = data.draw(utils.st_one_of(conf.backend))

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	# ========================================
	# test interface
	# ========================================
	i = gt.Index()
	j = gt.Index()

	u_eq = gt.Equation()
	v_eq = gt.Equation()

	adv = BurgersAdvection.factory('fourth_order')

	assert isinstance(adv, BurgersAdvection)
	assert isinstance(adv, _FourthOrder)

	out = adv(i, j, dx, dy, u_eq, v_eq)

	assert len(out) == 4
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'adv_u_x'
	assert out[1].get_name() == 'adv_u_y'
	assert out[2].get_name() == 'adv_v_x'
	assert out[3].get_name() == 'adv_v_y'

	# ========================================
	# test numerics
	# ========================================
	ws = WrappingStencil(adv, backend)

	ws(dx, dy, u, v)

	adv_u_x, adv_u_y = fourth_order_advection(dx, dy, u, v, u)
	adv_v_x, adv_v_y = fourth_order_advection(dx, dy, u, v, v)

	assert np.allclose(adv_u_x[2:-2, 2:-2, :], ws.adv_u_x[2:-2, 2:-2, :])
	assert np.allclose(adv_u_y[2:-2, 2:-2, :], ws.adv_u_y[2:-2, 2:-2, :])
	assert np.allclose(adv_v_x[2:-2, 2:-2, :], ws.adv_v_x[2:-2, 2:-2, :])
	assert np.allclose(adv_v_y[2:-2, 2:-2, :], ws.adv_v_y[2:-2, 2:-2, :])


def fifth_order_advection(dx, dy, u, v, phi):
	adv_x = np.zeros_like(phi, dtype=phi.dtype)
	adv_x[3:-3, :, :] = u[3:-3, :, :] / (60.0 * dx) * (
			45.0 * (phi[4:-2, :, :] - phi[2:-4, :, :])
			- 9.0 * (phi[5:-1, :, :] - phi[1:-5, :, :])
			+ (phi[6:, :, :] - phi[:-6, :, :])
		) \
		- np.abs(u)[3:-3, :, :] / (60.0 * dx) * (
			(phi[6:, :, :] + phi[:-6, :, :])
			- 6.0 * (phi[5:-1, :, :] + phi[1:-5, :, :])
			+ 15.0 * (phi[4:-2, :, :] + phi[2:-4, :, :])
			- 20.0 * phi[3:-3, :, :]
		)
	adv_y = np.zeros_like(phi, dtype=phi.dtype)
	adv_y[:, 3:-3, :] = v[:, 3:-3, :] / (60.0 * dy) * (
			45.0 * (phi[:, 4:-2, :] - phi[:, 2:-4, :])
			- 9.0 * (phi[:, 5:-1, :] - phi[:, 1:-5, :])
			+ (phi[:, 6:, :] - phi[:, :-6, :])
		) \
		- np.abs(v)[:, 3:-3, :] / (60.0 * dy) * (
			(phi[:, 6:, :] + phi[:, :-6, :])
			- 6.0 * (phi[:, 5:-1, :] + phi[:, 1:-5, :])
			+ 15.0 * (phi[:, 4:-2, :] + phi[:, 2:-4, :])
			- 20.0 * phi[:, 3:-3, :]
		)
	return adv_x, adv_y


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_fifth_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*_FifthOrder.extent+1, 40),
			yaxis_length=(2*_FifthOrder.extent+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
	state = data.draw(utils.st_burgers_state(grid), label='state')
	backend = data.draw(utils.st_one_of(conf.backend))

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	# ========================================
	# test interface
	# ========================================
	i = gt.Index()
	j = gt.Index()

	u_eq = gt.Equation()
	v_eq = gt.Equation()

	adv = BurgersAdvection.factory('fifth_order')

	assert isinstance(adv, BurgersAdvection)
	assert isinstance(adv, _FifthOrder)

	out = adv(i, j, dx, dy, u_eq, v_eq)

	assert len(out) == 4
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'adv_u_x'
	assert out[1].get_name() == 'adv_u_y'
	assert out[2].get_name() == 'adv_v_x'
	assert out[3].get_name() == 'adv_v_y'

	# ========================================
	# test numerics
	# ========================================
	ws = WrappingStencil(adv, backend)

	ws(dx, dy, u, v)

	adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u, v, u)
	adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u, v, v)

	assert np.allclose(adv_u_x[3:-3, 3:-3, :], ws.adv_u_x[3:-3, 3:-3, :])
	assert np.allclose(adv_u_y[3:-3, 3:-3, :], ws.adv_u_y[3:-3, 3:-3, :])
	assert np.allclose(adv_v_x[3:-3, 3:-3, :], ws.adv_v_x[3:-3, 3:-3, :])
	assert np.allclose(adv_v_y[3:-3, 3:-3, :], ws.adv_v_y[3:-3, 3:-3, :])


if __name__ == '__main__':
	pytest.main([__file__])
