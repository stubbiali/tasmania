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
from datetime import timedelta
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.dynamics.diagnostics import HorizontalVelocity
from tasmania.python.dwarfs.horizontal_boundary import HorizontalBoundary
from tasmania.python.isentropic.dynamics.homogeneous_prognostic \
	import HomogeneousIsentropicPrognostic as IsentropicPrognostic
from tasmania.python.isentropic.dynamics._homogeneous_prognostic \
	import Centered, ForwardEuler, RK2, RK3COSMO, RK3
from tasmania.python.utils.data_utils import make_raw_state


mfwv  = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw  = 'mass_fraction_of_precipitation_water_in_air'


def test_factory(grid):
	mode = 'xy'
	backend = gt.mode.NUMPY
	hb = HorizontalBoundary.factory('relaxed', grid, 1)

	ip_centered = IsentropicPrognostic.factory(
		'centered', mode, grid, True, hb,
		horizontal_flux_scheme='centered', backend=backend
	)
	ip_euler = IsentropicPrognostic.factory(
		'forward_euler', mode, grid, True, hb,
		horizontal_flux_scheme='upwind', backend=backend
	)
	ip_rk2 = IsentropicPrognostic.factory(
		'rk2', mode, grid, True, hb,
		horizontal_flux_scheme='third_order_upwind', backend=backend
	)
	ip_rk3cosmo = IsentropicPrognostic.factory(
		'rk3cosmo', mode, grid, True, hb,
		horizontal_flux_scheme='fifth_order_upwind', backend=backend
	)
	ip_rk3 = IsentropicPrognostic.factory(
		'rk3', mode, grid, True, hb,
		horizontal_flux_scheme='fifth_order_upwind', backend=backend
	)

	assert isinstance(ip_centered, Centered)
	assert isinstance(ip_euler, ForwardEuler)
	assert isinstance(ip_rk2, RK2)
	assert isinstance(ip_rk3cosmo, RK3COSMO)
	assert isinstance(ip_rk3, RK3)


def test_leapfrog(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	mode = 'xy'
	backend = gt.mode.NUMPY
	dtype = state['air_isentropic_density'].values.dtype

	hb = HorizontalBoundary.factory('periodic', grid, 1)

	ip_centered = IsentropicPrognostic.factory(
		'centered', mode, grid, True, hb,
		horizontal_flux_scheme='centered',
		backend=backend, dtype=dtype
	)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mfwv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfcw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = {}

	raw_state_new = ip_centered.stage_call(0, dt, raw_state, raw_tendencies)

	assert 'time' in raw_state_new.keys()
	assert raw_state_new['time'] == raw_state['time'] + dt

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']

	s_new = np.zeros_like(s, dtype=dtype)
	s_new[1:-1, :, :] = s[1:-1, :, :] - \
						dt.seconds * (su[2:, :, :] - su[:-2, :, :]) / dx
	assert 'air_isentropic_density' in raw_state_new.keys()
	assert np.allclose(s_new[1:-1, :, :], raw_state_new['air_isentropic_density'][1:-1, :, :])

	su_new = np.zeros_like(su, dtype=dtype)
	su_new[1:-1, :, :] = su[1:-1, :, :] - \
						 dt.seconds / dx * (u[2:-1, :, :] * (su[2:, :, :] + su[1:-1, :, :]) -
											u[1:-2, :, :] * (su[1:-1, :, :] + su[:-2, :, :]))
	assert 'x_momentum_isentropic' in raw_state_new.keys()
	assert np.allclose(su_new[1:-1, :, :], raw_state_new['x_momentum_isentropic'][1:-1, :, :])

	assert 'y_momentum_isentropic' in raw_state_new.keys()
	assert np.allclose(sv, raw_state_new['y_momentum_isentropic'])

	sqv_new = np.zeros_like(sqv, dtype=dtype)
	sqv_new[1:-1, :, :] = sqv[1:-1, :, :] - \
						  dt.seconds / dx * (u[2:-1, :, :] * (sqv[2:, :, :] + sqv[1:-1, :, :]) -
											 u[1:-2, :, :] * (sqv[1:-1, :, :] + sqv[:-2, :, :]))
	assert 'isentropic_density_of_water_vapor' in raw_state_new.keys()
	assert np.allclose(
		sqv_new[1:-1, :, :], raw_state_new['isentropic_density_of_water_vapor'][1:-1, :, :]
	)

	sqc_new = np.zeros_like(sqc, dtype=dtype)
	sqc_new[1:-1, :, :] = sqc[1:-1, :, :] - \
						  dt.seconds / dx * (u[2:-1, :, :] * (sqc[2:, :, :] + sqc[1:-1, :, :]) -
											 u[1:-2, :, :] * (sqc[1:-1, :, :] + sqc[:-2, :, :]))
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_new.keys()
	assert np.allclose(
		sqc_new[1:-1, :, :], raw_state_new['isentropic_density_of_cloud_liquid_water'][1:-1, :, :]
	)

	sqr_new = np.zeros_like(sqr, dtype=dtype)
	sqr_new[1:-1, :, :] = sqr[1:-1, :, :] - \
						  dt.seconds / dx * (u[2:-1, :, :] * (sqr[2:, :, :] + sqr[1:-1, :, :]) -
											 u[1:-2, :, :] * (sqr[1:-1, :, :] + sqr[:-2, :, :]))
	assert 'isentropic_density_of_precipitation_water' in raw_state_new.keys()
	assert np.allclose(
		sqr_new[1:-1, :, :], raw_state_new['isentropic_density_of_precipitation_water'][1:-1, :, :]
	)


def test_upwind(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	mode = 'xy'
	backend = gt.mode.NUMPY
	dtype = state['air_isentropic_density'].values.dtype

	hb = HorizontalBoundary.factory('periodic', grid, 1)

	ip_euler = IsentropicPrognostic.factory(
		'forward_euler', mode, grid, True, hb,
		horizontal_flux_scheme='upwind',
		backend=backend, dtype=dtype
	)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mfwv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfcw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = {}

	raw_state_new = ip_euler.stage_call(0, dt, raw_state, raw_tendencies)

	assert 'time' in raw_state_new.keys()
	assert raw_state_new['time'] == raw_state['time'] + dt

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * s[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * s[1:, :, :]
	flux_s_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	s_new = s - dt.seconds / dx * (flux_s_x[1:, :, :] - flux_s_x[:-1, :, :])
	assert 'air_isentropic_density' in raw_state_new.keys()
	assert np.allclose(s_new, raw_state_new['air_isentropic_density'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * su[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * su[1:, :, :]
	flux_su_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	su_new = su - dt.seconds / dx * (flux_su_x[1:, :, :] - flux_su_x[:-1, :, :])
	assert 'x_momentum_isentropic' in raw_state_new.keys()
	assert np.allclose(su_new, raw_state_new['x_momentum_isentropic'])

	assert np.allclose(np.zeros_like(sv, dtype=dtype),
					   raw_state_new['y_momentum_isentropic'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * sqv[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * sqv[1:, :, :]
	flux_sqv_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	sqv_new = sqv - dt.seconds / dx * (flux_sqv_x[1:, :, :] - flux_sqv_x[:-1, :, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_new.keys()
	assert np.allclose(sqv_new, raw_state_new['isentropic_density_of_water_vapor'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * sqc[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * sqc[1:, :, :]
	flux_sqc_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	sqc_new = sqc - dt.seconds / dx * (flux_sqc_x[1:, :, :] - flux_sqc_x[:-1, :, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_new.keys()
	assert np.allclose(sqc_new, raw_state_new['isentropic_density_of_cloud_liquid_water'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * sqr[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * sqr[1:, :, :]
	flux_sqr_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	sqr_new = sqr - dt.seconds / dx * (flux_sqr_x[1:, :, :] - flux_sqr_x[:-1, :, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_new.keys()
	assert np.allclose(sqr_new, raw_state_new['isentropic_density_of_precipitation_water'])


def test_rk2(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	mode = 'xy'
	backend = gt.mode.NUMPY
	dtype = state['air_isentropic_density'].values.dtype

	hb = HorizontalBoundary.factory('periodic', grid)

	hv = HorizontalVelocity(grid, dtype=dtype)

	ip_rk2 = IsentropicPrognostic.factory(
		'rk2', mode, grid, True, hb,
		horizontal_flux_scheme='third_order_upwind',
		backend=backend, dtype=dtype
	)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mfwv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfcw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = {}

	raw_state_1 = ip_rk2.stage_call(0, dt, raw_state, raw_tendencies)

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']

	s_   = hb.from_physical_to_computational_domain(s)
	u_   = hb.from_physical_to_computational_domain(u)
	su_  = hb.from_physical_to_computational_domain(su)
	sqv_ = hb.from_physical_to_computational_domain(sqv)
	sqc_ = hb.from_physical_to_computational_domain(sqc)
	sqr_ = hb.from_physical_to_computational_domain(sqr)

	flux4 = u_[2:-2, :, :] / 12. * (7. * (s_[2:-1, :, :] + s_[1:-2, :, :]) -
									1. * (s_[3:, :, :] + s_[:-3, :, :]))
	flux = flux4 - np.abs(u_[2:-2, :, :]) / 12. * (3. * (s_[2:-1, :, :] - s_[1:-2, :, :]) -
												   1. * (s_[3:, :, :] - s_[:-3, :, :]))
	s1 = s - 0.5 * dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'air_isentropic_density' in raw_state_1.keys()
	assert np.allclose(s1, raw_state_1['air_isentropic_density'])

	flux4 = u_[2:-2, :, :] / 12. * (7. * (su_[2:-1, :, :] + su_[1:-2, :, :]) -
									1. * (su_[3:, :, :] + su_[:-3, :, :]))
	flux = flux4 - np.abs(u_[2:-2, :, :]) / 12. * (3. * (su_[2:-1, :, :] - su_[1:-2, :, :]) -
												   1. * (su_[3:, :, :] - su_[:-3, :, :]))
	su1 = su - 1./2. * dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'x_momentum_isentropic' in raw_state_1.keys()
	assert np.allclose(su1, raw_state_1['x_momentum_isentropic'])

	flux4 = u_[2:-2, :, :] / 12. * (7. * (sqv_[2:-1, :, :] + sqv_[1:-2, :, :]) -
									1. * (sqv_[3:, :, :] + sqv_[:-3, :, :]))
	flux = flux4 - np.abs(u_[2:-2, :, :]) / 12. * (3. * (sqv_[2:-1, :, :] - sqv_[1:-2, :, :]) -
												   1. * (sqv_[3:, :, :] - sqv_[:-3, :, :]))
	sqv1 = sqv - 1./2. * dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_1.keys()
	assert np.allclose(sqv1, raw_state_1['isentropic_density_of_water_vapor'])

	flux4 = u_[2:-2, :, :] / 12. * (7. * (sqc_[2:-1, :, :] + sqc_[1:-2, :, :]) -
									1. * (sqc_[3:, :, :] + sqc_[:-3, :, :]))
	flux = flux4 - np.abs(u_[2:-2, :, :]) / 12. * (3. * (sqc_[2:-1, :, :] - sqc_[1:-2, :, :]) -
												   1. * (sqc_[3:, :, :] - sqc_[:-3, :, :]))
	sqc1 = sqc - 1./2. * dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_1.keys()
	assert np.allclose(sqc1, raw_state_1['isentropic_density_of_cloud_liquid_water'])

	flux4 = u_[2:-2, :, :] / 12. * (7. * (sqr_[2:-1, :, :] + sqr_[1:-2, :, :]) -
									1. * (sqr_[3:, :, :] + sqr_[:-3, :, :]))
	flux = flux4 - np.abs(u_[2:-2, :, :]) / 12. * (3. * (sqr_[2:-1, :, :] - sqr_[1:-2, :, :]) -
												   5. * (sqr_[3:, :, :] - sqr_[:-3, :, :]))
	sqr1 = sqr - 1./2. * dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_1.keys()
	assert np.allclose(sqr1, raw_state_1['isentropic_density_of_precipitation_water'])

	u1, v1 = hv.get_velocity_components(s1, su1, sv)
	raw_state_1['x_velocity_at_u_locations'] = u1
	raw_state_1['y_velocity_at_v_locations'] = v1

	raw_state_2 = ip_rk2.stage_call(1, dt, raw_state_1, raw_tendencies)

	s1_   = hb.from_physical_to_computational_domain(s1)
	u1_   = hb.from_physical_to_computational_domain(u1)
	su1_  = hb.from_physical_to_computational_domain(su1)
	sqv1_ = hb.from_physical_to_computational_domain(sqv1)
	sqc1_ = hb.from_physical_to_computational_domain(sqc1)
	sqr1_ = hb.from_physical_to_computational_domain(sqr1)

	flux4 = u1_[2:-2, :, :] / 12. * (7. * (s1_[2:-1, :, :] + s1_[1:-2, :, :]) -
									 1. * (s1_[3:, :, :] + s1_[:-3, :, :]))
	flux = flux4 - np.abs(u1_[2:-2, :, :]) / 12. * (3. * (s1_[2:-1, :, :] - s1_[1:-2, :, :]) -
													1. * (s1_[3:, :, :] - s1_[:-3, :, :]))
	s2 = s - dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'air_isentropic_density' in raw_state_2.keys()
	assert np.allclose(s2, raw_state_2['air_isentropic_density'])

	flux4 = u1_[2:-2, :, :] / 12. * (7. * (su1_[2:-1, :, :] + su1_[1:-2, :, :]) -
									 1. * (su1_[3:, :, :] + su1_[:-3, :, :]))
	flux = flux4 - np.abs(u1_[2:-2, :, :]) / 12. * (3. * (su1_[2:-1, :, :] - su1_[1:-2, :, :]) -
													1. * (su1_[3:, :, :] - su1_[:-3, :, :]))
	su2 = su - dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'x_momentum_isentropic' in raw_state_2.keys()
	assert np.allclose(su2, raw_state_2['x_momentum_isentropic'])

	flux4 = u1_[2:-2, :, :] / 12. * (7. * (sqv1_[2:-1, :, :] + sqv1_[1:-2, :, :]) -
									 1. * (sqv1_[3:, :, :] + sqv1_[:-3, :, :]))
	flux = flux4 - np.abs(u1_[2:-2, :, :]) / 12. * (3. * (sqv1_[2:-1, :, :] - sqv1_[1:-2, :, :]) -
													1. * (sqv1_[3:, :, :] - sqv1_[:-3, :, :]))
	sqv2 = sqv - dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_2.keys()
	assert np.allclose(sqv2, raw_state_2['isentropic_density_of_water_vapor'])

	flux4 = u1_[2:-2, :, :] / 12. * (7. * (sqc1_[2:-1, :, :] + sqc1_[1:-2, :, :]) -
									 1. * (sqc1_[3:, :, :] + sqc1_[:-3, :, :]))
	flux = flux4 - np.abs(u1_[2:-2, :, :]) / 12. * (3. * (sqc1_[2:-1, :, :] - sqc1_[1:-2, :, :]) -
													1. * (sqc1_[3:, :, :] - sqc1_[:-3, :, :]))
	sqc2 = sqc - dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_2.keys()
	assert np.allclose(sqc2, raw_state_2['isentropic_density_of_cloud_liquid_water'])

	flux4 = u1_[2:-2, :, :] / 12. * (7. * (sqr1_[2:-1, :, :] + sqr1_[1:-2, :, :]) -
									 1. * (sqr1_[3:, :, :] + sqr1_[:-3, :, :]))
	flux = flux4 - np.abs(u1_[2:-2, :, :]) / 12. * (3. * (sqr1_[2:-1, :, :] - sqr1_[1:-2, :, :]) -
													1. * (sqr1_[3:, :, :] - sqr1_[:-3, :, :]))
	sqr2 = sqr - dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_2.keys()
	assert np.allclose(sqr2, raw_state_2['isentropic_density_of_precipitation_water'])


def test_rk3cosmo(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	mode = 'xy'
	backend = gt.mode.NUMPY
	dtype = state['air_isentropic_density'].values.dtype

	hb = HorizontalBoundary.factory('periodic', grid)

	hv = HorizontalVelocity(grid, dtype=dtype)

	ip_rk3 = IsentropicPrognostic.factory(
		'rk3cosmo', mode, grid, True, hb,
		horizontal_flux_scheme='fifth_order_upwind',
		backend=backend, dtype=dtype
	)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mfwv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfcw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = {}

	raw_state_1 = ip_rk3.stage_call(0, dt, raw_state, raw_tendencies)

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']

	s_   = hb.from_physical_to_computational_domain(s)
	u_   = hb.from_physical_to_computational_domain(u)
	su_  = hb.from_physical_to_computational_domain(su)
	sqv_ = hb.from_physical_to_computational_domain(sqv)
	sqc_ = hb.from_physical_to_computational_domain(sqc)
	sqr_ = hb.from_physical_to_computational_domain(sqr)

	flux6 = u_[3:-3, :, :] / 60. * (37. * (s_[3:-2, :, :] + s_[2:-3, :, :]) -
								     8. * (s_[4:-1, :, :] + s_[1:-4, :, :]) +
								     1. * (s_[  5:, :, :] + s_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (s_[3:-2, :, :] - s_[2:-3, :, :]) -
											  		5. * (s_[4:-1, :, :] - s_[1:-4, :, :]) +
											  		1. * (s_[  5:, :, :] - s_[ :-5, :, :]))
	s1 = s - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'air_isentropic_density' in raw_state_1.keys()
	assert np.allclose(s1, raw_state_1['air_isentropic_density'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (su_[3:-2, :, :] + su_[2:-3, :, :]) -
									 8. * (su_[4:-1, :, :] + su_[1:-4, :, :]) +
									 1. * (su_[  5:, :, :] + su_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (su_[3:-2, :, :] - su_[2:-3, :, :]) -
											  		5. * (su_[4:-1, :, :] - su_[1:-4, :, :]) +
											  		1. * (su_[  5:, :, :] - su_[ :-5, :, :]))
	su1 = su - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'x_momentum_isentropic' in raw_state_1.keys()
	assert np.allclose(su1, raw_state_1['x_momentum_isentropic'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqv_[3:-2, :, :] + sqv_[2:-3, :, :]) -
									 8. * (sqv_[4:-1, :, :] + sqv_[1:-4, :, :]) +
									 1. * (sqv_[  5:, :, :] + sqv_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqv_[3:-2, :, :] - sqv_[2:-3, :, :]) -
											  		5. * (sqv_[4:-1, :, :] - sqv_[1:-4, :, :]) +
											  		1. * (sqv_[  5:, :, :] - sqv_[ :-5, :, :]))
	sqv1 = sqv - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_1.keys()
	assert np.allclose(sqv1, raw_state_1['isentropic_density_of_water_vapor'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqc_[3:-2, :, :] + sqc_[2:-3, :, :]) -
									 8. * (sqc_[4:-1, :, :] + sqc_[1:-4, :, :]) +
									 1. * (sqc_[  5:, :, :] + sqc_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqc_[3:-2, :, :] - sqc_[2:-3, :, :]) -
											  		5. * (sqc_[4:-1, :, :] - sqc_[1:-4, :, :]) +
											  		1. * (sqc_[  5:, :, :] - sqc_[ :-5, :, :]))
	sqc1 = sqc - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_1.keys()
	assert np.allclose(sqc1, raw_state_1['isentropic_density_of_cloud_liquid_water'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqr_[3:-2, :, :] + sqr_[2:-3, :, :]) -
									 8. * (sqr_[4:-1, :, :] + sqr_[1:-4, :, :]) +
									 1. * (sqr_[  5:, :, :] + sqr_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqr_[3:-2, :, :] - sqr_[2:-3, :, :]) -
											  		5. * (sqr_[4:-1, :, :] - sqr_[1:-4, :, :]) +
											  		1. * (sqr_[  5:, :, :] - sqr_[ :-5, :, :]))
	sqr1 = sqr - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_1.keys()
	assert np.allclose(sqr1, raw_state_1['isentropic_density_of_precipitation_water'])

	u1, v1 = hv.get_velocity_components(s1, su1, sv)
	raw_state_1['x_velocity_at_u_locations'] = u1
	raw_state_1['y_velocity_at_v_locations'] = v1

	raw_state_2 = ip_rk3.stage_call(1, dt, raw_state_1, raw_tendencies)

	s1_   = hb.from_physical_to_computational_domain(s1)
	u1_   = hb.from_physical_to_computational_domain(u1)
	su1_  = hb.from_physical_to_computational_domain(su1)
	sqv1_ = hb.from_physical_to_computational_domain(sqv1)
	sqc1_ = hb.from_physical_to_computational_domain(sqc1)
	sqr1_ = hb.from_physical_to_computational_domain(sqr1)

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (s1_[3:-2, :, :] + s1_[2:-3, :, :]) -
									  8. * (s1_[4:-1, :, :] + s1_[1:-4, :, :]) +
									  1. * (s1_[  5:, :, :] + s1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (s1_[3:-2, :, :] - s1_[2:-3, :, :]) -
											   		 5. * (s1_[4:-1, :, :] - s1_[1:-4, :, :]) +
											   		 1. * (s1_[  5:, :, :] - s1_[ :-5, :, :]))
	s2 = s - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'air_isentropic_density' in raw_state_2.keys()
	assert np.allclose(s2, raw_state_2['air_isentropic_density'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (su1_[3:-2, :, :] + su1_[2:-3, :, :]) -
									  8. * (su1_[4:-1, :, :] + su1_[1:-4, :, :]) +
									  1. * (su1_[  5:, :, :] + su1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (su1_[3:-2, :, :] - su1_[2:-3, :, :]) -
											   		 5. * (su1_[4:-1, :, :] - su1_[1:-4, :, :]) +
											   		 1. * (su1_[  5:, :, :] - su1_[ :-5, :, :]))
	su2 = su - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'x_momentum_isentropic' in raw_state_2.keys()
	assert np.allclose(su2, raw_state_2['x_momentum_isentropic'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqv1_[3:-2, :, :] + sqv1_[2:-3, :, :]) -
									  8. * (sqv1_[4:-1, :, :] + sqv1_[1:-4, :, :]) +
									  1. * (sqv1_[  5:, :, :] + sqv1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqv1_[3:-2, :, :] - sqv1_[2:-3, :, :]) -
											   		 5. * (sqv1_[4:-1, :, :] - sqv1_[1:-4, :, :]) +
											   		 1. * (sqv1_[  5:, :, :] - sqv1_[ :-5, :, :]))
	sqv2 = sqv - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_2.keys()
	assert np.allclose(sqv2, raw_state_2['isentropic_density_of_water_vapor'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqc1_[3:-2, :, :] + sqc1_[2:-3, :, :]) -
									  8. * (sqc1_[4:-1, :, :] + sqc1_[1:-4, :, :]) +
									  1. * (sqc1_[  5:, :, :] + sqc1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqc1_[3:-2, :, :] - sqc1_[2:-3, :, :]) -
											   		 5. * (sqc1_[4:-1, :, :] - sqc1_[1:-4, :, :]) +
											   		 1. * (sqc1_[  5:, :, :] - sqc1_[ :-5, :, :]))
	sqc2 = sqc - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_2.keys()
	assert np.allclose(sqc2, raw_state_2['isentropic_density_of_cloud_liquid_water'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqr1_[3:-2, :, :] + sqr1_[2:-3, :, :]) -
									  8. * (sqr1_[4:-1, :, :] + sqr1_[1:-4, :, :]) +
									  1. * (sqr1_[  5:, :, :] + sqr1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqr1_[3:-2, :, :] - sqr1_[2:-3, :, :]) -
											   		 5. * (sqr1_[4:-1, :, :] - sqr1_[1:-4, :, :]) +
											   		 1. * (sqr1_[  5:, :, :] - sqr1_[ :-5, :, :]))
	sqr2 = sqr - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_2.keys()
	assert np.allclose(sqr2, raw_state_2['isentropic_density_of_precipitation_water'])

	u2, v2 = hv.get_velocity_components(s2, su2, sv)
	raw_state_2['x_velocity_at_u_locations'] = u2
	raw_state_2['y_velocity_at_v_locations'] = v2

	raw_state_3 = ip_rk3.stage_call(2, dt, raw_state_2, raw_tendencies)

	s2_   = hb.from_physical_to_computational_domain(s2)
	u2_   = hb.from_physical_to_computational_domain(u2)
	su2_  = hb.from_physical_to_computational_domain(su2)
	sqv2_ = hb.from_physical_to_computational_domain(sqv2)
	sqc2_ = hb.from_physical_to_computational_domain(sqc2)
	sqr2_ = hb.from_physical_to_computational_domain(sqr2)

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (s2_[3:-2, :, :] + s2_[2:-3, :, :]) -
									  8. * (s2_[4:-1, :, :] + s2_[1:-4, :, :]) +
									  1. * (s2_[  5:, :, :] + s2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (s2_[3:-2, :, :] - s2_[2:-3, :, :]) -
													 5. * (s2_[4:-1, :, :] - s2_[1:-4, :, :]) +
													 1. * (s2_[  5:, :, :] - s2_[ :-5, :, :]))
	s3 = s - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'air_isentropic_density' in raw_state_3.keys()
	assert np.allclose(s3, raw_state_3['air_isentropic_density'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (su2_[3:-2, :, :] + su2_[2:-3, :, :]) -
									  8. * (su2_[4:-1, :, :] + su2_[1:-4, :, :]) +
									  1. * (su2_[  5:, :, :] + su2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (su2_[3:-2, :, :] - su2_[2:-3, :, :]) -
													 5. * (su2_[4:-1, :, :] - su2_[1:-4, :, :]) +
													 1. * (su2_[  5:, :, :] - su2_[ :-5, :, :]))
	su3 = su - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'x_momentum_isentropic' in raw_state_3.keys()
	assert np.allclose(su3[:, 3:4, :], raw_state_3['x_momentum_isentropic'][:, 3:4, :])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqv2_[3:-2, :, :] + sqv2_[2:-3, :, :]) -
									  8. * (sqv2_[4:-1, :, :] + sqv2_[1:-4, :, :]) +
									  1. * (sqv2_[  5:, :, :] + sqv2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqv2_[3:-2, :, :] - sqv2_[2:-3, :, :]) -
											 		 5. * (sqv2_[4:-1, :, :] - sqv2_[1:-4, :, :]) +
											 		 1. * (sqv2_[  5:, :, :] - sqv2_[ :-5, :, :]))
	sqv3 = sqv - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_3.keys()
	assert np.allclose(sqv3, raw_state_3['isentropic_density_of_water_vapor'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqc2_[3:-2, :, :] + sqc2_[2:-3, :, :]) -
									  8. * (sqc2_[4:-1, :, :] + sqc2_[1:-4, :, :]) +
									  1. * (sqc2_[  5:, :, :] + sqc2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqc2_[3:-2, :, :] - sqc2_[2:-3, :, :]) -
													 5. * (sqc2_[4:-1, :, :] - sqc2_[1:-4, :, :]) +
													 1. * (sqc2_[  5:, :, :] - sqc2_[ :-5, :, :]))
	sqc3 = sqc - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_3.keys()
	assert np.allclose(sqc3, raw_state_3['isentropic_density_of_cloud_liquid_water'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqr2_[3:-2, :, :] + sqr2_[2:-3, :, :]) -
									  8. * (sqr2_[4:-1, :, :] + sqr2_[1:-4, :, :]) +
									  1. * (sqr2_[  5:, :, :] + sqr2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqr2_[3:-2, :, :] - sqr2_[2:-3, :, :]) -
													 5. * (sqr2_[4:-1, :, :] - sqr2_[1:-4, :, :]) +
													 1. * (sqr2_[  5:, :, :] - sqr2_[ :-5, :, :]))
	sqr3 = sqr - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_3.keys()
	assert np.allclose(sqr3, raw_state_3['isentropic_density_of_precipitation_water'])


def test_rk3(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	mode = 'xy'
	backend = gt.mode.NUMPY
	dtype = state['air_isentropic_density'].values.dtype

	hb = HorizontalBoundary.factory('periodic', grid)
	hv = HorizontalVelocity(grid, dtype=dtype)

	ip_rk3 = IsentropicPrognostic.factory(
		'rk3', mode, grid, True, hb,
		horizontal_flux_scheme='fifth_order_upwind',
		backend=backend, dtype=dtype
	)

	a1, a2 = ip_rk3._alpha1, ip_rk3._alpha2
	b21 = ip_rk3._beta21
	g0, g1, g2 = ip_rk3._gamma0, ip_rk3._gamma1, ip_rk3._gamma2

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mfwv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfcw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = {}

	raw_state_1 = ip_rk3.stage_call(0, dt, raw_state, raw_tendencies)

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']

	s_   = hb.from_physical_to_computational_domain(s)
	u_   = hb.from_physical_to_computational_domain(u)
	su_  = hb.from_physical_to_computational_domain(su)
	sqv_ = hb.from_physical_to_computational_domain(sqv)
	sqc_ = hb.from_physical_to_computational_domain(sqc)
	sqr_ = hb.from_physical_to_computational_domain(sqr)

	flux6 = u_[3:-3, :, :] / 60. * (37. * (s_[3:-2, :, :] + s_[2:-3, :, :]) -
									8. * (s_[4:-1, :, :] + s_[1:-4, :, :]) +
									1. * (s_[  5:, :, :] + s_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (s_[3:-2, :, :] - s_[2:-3, :, :]) -
												   5. * (s_[4:-1, :, :] - s_[1:-4, :, :]) +
												   1. * (s_[  5:, :, :] - s_[ :-5, :, :]))
	k0_s = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	s1 = s + a1 * k0_s
	assert 'air_isentropic_density' in raw_state_1.keys()
	assert np.allclose(s1, raw_state_1['air_isentropic_density'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (su_[3:-2, :, :] + su_[2:-3, :, :]) -
									8. * (su_[4:-1, :, :] + su_[1:-4, :, :]) +
									1. * (su_[  5:, :, :] + su_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (su_[3:-2, :, :] - su_[2:-3, :, :]) -
												   5. * (su_[4:-1, :, :] - su_[1:-4, :, :]) +
												   1. * (su_[  5:, :, :] - su_[ :-5, :, :]))
	k0_su = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	su1 = su + a1 * k0_su
	assert 'x_momentum_isentropic' in raw_state_1.keys()
	assert np.allclose(su1, raw_state_1['x_momentum_isentropic'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqv_[3:-2, :, :] + sqv_[2:-3, :, :]) -
									8. * (sqv_[4:-1, :, :] + sqv_[1:-4, :, :]) +
									1. * (sqv_[  5:, :, :] + sqv_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqv_[3:-2, :, :] - sqv_[2:-3, :, :]) -
												   5. * (sqv_[4:-1, :, :] - sqv_[1:-4, :, :]) +
												   1. * (sqv_[  5:, :, :] - sqv_[ :-5, :, :]))
	k0_sqv = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	sqv1 = sqv + a1 * k0_sqv
	assert 'isentropic_density_of_water_vapor' in raw_state_1.keys()
	assert np.allclose(sqv1, raw_state_1['isentropic_density_of_water_vapor'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqc_[3:-2, :, :] + sqc_[2:-3, :, :]) -
									8. * (sqc_[4:-1, :, :] + sqc_[1:-4, :, :]) +
									1. * (sqc_[  5:, :, :] + sqc_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqc_[3:-2, :, :] - sqc_[2:-3, :, :]) -
												   5. * (sqc_[4:-1, :, :] - sqc_[1:-4, :, :]) +
												   1. * (sqc_[  5:, :, :] - sqc_[ :-5, :, :]))
	k0_sqc = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	sqc1 = sqc + a1 * k0_sqc
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_1.keys()
	assert np.allclose(sqc1, raw_state_1['isentropic_density_of_cloud_liquid_water'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqr_[3:-2, :, :] + sqr_[2:-3, :, :]) -
									8. * (sqr_[4:-1, :, :] + sqr_[1:-4, :, :]) +
									1. * (sqr_[  5:, :, :] + sqr_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqr_[3:-2, :, :] - sqr_[2:-3, :, :]) -
												   5. * (sqr_[4:-1, :, :] - sqr_[1:-4, :, :]) +
												   1. * (sqr_[  5:, :, :] - sqr_[ :-5, :, :]))
	k0_sqr = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	sqr1 = sqr + a1 * k0_sqr
	assert 'isentropic_density_of_precipitation_water' in raw_state_1.keys()
	assert np.allclose(sqr1, raw_state_1['isentropic_density_of_precipitation_water'])

	u1, v1 = hv.get_velocity_components(s1, su1, sv)
	raw_state_1['x_velocity_at_u_locations'] = u1
	raw_state_1['y_velocity_at_v_locations'] = v1

	raw_state_2 = ip_rk3.stage_call(1, dt, raw_state_1, raw_tendencies)

	s1_   = hb.from_physical_to_computational_domain(s1)
	u1_   = hb.from_physical_to_computational_domain(u1)
	su1_  = hb.from_physical_to_computational_domain(su1)
	sqv1_ = hb.from_physical_to_computational_domain(sqv1)
	sqc1_ = hb.from_physical_to_computational_domain(sqc1)
	sqr1_ = hb.from_physical_to_computational_domain(sqr1)

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (s1_[3:-2, :, :] + s1_[2:-3, :, :]) -
									 8. * (s1_[4:-1, :, :] + s1_[1:-4, :, :]) +
									 1. * (s1_[  5:, :, :] + s1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (s1_[3:-2, :, :] - s1_[2:-3, :, :]) -
													5. * (s1_[4:-1, :, :] - s1_[1:-4, :, :]) +
													1. * (s1_[  5:, :, :] - s1_[ :-5, :, :]))
	k1_s = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	s2 = s + b21 * k0_s + (a2 - b21) * k1_s
	assert 'air_isentropic_density' in raw_state_2.keys()
	assert np.allclose(s2, raw_state_2['air_isentropic_density'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (su1_[3:-2, :, :] + su1_[2:-3, :, :]) -
									 8. * (su1_[4:-1, :, :] + su1_[1:-4, :, :]) +
									 1. * (su1_[  5:, :, :] + su1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (su1_[3:-2, :, :] - su1_[2:-3, :, :]) -
													5. * (su1_[4:-1, :, :] - su1_[1:-4, :, :]) +
													1. * (su1_[  5:, :, :] - su1_[ :-5, :, :]))
	k1_su = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	su2 = su + b21 * k0_su + (a2 - b21) * k1_su
	assert 'x_momentum_isentropic' in raw_state_2.keys()
	assert np.allclose(su2, raw_state_2['x_momentum_isentropic'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqv1_[3:-2, :, :] + sqv1_[2:-3, :, :]) -
									 8. * (sqv1_[4:-1, :, :] + sqv1_[1:-4, :, :]) +
									 1. * (sqv1_[  5:, :, :] + sqv1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqv1_[3:-2, :, :] - sqv1_[2:-3, :, :]) -
													5. * (sqv1_[4:-1, :, :] - sqv1_[1:-4, :, :]) +
													1. * (sqv1_[  5:, :, :] - sqv1_[ :-5, :, :]))
	k1_sqv = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	sqv2 = sqv + b21 * k0_sqv + (a2 - b21) * k1_sqv
	assert 'isentropic_density_of_water_vapor' in raw_state_2.keys()
	assert np.allclose(sqv2, raw_state_2['isentropic_density_of_water_vapor'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqc1_[3:-2, :, :] + sqc1_[2:-3, :, :]) -
									 8. * (sqc1_[4:-1, :, :] + sqc1_[1:-4, :, :]) +
									 1. * (sqc1_[  5:, :, :] + sqc1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqc1_[3:-2, :, :] - sqc1_[2:-3, :, :]) -
													5. * (sqc1_[4:-1, :, :] - sqc1_[1:-4, :, :]) +
													1. * (sqc1_[  5:, :, :] - sqc1_[ :-5, :, :]))
	k1_sqc = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	sqc2 = sqc + b21 * k0_sqc + (a2 - b21) * k1_sqc
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_2.keys()
	assert np.allclose(sqc2, raw_state_2['isentropic_density_of_cloud_liquid_water'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqr1_[3:-2, :, :] + sqr1_[2:-3, :, :]) -
									 8. * (sqr1_[4:-1, :, :] + sqr1_[1:-4, :, :]) +
									 1. * (sqr1_[  5:, :, :] + sqr1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqr1_[3:-2, :, :] - sqr1_[2:-3, :, :]) -
													5. * (sqr1_[4:-1, :, :] - sqr1_[1:-4, :, :]) +
													1. * (sqr1_[  5:, :, :] - sqr1_[ :-5, :, :]))
	k1_sqr = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	sqr2 = sqr + b21 * k0_sqr + (a2 - b21) * k1_sqr
	assert 'isentropic_density_of_precipitation_water' in raw_state_2.keys()
	assert np.allclose(sqr2, raw_state_2['isentropic_density_of_precipitation_water'])

	u2, v2 = hv.get_velocity_components(s2, su2, sv)
	raw_state_2['x_velocity_at_u_locations'] = u2
	raw_state_2['y_velocity_at_v_locations'] = v2

	raw_state_3 = ip_rk3.stage_call(2, dt, raw_state_2, raw_tendencies)

	s2_   = hb.from_physical_to_computational_domain(s2)
	u2_   = hb.from_physical_to_computational_domain(u2)
	su2_  = hb.from_physical_to_computational_domain(su2)
	sqv2_ = hb.from_physical_to_computational_domain(sqv2)
	sqc2_ = hb.from_physical_to_computational_domain(sqc2)
	sqr2_ = hb.from_physical_to_computational_domain(sqr2)

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (s2_[3:-2, :, :] + s2_[2:-3, :, :]) -
									 8. * (s2_[4:-1, :, :] + s2_[1:-4, :, :]) +
									 1. * (s2_[  5:, :, :] + s2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (s2_[3:-2, :, :] - s2_[2:-3, :, :]) -
													5. * (s2_[4:-1, :, :] - s2_[1:-4, :, :]) +
													1. * (s2_[  5:, :, :] - s2_[ :-5, :, :]))
	s3 = s + g0 * k0_s + g1 * k1_s \
		 - g2 * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'air_isentropic_density' in raw_state_3.keys()
	assert np.allclose(s3, raw_state_3['air_isentropic_density'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (su2_[3:-2, :, :] + su2_[2:-3, :, :]) -
									 8. * (su2_[4:-1, :, :] + su2_[1:-4, :, :]) +
									 1. * (su2_[  5:, :, :] + su2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (su2_[3:-2, :, :] - su2_[2:-3, :, :]) -
													5. * (su2_[4:-1, :, :] - su2_[1:-4, :, :]) +
													1. * (su2_[  5:, :, :] - su2_[ :-5, :, :]))
	su3 = su + g0 * k0_su + g1 * k1_su \
		  - g2 * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'x_momentum_isentropic' in raw_state_3.keys()
	assert np.allclose(su3[:, 3:4, :], raw_state_3['x_momentum_isentropic'][:, 3:4, :])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqv2_[3:-2, :, :] + sqv2_[2:-3, :, :]) -
									 8. * (sqv2_[4:-1, :, :] + sqv2_[1:-4, :, :]) +
									 1. * (sqv2_[  5:, :, :] + sqv2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqv2_[3:-2, :, :] - sqv2_[2:-3, :, :]) -
													5. * (sqv2_[4:-1, :, :] - sqv2_[1:-4, :, :]) +
													1. * (sqv2_[  5:, :, :] - sqv2_[ :-5, :, :]))
	sqv3 = sqv + g0 * k0_sqv + g1 * k1_sqv \
		   - g2 * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_3.keys()
	assert np.allclose(sqv3, raw_state_3['isentropic_density_of_water_vapor'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqc2_[3:-2, :, :] + sqc2_[2:-3, :, :]) -
									 8. * (sqc2_[4:-1, :, :] + sqc2_[1:-4, :, :]) +
									 1. * (sqc2_[  5:, :, :] + sqc2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqc2_[3:-2, :, :] - sqc2_[2:-3, :, :]) -
													5. * (sqc2_[4:-1, :, :] - sqc2_[1:-4, :, :]) +
													1. * (sqc2_[  5:, :, :] - sqc2_[ :-5, :, :]))
	sqc3 = sqc + g0 * k0_sqc + g1 * k1_sqc \
		   - g2 * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_3.keys()
	assert np.allclose(sqc3, raw_state_3['isentropic_density_of_cloud_liquid_water'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqr2_[3:-2, :, :] + sqr2_[2:-3, :, :]) -
									 8. * (sqr2_[4:-1, :, :] + sqr2_[1:-4, :, :]) +
									 1. * (sqr2_[  5:, :, :] + sqr2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqr2_[3:-2, :, :] - sqr2_[2:-3, :, :]) -
													5. * (sqr2_[4:-1, :, :] - sqr2_[1:-4, :, :]) +
													1. * (sqr2_[  5:, :, :] - sqr2_[ :-5, :, :]))
	sqr3 = sqr + g0 * k0_sqr + g1 * k1_sqr \
		   - g2 * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_3.keys()
	assert np.allclose(sqr3, raw_state_3['isentropic_density_of_precipitation_water'])


def test_rk3cosmo_substepping(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	mode = 'xy'
	backend = gt.mode.NUMPY
	dtype = state['air_isentropic_density'].values.dtype

	hb = HorizontalBoundary.factory('periodic', grid)

	hv = HorizontalVelocity(grid, dtype=dtype)

	ip_rk3 = IsentropicPrognostic.factory(
		'rk3cosmo', mode, grid, True, hb,
		horizontal_flux_scheme='fifth_order_upwind', substeps=6,
		backend=backend, dtype=dtype
	)

	ip_rk3.substep_output_properties = {
		'air_isentropic_density': 'kg m^-2 K^-1',
		'x_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
		'y_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
		'mass_fraction_of_water_vapor_in_air': 'g g^-1',
		'mass_fraction_of_cloud_liquid_water_in_air': 'g g^-1',
		'mass_fraction_of_precipitation_water_in_air': 'g g^-1',
	}

	dt = timedelta(seconds=20)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mfwv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfcw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = {}

	#
	# Stage 1
	#
	raw_state_1 = ip_rk3.stage_call(0, dt, raw_state, raw_tendencies)

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	qv  = raw_state[mfwv]
	sqv = raw_state['isentropic_density_of_water_vapor']
	qc  = raw_state[mfcw]
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	qr  = raw_state[mfpw]
	sqr = raw_state['isentropic_density_of_precipitation_water']

	s_   = hb.from_physical_to_computational_domain(s)
	u_   = hb.from_physical_to_computational_domain(u)
	su_  = hb.from_physical_to_computational_domain(su)
	sqv_ = hb.from_physical_to_computational_domain(sqv)
	sqc_ = hb.from_physical_to_computational_domain(sqc)
	sqr_ = hb.from_physical_to_computational_domain(sqr)

	assert raw_state_1['time'] == state['time'] + 1/3 * dt

	flux6 = u_[3:-3, :, :] / 60. * (37. * (s_[3:-2, :, :] + s_[2:-3, :, :]) -
									8. * (s_[4:-1, :, :] + s_[1:-4, :, :]) +
									1. * (s_[  5:, :, :] + s_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (s_[3:-2, :, :] - s_[2:-3, :, :]) -
												   5. * (s_[4:-1, :, :] - s_[1:-4, :, :]) +
												   1. * (s_[  5:, :, :] - s_[ :-5, :, :]))
	s1 = s - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'air_isentropic_density' in raw_state_1.keys()
	assert np.allclose(s1, raw_state_1['air_isentropic_density'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (su_[3:-2, :, :] + su_[2:-3, :, :]) -
									8. * (su_[4:-1, :, :] + su_[1:-4, :, :]) +
									1. * (su_[  5:, :, :] + su_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (su_[3:-2, :, :] - su_[2:-3, :, :]) -
												   5. * (su_[4:-1, :, :] - su_[1:-4, :, :]) +
												   1. * (su_[  5:, :, :] - su_[ :-5, :, :]))
	su1 = su - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'x_momentum_isentropic' in raw_state_1.keys()
	assert np.allclose(su1, raw_state_1['x_momentum_isentropic'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqv_[3:-2, :, :] + sqv_[2:-3, :, :]) -
									8. * (sqv_[4:-1, :, :] + sqv_[1:-4, :, :]) +
									1. * (sqv_[  5:, :, :] + sqv_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqv_[3:-2, :, :] - sqv_[2:-3, :, :]) -
												   5. * (sqv_[4:-1, :, :] - sqv_[1:-4, :, :]) +
												   1. * (sqv_[  5:, :, :] - sqv_[ :-5, :, :]))
	sqv1 = sqv - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_1.keys()
	assert np.allclose(sqv1, raw_state_1['isentropic_density_of_water_vapor'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqc_[3:-2, :, :] + sqc_[2:-3, :, :]) -
									8. * (sqc_[4:-1, :, :] + sqc_[1:-4, :, :]) +
									1. * (sqc_[  5:, :, :] + sqc_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqc_[3:-2, :, :] - sqc_[2:-3, :, :]) -
												   5. * (sqc_[4:-1, :, :] - sqc_[1:-4, :, :]) +
												   1. * (sqc_[  5:, :, :] - sqc_[ :-5, :, :]))
	sqc1 = sqc - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_1.keys()
	assert np.allclose(sqc1, raw_state_1['isentropic_density_of_cloud_liquid_water'])

	flux6 = u_[3:-3, :, :] / 60. * (37. * (sqr_[3:-2, :, :] + sqr_[2:-3, :, :]) -
									8. * (sqr_[4:-1, :, :] + sqr_[1:-4, :, :]) +
									1. * (sqr_[  5:, :, :] + sqr_[ :-5, :, :]))
	flux = flux6 - np.abs(u_[3:-3, :, :]) / 60. * (10. * (sqr_[3:-2, :, :] - sqr_[2:-3, :, :]) -
												   5. * (sqr_[4:-1, :, :] - sqr_[1:-4, :, :]) +
												   1. * (sqr_[  5:, :, :] - sqr_[ :-5, :, :]))
	sqr1 = sqr - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_1.keys()
	assert np.allclose(sqr1, raw_state_1['isentropic_density_of_precipitation_water'])

	assert len(raw_state_1) == 7

	u1, v1 = hv.get_velocity_components(s1, su1, sv)
	raw_state_1['x_velocity_at_u_locations'] = u1
	raw_state_1['y_velocity_at_v_locations'] = v1
	raw_state_1[mfwv] = qv1 = sqv1 / s1
	raw_state_1[mfcw] = qc1 = sqc1 / s1
	raw_state_1[mfpw] = qr1 = sqr1 / s1

	#
	# Stage 1, substep 1
	#
	raw_state_00 = ip_rk3.substep_call(0, 0, dt, raw_state, raw_state_1, raw_state_1)

	assert np.isclose(
		(raw_state_00['time'] - raw_state['time']).total_seconds(), dt.total_seconds()/6
	)

	s00 = s + (s1 - s) / 2.0
	assert 'air_isentropic_density' in raw_state_00
	assert np.allclose(s00, raw_state_00['air_isentropic_density'])

	su00 = su + (su1 - su) / 2.0
	assert 'x_momentum_isentropic' in raw_state_00
	assert np.allclose(su00, raw_state_00['x_momentum_isentropic'])

	qv00 = qv + (qv1 - qv) / 2.0
	assert mfwv in raw_state_00
	assert np.allclose(qv00, raw_state_00[mfwv])

	qc00 = qc + (qc1 - qc) / 2.0
	assert mfcw in raw_state_00
	assert np.allclose(qc00, raw_state_00[mfcw])

	qr00 = qr + (qr1 - qr) / 2.0
	assert mfpw in raw_state_00
	assert np.allclose(qr00, raw_state_00[mfpw])

	assert len(raw_state_00) == 7

	#
	# Stage 1, substep 2
	#
	raw_state_01 = ip_rk3.substep_call(0, 1, dt, raw_state, raw_state_1, raw_state_00)

	assert np.isclose(
		(raw_state_01['time'] - raw_state['time']).total_seconds(), dt.total_seconds()/3
	)

	assert 'air_isentropic_density' in raw_state_01
	assert np.allclose(s1, raw_state_01['air_isentropic_density'])

	assert 'x_momentum_isentropic' in raw_state_01
	assert np.allclose(su1, raw_state_01['x_momentum_isentropic'])

	assert mfwv in raw_state_01
	assert np.allclose(qv1, raw_state_01[mfwv])

	assert mfcw in raw_state_01
	assert np.allclose(qc1, raw_state_01[mfcw])

	assert mfpw in raw_state_01
	assert np.allclose(qr1, raw_state_01[mfpw])

	assert len(raw_state_01) == 7

	#
	# Stage 2
	#
	raw_state_2 = ip_rk3.stage_call(1, dt, raw_state_1, raw_tendencies)

	s1_   = hb.from_physical_to_computational_domain(s1)
	u1_   = hb.from_physical_to_computational_domain(u1)
	su1_  = hb.from_physical_to_computational_domain(su1)
	sqv1_ = hb.from_physical_to_computational_domain(sqv1)
	sqc1_ = hb.from_physical_to_computational_domain(sqc1)
	sqr1_ = hb.from_physical_to_computational_domain(sqr1)

	assert raw_state_2['time'] == raw_state['time'] + 0.5*dt

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (s1_[3:-2, :, :] + s1_[2:-3, :, :]) -
									 8. * (s1_[4:-1, :, :] + s1_[1:-4, :, :]) +
									 1. * (s1_[  5:, :, :] + s1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (s1_[3:-2, :, :] - s1_[2:-3, :, :]) -
													5. * (s1_[4:-1, :, :] - s1_[1:-4, :, :]) +
													1. * (s1_[  5:, :, :] - s1_[ :-5, :, :]))
	s2 = s - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'air_isentropic_density' in raw_state_2.keys()
	assert np.allclose(s2, raw_state_2['air_isentropic_density'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (su1_[3:-2, :, :] + su1_[2:-3, :, :]) -
									 8. * (su1_[4:-1, :, :] + su1_[1:-4, :, :]) +
									 1. * (su1_[  5:, :, :] + su1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (su1_[3:-2, :, :] - su1_[2:-3, :, :]) -
													5. * (su1_[4:-1, :, :] - su1_[1:-4, :, :]) +
													1. * (su1_[  5:, :, :] - su1_[ :-5, :, :]))
	su2 = su - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'x_momentum_isentropic' in raw_state_2.keys()
	assert np.allclose(su2, raw_state_2['x_momentum_isentropic'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqv1_[3:-2, :, :] + sqv1_[2:-3, :, :]) -
									 8. * (sqv1_[4:-1, :, :] + sqv1_[1:-4, :, :]) +
									 1. * (sqv1_[  5:, :, :] + sqv1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqv1_[3:-2, :, :] - sqv1_[2:-3, :, :]) -
													5. * (sqv1_[4:-1, :, :] - sqv1_[1:-4, :, :]) +
													1. * (sqv1_[  5:, :, :] - sqv1_[ :-5, :, :]))
	sqv2 = sqv - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_2.keys()
	assert np.allclose(sqv2, raw_state_2['isentropic_density_of_water_vapor'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqc1_[3:-2, :, :] + sqc1_[2:-3, :, :]) -
									 8. * (sqc1_[4:-1, :, :] + sqc1_[1:-4, :, :]) +
									 1. * (sqc1_[  5:, :, :] + sqc1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqc1_[3:-2, :, :] - sqc1_[2:-3, :, :]) -
													5. * (sqc1_[4:-1, :, :] - sqc1_[1:-4, :, :]) +
													1. * (sqc1_[  5:, :, :] - sqc1_[ :-5, :, :]))
	sqc2 = sqc - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_2.keys()
	assert np.allclose(sqc2, raw_state_2['isentropic_density_of_cloud_liquid_water'])

	flux6 = u1_[3:-3, :, :] / 60. * (37. * (sqr1_[3:-2, :, :] + sqr1_[2:-3, :, :]) -
									 8. * (sqr1_[4:-1, :, :] + sqr1_[1:-4, :, :]) +
									 1. * (sqr1_[  5:, :, :] + sqr1_[ :-5, :, :]))
	flux = flux6 - np.abs(u1_[3:-3, :, :]) / 60. * (10. * (sqr1_[3:-2, :, :] - sqr1_[2:-3, :, :]) -
													5. * (sqr1_[4:-1, :, :] - sqr1_[1:-4, :, :]) +
													1. * (sqr1_[  5:, :, :] - sqr1_[ :-5, :, :]))
	sqr2 = sqr - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_2.keys()
	assert np.allclose(sqr2, raw_state_2['isentropic_density_of_precipitation_water'])

	u2, v2 = hv.get_velocity_components(s2, su2, sv)
	raw_state_2['x_velocity_at_u_locations'] = u2
	raw_state_2['y_velocity_at_v_locations'] = v2
	raw_state_2[mfwv] = qv2 = sqv2 / s2
	raw_state_2[mfcw] = qc2 = sqc2 / s2
	raw_state_2[mfpw] = qr2 = sqr2 / s2

	#
	# Stage 2, substep 1
	#
	raw_state_10 = ip_rk3.substep_call(1, 0, dt, raw_state, raw_state_2, raw_state_1)

	assert np.isclose(
		(raw_state_10['time'] - raw_state['time']).total_seconds(), dt.total_seconds()/6
	)

	s10 = s + (s2 - s) / 3.0
	assert 'air_isentropic_density' in raw_state_10
	assert np.allclose(s10, raw_state_10['air_isentropic_density'])

	su10 = su + (su2 - su) / 3.0
	assert 'x_momentum_isentropic' in raw_state_10
	assert np.allclose(su10, raw_state_10['x_momentum_isentropic'])

	qv10 = qv + (qv2 - qv) / 3.0
	assert mfwv in raw_state_10
	assert np.allclose(qv10, raw_state_10[mfwv])

	qc10 = qc + (qc2 - qc) / 3.0
	assert mfcw in raw_state_10
	assert np.allclose(qc10, raw_state_10[mfcw])

	qr10 = qr + (qr2 - qr) / 3.0
	assert mfpw in raw_state_10
	assert np.allclose(qr10, raw_state_10[mfpw])

	assert len(raw_state_10) == 7

	#
	# Stage 2, substep 2
	#
	raw_state_11 = ip_rk3.substep_call(1, 1, dt, raw_state, raw_state_2, raw_state_10)

	assert np.isclose(
		(raw_state_11['time'] - raw_state['time']).total_seconds(), dt.total_seconds()/3
	)

	s11 = s10 + (s2 - s) / 3.0
	assert 'air_isentropic_density' in raw_state_11
	assert np.allclose(s11, raw_state_11['air_isentropic_density'])

	su11 = su10 + (su2 - su) / 3.0
	assert 'x_momentum_isentropic' in raw_state_01
	assert np.allclose(su11, raw_state_11['x_momentum_isentropic'])

	qv11 = qv10 + (qv2 - qv) / 3.0
	assert mfwv in raw_state_11
	assert np.allclose(qv11, raw_state_11[mfwv])

	qc11 = qc10 + (qc2 - qc) / 3.0
	assert mfcw in raw_state_11
	assert np.allclose(qc11, raw_state_11[mfcw])

	qr11 = qr10 + (qr2 - qr) / 3.0
	assert mfpw in raw_state_11
	assert np.allclose(qr11, raw_state_11[mfpw])

	assert len(raw_state_11) == 7

	#
	# Stage 2, substep 3
	#
	raw_state_12 = ip_rk3.substep_call(1, 2, dt, raw_state, raw_state_2, raw_state_11)

	assert np.isclose(
		(raw_state_12['time'] - raw_state['time']).total_seconds(), dt.total_seconds()/2
	)

	assert 'air_isentropic_density' in raw_state_12
	assert np.allclose(s2, raw_state_12['air_isentropic_density'])

	assert 'x_momentum_isentropic' in raw_state_12
	assert np.allclose(su2, raw_state_12['x_momentum_isentropic'])

	assert mfwv in raw_state_12
	assert np.allclose(qv2, raw_state_12[mfwv])

	assert mfcw in raw_state_12
	assert np.allclose(qc2, raw_state_12[mfcw])

	assert mfpw in raw_state_12
	assert np.allclose(qr2, raw_state_12[mfpw])

	assert len(raw_state_12) == 7

	#
	# Stage 3
	#
	raw_state_3 = ip_rk3.stage_call(2, dt, raw_state_2, raw_tendencies)

	s2_   = hb.from_physical_to_computational_domain(s2)
	u2_   = hb.from_physical_to_computational_domain(u2)
	su2_  = hb.from_physical_to_computational_domain(su2)
	sqv2_ = hb.from_physical_to_computational_domain(sqv2)
	sqc2_ = hb.from_physical_to_computational_domain(sqc2)
	sqr2_ = hb.from_physical_to_computational_domain(sqr2)

	assert raw_state_3['time'] == raw_state['time'] + dt

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (s2_[3:-2, :, :] + s2_[2:-3, :, :]) -
									 8. * (s2_[4:-1, :, :] + s2_[1:-4, :, :]) +
									 1. * (s2_[  5:, :, :] + s2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (s2_[3:-2, :, :] - s2_[2:-3, :, :]) -
													5. * (s2_[4:-1, :, :] - s2_[1:-4, :, :]) +
													1. * (s2_[  5:, :, :] - s2_[ :-5, :, :]))
	s3 = s - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'air_isentropic_density' in raw_state_3.keys()
	assert np.allclose(s3, raw_state_3['air_isentropic_density'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (su2_[3:-2, :, :] + su2_[2:-3, :, :]) -
									 8. * (su2_[4:-1, :, :] + su2_[1:-4, :, :]) +
									 1. * (su2_[  5:, :, :] + su2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (su2_[3:-2, :, :] - su2_[2:-3, :, :]) -
													5. * (su2_[4:-1, :, :] - su2_[1:-4, :, :]) +
													1. * (su2_[  5:, :, :] - su2_[ :-5, :, :]))
	su3 = su - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'x_momentum_isentropic' in raw_state_3.keys()
	assert np.allclose(su3[:, 3:4, :], raw_state_3['x_momentum_isentropic'][:, 3:4, :])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqv2_[3:-2, :, :] + sqv2_[2:-3, :, :]) -
									 8. * (sqv2_[4:-1, :, :] + sqv2_[1:-4, :, :]) +
									 1. * (sqv2_[  5:, :, :] + sqv2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqv2_[3:-2, :, :] - sqv2_[2:-3, :, :]) -
													5. * (sqv2_[4:-1, :, :] - sqv2_[1:-4, :, :]) +
													1. * (sqv2_[  5:, :, :] - sqv2_[ :-5, :, :]))
	sqv3 = sqv - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_3.keys()
	assert np.allclose(sqv3, raw_state_3['isentropic_density_of_water_vapor'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqc2_[3:-2, :, :] + sqc2_[2:-3, :, :]) -
									 8. * (sqc2_[4:-1, :, :] + sqc2_[1:-4, :, :]) +
									 1. * (sqc2_[  5:, :, :] + sqc2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqc2_[3:-2, :, :] - sqc2_[2:-3, :, :]) -
													5. * (sqc2_[4:-1, :, :] - sqc2_[1:-4, :, :]) +
													1. * (sqc2_[  5:, :, :] - sqc2_[ :-5, :, :]))
	sqc3 = sqc - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_3.keys()
	assert np.allclose(sqc3, raw_state_3['isentropic_density_of_cloud_liquid_water'])

	flux6 = u2_[3:-3, :, :] / 60. * (37. * (sqr2_[3:-2, :, :] + sqr2_[2:-3, :, :]) -
									 8. * (sqr2_[4:-1, :, :] + sqr2_[1:-4, :, :]) +
									 1. * (sqr2_[  5:, :, :] + sqr2_[ :-5, :, :]))
	flux = flux6 - np.abs(u2_[3:-3, :, :]) / 60. * (10. * (sqr2_[3:-2, :, :] - sqr2_[2:-3, :, :]) -
													5. * (sqr2_[4:-1, :, :] - sqr2_[1:-4, :, :]) +
													1. * (sqr2_[  5:, :, :] - sqr2_[ :-5, :, :]))
	sqr3 = sqr - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_3.keys()
	assert np.allclose(sqr3, raw_state_3['isentropic_density_of_precipitation_water'])

	assert len(raw_state_3) == 7

	raw_state_3[mfwv] = qv3 = sqv3 / s3
	raw_state_3[mfcw] = qc3 = sqc3 / s3
	raw_state_3[mfpw] = qr3 = sqr3 / s3

	#
	# Stage 3, substep 1
	#
	raw_state_20 = ip_rk3.substep_call(2, 0, dt, raw_state, raw_state_3, raw_state_10)

	assert np.isclose(
		(raw_state_20['time'] - raw_state['time']).total_seconds(), 1*dt.total_seconds()/6
	)

	s20 = s + (s3 - s) / 6.0
	assert 'air_isentropic_density' in raw_state_20
	assert np.allclose(s20, raw_state_20['air_isentropic_density'])

	su20 = su + (su3 - su) / 6.0
	assert 'x_momentum_isentropic' in raw_state_20
	assert np.allclose(su20, raw_state_20['x_momentum_isentropic'])

	qv20 = qv + (qv3 - qv) / 6.0
	assert mfwv in raw_state_20
	assert np.allclose(qv20, raw_state_20[mfwv])

	qc20 = qc + (qc3 - qc) / 6.0
	assert mfcw in raw_state_20
	assert np.allclose(qc20, raw_state_20[mfcw])

	qr20 = qr + (qr3 - qr) / 6.0
	assert mfpw in raw_state_20
	assert np.allclose(qr20, raw_state_20[mfpw])

	assert len(raw_state_20) == 7

	#
	# Stage 3, substep 2
	#
	raw_state_21 = ip_rk3.substep_call(2, 1, dt, raw_state, raw_state_3, raw_state_20)

	assert np.isclose(
		(raw_state_21['time'] - raw_state['time']).total_seconds(), 2*dt.total_seconds()/6
	)

	s21 = s20 + (s3 - s) / 6.0
	assert 'air_isentropic_density' in raw_state_21
	assert np.allclose(s21, raw_state_21['air_isentropic_density'])

	su21 = su20 + (su3 - su) / 6.0
	assert 'x_momentum_isentropic' in raw_state_21
	assert np.allclose(su21, raw_state_21['x_momentum_isentropic'])

	qv21 = qv20 + (qv3 - qv) / 6.0
	assert mfwv in raw_state_21
	assert np.allclose(qv21, raw_state_21[mfwv])

	qc21 = qc20 + (qc3 - qc) / 6.0
	assert mfcw in raw_state_21
	assert np.allclose(qc21, raw_state_21[mfcw])

	qr21 = qr20 + (qr3 - qr) / 6.0
	assert mfpw in raw_state_21
	assert np.allclose(qr21, raw_state_21[mfpw])

	assert len(raw_state_21) == 7

	#
	# Stage 3, substep 3
	#
	raw_state_22 = ip_rk3.substep_call(2, 2, dt, raw_state, raw_state_3, raw_state_21)

	assert np.isclose(
		(raw_state_22['time'] - raw_state['time']).total_seconds(), 3*dt.total_seconds()/6
	)

	s22 = s21 + (s3 - s) / 6.0
	assert 'air_isentropic_density' in raw_state_22
	assert np.allclose(s22, raw_state_21['air_isentropic_density'])

	su22 = su21 + (su3 - su) / 6.0
	assert 'x_momentum_isentropic' in raw_state_22
	assert np.allclose(su22, raw_state_22['x_momentum_isentropic'])

	qv22 = qv21 + (qv3 - qv) / 6.0
	assert mfwv in raw_state_22
	assert np.allclose(qv22, raw_state_22[mfwv])

	qc22 = qc21 + (qc3 - qc) / 6.0
	assert mfcw in raw_state_22
	assert np.allclose(qc22, raw_state_22[mfcw])

	qr22 = qr21 + (qr3 - qr) / 6.0
	assert mfpw in raw_state_22
	assert np.allclose(qr22, raw_state_22[mfpw])

	assert len(raw_state_21) == 7

	#
	# Stage 3, substep 4
	#
	raw_state_23 = ip_rk3.substep_call(2, 3, dt, raw_state, raw_state_3, raw_state_22)

	assert np.isclose(
		(raw_state_23['time'] - raw_state['time']).total_seconds(), 4*dt.total_seconds()/6
	)

	s23 = s22 + (s3 - s) / 6.0
	assert 'air_isentropic_density' in raw_state_23
	assert np.allclose(s23, raw_state_23['air_isentropic_density'])

	su23 = su22 + (su3 - su) / 6.0
	assert 'x_momentum_isentropic' in raw_state_23
	assert np.allclose(su23, raw_state_23['x_momentum_isentropic'])

	qv23 = qv22 + (qv3 - qv) / 6.0
	assert mfwv in raw_state_23
	assert np.allclose(qv23, raw_state_23[mfwv])

	qc23 = qc22 + (qc3 - qc) / 6.0
	assert mfcw in raw_state_23
	assert np.allclose(qc23, raw_state_23[mfcw])

	qr23 = qr22 + (qr3 - qr) / 6.0
	assert mfpw in raw_state_23
	assert np.allclose(qr23, raw_state_23[mfpw])

	assert len(raw_state_23) == 7

	#
	# Stage 3, substep 5
	#
	raw_state_24 = ip_rk3.substep_call(2, 4, dt, raw_state, raw_state_3, raw_state_23)

	assert np.isclose(
		(raw_state_24['time'] - raw_state['time']).total_seconds(), 5*dt.total_seconds()/6
	)

	s24 = s23 + (s3 - s) / 6.0
	assert 'air_isentropic_density' in raw_state_24
	assert np.allclose(s24, raw_state_24['air_isentropic_density'])

	su24 = su23 + (su3 - su) / 6.0
	assert 'x_momentum_isentropic' in raw_state_24
	assert np.allclose(su24, raw_state_24['x_momentum_isentropic'])

	qv24 = qv23 + (qv3 - qv) / 6.0
	assert mfwv in raw_state_24
	assert np.allclose(qv24, raw_state_24[mfwv])

	qc24 = qc23 + (qc3 - qc) / 6.0
	assert mfcw in raw_state_24
	assert np.allclose(qc24, raw_state_24[mfcw])

	qr24 = qr23 + (qr3 - qr) / 6.0
	assert mfpw in raw_state_24
	assert np.allclose(qr24, raw_state_24[mfpw])

	assert len(raw_state_24) == 7

	#
	# Stage 3, substep 6
	#
	raw_state_25 = ip_rk3.substep_call(2, 5, dt, raw_state, raw_state_3, raw_state_24)

	assert np.isclose(
		(raw_state_25['time'] - raw_state['time']).total_seconds(), dt.total_seconds()
	)

	assert 'air_isentropic_density' in raw_state_25
	assert np.allclose(s3, raw_state_25['air_isentropic_density'])

	assert 'x_momentum_isentropic' in raw_state_25
	assert np.allclose(su3, raw_state_25['x_momentum_isentropic'])

	assert mfwv in raw_state_25
	assert np.allclose(qv3, raw_state_25[mfwv])

	assert mfcw in raw_state_25
	assert np.allclose(qc3, raw_state_25[mfcw])

	assert mfpw in raw_state_25
	assert np.allclose(qr3, raw_state_25[mfpw])

	assert len(raw_state_25) == 7


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_moist_data
	#test_rk3cosmo_substepping(isentropic_moist_data())
