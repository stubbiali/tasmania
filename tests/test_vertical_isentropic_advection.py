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
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.physics.tendencies import VerticalIsentropicAdvection
from tasmania.python.utils.data_utils import make_dataarray_3d


mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


def test_upwind(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	dz = grid.dz.to_units('K').values.item()

	backend = gt.mode.NUMPY

	#
	# tendency_of_air_potential_temperature_on_interface_levels=False
	#
	fluxer = VerticalIsentropicAdvection(
		grid, True, 'upwind', backend=backend,
		tendency_of_air_potential_temperature_on_interface_levels=False
	)

	assert 'tendency_of_air_potential_temperature' in fluxer.input_properties
	assert 'air_isentropic_density' in fluxer.input_properties
	assert 'x_momentum_isentropic' in fluxer.input_properties
	assert 'y_momentum_isentropic' in fluxer.input_properties
	assert mfwv in fluxer.input_properties
	assert mfcw in fluxer.input_properties
	assert mfpw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mfwv in fluxer.tendency_properties
	assert mfcw in fluxer.tendency_properties
	assert mfpw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz)
	wm = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
	state['tendency_of_air_potential_temperature'] = make_dataarray_3d(w, grid, 'K s^-1')

	s  = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv = state[mfwv].to_units('g g^-1').values[...]
	sqv = s * qv
	qc = state[mfcw].to_units('g g^-1').values[...]
	sqc = s * qc
	qr = state[mfpw].to_units('g g^-1').values[...]
	sqr = s * qr

	tendencies, diagnostics = fluxer(state)

	flux_s = (wm > 0) * wm * s[:, :, 1:] - (wm < 0) * wm * s[:, :, :-1]
	tnd_s = - (flux_s[:, :, :-1] - flux_s[:, :, 1:]) / dz
	assert 'air_isentropic_density' in tendencies
	assert np.allclose(tnd_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	flux_su = (wm > 0) * wm * su[:, :, 1:] - (wm < 0) * wm * su[:, :, :-1]
	tnd_su = - (flux_su[:, :, :-1] - flux_su[:, :, 1:]) / dz
	assert 'x_momentum_isentropic' in tendencies
	assert np.allclose(tnd_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	flux_sv = (wm > 0) * wm * sv[:, :, 1:] - (wm < 0) * wm * sv[:, :, :-1]
	tnd_sv = - (flux_sv[:, :, :-1] - flux_sv[:, :, 1:]) / dz
	assert 'y_momentum_isentropic' in tendencies
	assert np.allclose(tnd_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	flux_sqv = (wm > 0) * wm * sqv[:, :, 1:] - (wm < 0) * wm * sqv[:, :, :-1]
	tnd_qv = - (flux_sqv[:, :, :-1] - flux_sqv[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfwv in tendencies
	assert np.allclose(tnd_qv, tendencies[mfwv][:, :, 1:-1])
	tendencies.pop(mfwv)

	flux_sqc = (wm > 0) * wm * sqc[:, :, 1:] - (wm < 0) * wm * sqc[:, :, :-1]
	tnd_qc = - (flux_sqc[:, :, :-1] - flux_sqc[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfcw in tendencies
	assert np.allclose(tnd_qc, tendencies[mfcw][:, :, 1:-1])
	tendencies.pop(mfcw)

	flux_sqr = (wm > 0) * wm * sqr[:, :, 1:] - (wm < 0) * wm * sqr[:, :, :-1]
	tnd_qr = - (flux_sqr[:, :, :-1] - flux_sqr[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfpw in tendencies
	assert np.allclose(tnd_qr, tendencies[mfpw][:, :, 1:-1])
	tendencies.pop(mfpw)

	assert tendencies == {}

	assert diagnostics == {}

	#
	# tendency_of_air_potential_temperature_on_interface_levels=True
	#
	fluxer = VerticalIsentropicAdvection(
		grid, True, 'upwind', backend=backend,
		tendency_of_air_potential_temperature_on_interface_levels=True
	)

	assert 'tendency_of_air_potential_temperature_on_interface_levels' in fluxer.input_properties
	assert 'air_isentropic_density' in fluxer.input_properties
	assert 'x_momentum_isentropic' in fluxer.input_properties
	assert 'y_momentum_isentropic' in fluxer.input_properties
	assert mfwv in fluxer.input_properties
	assert mfcw in fluxer.input_properties
	assert mfpw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mfwv in fluxer.tendency_properties
	assert mfcw in fluxer.tendency_properties
	assert mfpw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_dataarray_3d(w, grid, 'K s^-1')

	tendencies, diagnostics = fluxer(state)

	flux_s = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * s[:, :, 1:] - \
			 (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * s[:, :, :-1]
	tnd_s = - (flux_s[:, :, :-1] - flux_s[:, :, 1:]) / dz
	assert 'air_isentropic_density' in tendencies
	assert np.allclose(tnd_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	flux_su = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * su[:, :, 1:] - \
			  (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * su[:, :, :-1]
	tnd_su = - (flux_su[:, :, :-1] - flux_su[:, :, 1:]) / dz
	assert 'x_momentum_isentropic' in tendencies
	assert np.allclose(tnd_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	flux_sv = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * sv[:, :, 1:] - \
			  (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * sv[:, :, :-1]
	tnd_sv = - (flux_sv[:, :, :-1] - flux_sv[:, :, 1:]) / dz
	assert 'y_momentum_isentropic' in tendencies
	assert np.allclose(tnd_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	flux_sqv = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * sqv[:, :, 1:] - \
			   (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * sqv[:, :, :-1]
	tnd_qv = - (flux_sqv[:, :, :-1] - flux_sqv[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfwv in tendencies
	assert np.allclose(tnd_qv, tendencies[mfwv][:, :, 1:-1])
	tendencies.pop(mfwv)

	flux_sqc = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * sqc[:, :, 1:] - \
			   (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * sqc[:, :, :-1]
	tnd_qc = - (flux_sqc[:, :, :-1] - flux_sqc[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfcw in tendencies
	assert np.allclose(tnd_qc, tendencies[mfcw][:, :, 1:-1])
	tendencies.pop(mfcw)

	flux_sqr = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * sqr[:, :, 1:] - \
			   (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * sqr[:, :, :-1]
	tnd_qr = - (flux_sqr[:, :, :-1] - flux_sqr[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfpw in tendencies
	assert np.allclose(tnd_qr, tendencies[mfpw][:, :, 1:-1])
	tendencies.pop(mfpw)

	assert tendencies == {}

	assert diagnostics == {}


def test_centered(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	dz = grid.dz.to_units('K').values.item()

	backend = gt.mode.NUMPY

	#
	# tendency_of_air_potential_temperature_on_interface_levels=False
	#
	fluxer = VerticalIsentropicAdvection(
		grid, True, 'centered', backend=backend,
		tendency_of_air_potential_temperature_on_interface_levels=False
	)

	assert 'tendency_of_air_potential_temperature' in fluxer.input_properties
	assert 'air_isentropic_density' in fluxer.input_properties
	assert 'x_momentum_isentropic' in fluxer.input_properties
	assert 'y_momentum_isentropic' in fluxer.input_properties
	assert mfwv in fluxer.input_properties
	assert mfcw in fluxer.input_properties
	assert mfpw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mfwv in fluxer.tendency_properties
	assert mfcw in fluxer.tendency_properties
	assert mfpw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz)
	wm = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
	state['tendency_of_air_potential_temperature'] = make_dataarray_3d(w, grid, 'K s^-1')

	s   = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su  = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv  = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv  = state[mfwv].to_units('g g^-1').values[...]
	sqv = s * qv
	qc  = state[mfcw].to_units('g g^-1').values[...]
	sqc = s * qc
	qr  = state[mfpw].to_units('g g^-1').values[...]
	sqr = s * qr

	tendencies, diagnostics = fluxer(state)

	flux_s = wm * 0.5 * (s[:, :, 1:] + s[:, :, :-1])
	tnd_s = - (flux_s[:, :, :-1] - flux_s[:, :, 1:]) / dz
	assert 'air_isentropic_density' in tendencies
	assert np.allclose(tnd_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	flux_su = wm * 0.5 * (su[:, :, 1:] + su[:, :, :-1])
	tnd_su = - (flux_su[:, :, :-1] - flux_su[:, :, 1:]) / dz
	assert 'x_momentum_isentropic' in tendencies
	assert np.allclose(tnd_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	flux_sv = wm * 0.5 * (sv[:, :, 1:] + sv[:, :, :-1])
	tnd_sv = - (flux_sv[:, :, :-1] - flux_sv[:, :, 1:]) / dz
	assert 'y_momentum_isentropic' in tendencies
	assert np.allclose(tnd_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	flux_sqv = wm * 0.5 * (sqv[:, :, 1:] + sqv[:, :, :-1])
	tnd_qv = - (flux_sqv[:, :, :-1] - flux_sqv[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfwv in tendencies
	assert np.allclose(tnd_qv, tendencies[mfwv][:, :, 1:-1])
	tendencies.pop(mfwv)

	flux_sqc = wm * 0.5 * (sqc[:, :, 1:] + sqc[:, :, :-1])
	tnd_qc = - (flux_sqc[:, :, :-1] - flux_sqc[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfcw in tendencies
	assert np.allclose(tnd_qc, tendencies[mfcw][:, :, 1:-1])
	tendencies.pop(mfcw)

	flux_sqr = wm * 0.5 * (sqr[:, :, 1:] + sqr[:, :, :-1])
	tnd_qr = - (flux_sqr[:, :, :-1] - flux_sqr[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfpw in tendencies
	assert np.allclose(tnd_qr, tendencies[mfpw][:, :, 1:-1])
	tendencies.pop(mfpw)

	assert tendencies == {}

	assert diagnostics == {}

	#
	# tendency_of_air_potential_temperature_on_interface_levels=True
	#
	fluxer = VerticalIsentropicAdvection(
		grid, True, 'centered', backend=backend,
		tendency_of_air_potential_temperature_on_interface_levels=True
	)

	assert 'tendency_of_air_potential_temperature_on_interface_levels' in fluxer.input_properties
	assert 'air_isentropic_density' in fluxer.input_properties
	assert 'x_momentum_isentropic' in fluxer.input_properties
	assert 'y_momentum_isentropic' in fluxer.input_properties
	assert mfwv in fluxer.input_properties
	assert mfcw in fluxer.input_properties
	assert mfpw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mfwv in fluxer.tendency_properties
	assert mfcw in fluxer.tendency_properties
	assert mfpw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_dataarray_3d(w, grid, 'K s^-1')

	tendencies, diagnostics = fluxer(state)

	flux_s = w[:, :, 1:-1] * 0.5 * (s[:, :, 1:] + s[:, :, :-1])
	tnd_s = - (flux_s[:, :, :-1] - flux_s[:, :, 1:]) / dz
	assert 'air_isentropic_density' in tendencies
	assert np.allclose(tnd_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	flux_su = w[:, :, 1:-1] * 0.5 * (su[:, :, 1:] + su[:, :, :-1])
	tnd_su = - (flux_su[:, :, :-1] - flux_su[:, :, 1:]) / dz
	assert 'x_momentum_isentropic' in tendencies
	assert np.allclose(tnd_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	flux_sv = w[:, :, 1:-1] * 0.5 * (sv[:, :, 1:] + sv[:, :, :-1])
	tnd_sv = - (flux_sv[:, :, :-1] - flux_sv[:, :, 1:]) / dz
	assert 'y_momentum_isentropic' in tendencies
	assert np.allclose(tnd_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	flux_sqv = w[:, :, 1:-1] * 0.5 * (sqv[:, :, 1:] + sqv[:, :, :-1])
	tnd_qv = - (flux_sqv[:, :, :-1] - flux_sqv[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfwv in tendencies
	assert np.allclose(tnd_qv, tendencies[mfwv][:, :, 1:-1])
	tendencies.pop(mfwv)

	flux_sqc = w[:, :, 1:-1] * 0.5 * (sqc[:, :, 1:] + sqc[:, :, :-1])
	tnd_qc = - (flux_sqc[:, :, :-1] - flux_sqc[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfcw in tendencies
	assert np.allclose(tnd_qc, tendencies[mfcw][:, :, 1:-1])
	tendencies.pop(mfcw)

	flux_sqr = w[:, :, 1:-1] * 0.5 * (sqr[:, :, 1:] + sqr[:, :, :-1])
	tnd_qr = - (flux_sqr[:, :, :-1] - flux_sqr[:, :, 1:]) / (dz * s[:, :, 1:-1])
	assert mfpw in tendencies
	assert np.allclose(tnd_qr, tendencies[mfpw][:, :, 1:-1])
	tendencies.pop(mfpw)

	assert tendencies == {}

	assert diagnostics == {}


def test_third_order_upwind(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	dz = grid.dz.to_units('K').values.item()

	backend = gt.mode.NUMPY

	#
	# tendency_of_air_potential_temperature_on_interface_levels=True
	#
	fluxer = VerticalIsentropicAdvection(
		grid, True, 'third_order_upwind', backend=backend,
		tendency_of_air_potential_temperature_on_interface_levels=True,
	)

	assert 'tendency_of_air_potential_temperature_on_interface_levels' in fluxer.input_properties
	assert 'air_isentropic_density' in fluxer.input_properties
	assert 'x_momentum_isentropic' in fluxer.input_properties
	assert 'y_momentum_isentropic' in fluxer.input_properties
	assert mfwv in fluxer.input_properties
	assert mfcw in fluxer.input_properties
	assert mfpw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mfwv in fluxer.tendency_properties
	assert mfcw in fluxer.tendency_properties
	assert mfpw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_dataarray_3d(w, grid, 'K s^-1')

	s   = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su  = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv  = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv  = state[mfwv].to_units('g g^-1').values[...]
	sqv = s * qv
	qc  = state[mfcw].to_units('g g^-1').values[...]
	sqc = s * qc
	qr  = state[mfpw].to_units('g g^-1').values[...]
	sqr = s * qr

	tendencies, diagnostics = fluxer(state)

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (s[:, :, 1:-2] + s[:, :, 2:-1]) -
									1.0 * (s[:, :, :-3] + s[:, :, 3:]))
	flux_s = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (s[:, :, 1:-2] - s[:, :, 2:-1]) -
													 1.0 * (s[:, :, :-3] - s[:, :, 3:]))
	tnd_s = - (flux_s[:, :, :-1] - flux_s[:, :, 1:]) / dz
	assert 'air_isentropic_density' in tendencies
	assert np.allclose(tnd_s, tendencies['air_isentropic_density'][:, :, 2:-2])
	tendencies.pop('air_isentropic_density')

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (su[:, :, 1:-2] + su[:, :, 2:-1]) -
									1.0 * (su[:, :, :-3] + su[:, :, 3:]))
	flux_su = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (su[:, :, 1:-2] - su[:, :, 2:-1]) -
													 1.0 * (su[:, :, :-3] - su[:, :, 3:]))
	tnd_su = - (flux_su[:, :, :-1] - flux_su[:, :, 1:]) / dz
	assert 'x_momentum_isentropic' in tendencies
	assert np.allclose(tnd_su, tendencies['x_momentum_isentropic'][:, :, 2:-2])
	tendencies.pop('x_momentum_isentropic')

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (sv[:, :, 1:-2] + sv[:, :, 2:-1]) -
									1.0 * (sv[:, :, :-3] + sv[:, :, 3:]))
	flux_sv = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (sv[:, :, 1:-2] - sv[:, :, 2:-1]) -
													  1.0 * (sv[:, :, :-3] - sv[:, :, 3:]))
	tnd_sv = - (flux_sv[:, :, :-1] - flux_sv[:, :, 1:]) / dz
	assert 'y_momentum_isentropic' in tendencies
	assert np.allclose(tnd_sv, tendencies['y_momentum_isentropic'][:, :, 2:-2])
	tendencies.pop('y_momentum_isentropic')

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (sqv[:, :, 1:-2] + sqv[:, :, 2:-1]) -
									1.0 * (sqv[:, :, :-3] + sqv[:, :, 3:]))
	flux_sqv = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (sqv[:, :, 1:-2] - sqv[:, :, 2:-1]) -
													  1.0 * (sqv[:, :, :-3] - sqv[:, :, 3:]))
	tnd_qv = - (flux_sqv[:, :, :-1] - flux_sqv[:, :, 1:]) / (dz * s[:, :, 2:-2])
	assert mfwv in tendencies
	assert np.allclose(tnd_qv, tendencies[mfwv][:, :, 2:-2])
	tendencies.pop(mfwv)

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (sqc[:, :, 1:-2] + sqc[:, :, 2:-1]) -
									1.0 * (sqc[:, :, :-3] + sqc[:, :, 3:]))
	flux_sqc = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (sqc[:, :, 1:-2] - sqc[:, :, 2:-1]) -
													  1.0 * (sqc[:, :, :-3] - sqc[:, :, 3:]))
	tnd_qc = - (flux_sqc[:, :, :-1] - flux_sqc[:, :, 1:]) / (dz * s[:, :, 2:-2])
	assert mfcw in tendencies
	assert np.allclose(tnd_qc, tendencies[mfcw][:, :, 2:-2])
	tendencies.pop(mfcw)

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (sqr[:, :, 1:-2] + sqr[:, :, 2:-1]) -
									1.0 * (sqr[:, :, :-3] + sqr[:, :, 3:]))
	flux_sqr = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (sqr[:, :, 1:-2] - sqr[:, :, 2:-1]) -
													  1.0 * (sqr[:, :, :-3] - sqr[:, :, 3:]))
	tnd_qr = - (flux_sqr[:, :, :-1] - flux_sqr[:, :, 1:]) / (dz * s[:, :, 2:-2])
	assert mfpw in tendencies
	assert np.allclose(tnd_qr, tendencies[mfpw][:, :, 2:-2])
	tendencies.pop(mfpw)

	assert tendencies == {}

	assert diagnostics == {}


def test_fifth_order_upwind(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	dz = grid.dz.to_units('K').values.item()

	backend = gt.mode.NUMPY

	#
	# tendency_of_air_potential_temperature_on_interface_levels=True
	#
	fluxer = VerticalIsentropicAdvection(
		grid, True, 'fifth_order_upwind', backend=backend,
		tendency_of_air_potential_temperature_on_interface_levels=True,
	)

	assert 'tendency_of_air_potential_temperature_on_interface_levels' in fluxer.input_properties
	assert 'air_isentropic_density' in fluxer.input_properties
	assert 'x_momentum_isentropic' in fluxer.input_properties
	assert 'y_momentum_isentropic' in fluxer.input_properties
	assert mfwv in fluxer.input_properties
	assert mfcw in fluxer.input_properties
	assert mfpw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mfwv in fluxer.tendency_properties
	assert mfcw in fluxer.tendency_properties
	assert mfpw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_dataarray_3d(w, grid, 'K s^-1')

	s   = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su  = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv  = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv  = state[mfwv].to_units('g g^-1').values[...]
	sqv = s * qv
	qc  = state[mfcw].to_units('g g^-1').values[...]
	sqc = s * qc
	qr  = state[mfpw].to_units('g g^-1').values[...]
	sqr = s * qr

	tendencies, diagnostics = fluxer(state)

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (s[:, :, 2:-3] + s[:, :, 3:-2]) -
									 8.0 * (s[:, :, 1:-4] + s[:, :, 4:-1]) +
									 1.0 * (s[:, :, :-5] + s[:, :, 5:]))
	flux_s = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (s[:, :, 2:-3] - s[:, :, 3:-2]) -
													  5.0 * (s[:, :, 1:-4] - s[:, :, 4:-1]) +
													  1.0 * (s[:, :, :-5] - s[:, :, 5:]))
	tnd_s = - (flux_s[:, :, :-1] - flux_s[:, :, 1:]) / dz
	assert 'air_isentropic_density' in tendencies
	assert np.allclose(tnd_s, tendencies['air_isentropic_density'][:, :, 3:-3])
	tendencies.pop('air_isentropic_density')

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (su[:, :, 2:-3] + su[:, :, 3:-2]) -
									 8.0 * (su[:, :, 1:-4] + su[:, :, 4:-1]) +
									 1.0 * (su[:, :, :-5] + su[:, :, 5:]))
	flux_su = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (su[:, :, 2:-3] - su[:, :, 3:-2]) -
			 										   5.0 * (su[:, :, 1:-4] - su[:, :, 4:-1]) +
													   1.0 * (su[:, :, :-5] - su[:, :, 5:]))
	tnd_su = - (flux_su[:, :, :-1] - flux_su[:, :, 1:]) / dz
	assert 'x_momentum_isentropic' in tendencies
	assert np.allclose(tnd_su, tendencies['x_momentum_isentropic'][:, :, 3:-3])
	tendencies.pop('x_momentum_isentropic')

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (sv[:, :, 2:-3] + sv[:, :, 3:-2]) -
									 8.0 * (sv[:, :, 1:-4] + sv[:, :, 4:-1]) +
									 1.0 * (sv[:, :, :-5] + sv[:, :, 5:]))
	flux_sv = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (sv[:, :, 2:-3] - sv[:, :, 3:-2]) -
													   5.0 * (sv[:, :, 1:-4] - sv[:, :, 4:-1]) +
													   1.0 * (sv[:, :, :-5] - sv[:, :, 5:]))
	tnd_sv = - (flux_sv[:, :, :-1] - flux_sv[:, :, 1:]) / dz
	assert 'y_momentum_isentropic' in tendencies
	assert np.allclose(tnd_sv, tendencies['y_momentum_isentropic'][:, :, 3:-3])
	tendencies.pop('y_momentum_isentropic')

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (sqv[:, :, 2:-3] + sqv[:, :, 3:-2]) -
									 8.0 * (sqv[:, :, 1:-4] + sqv[:, :, 4:-1]) +
									 1.0 * (sqv[:, :, :-5] + sqv[:, :, 5:]))
	flux_sqv = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (sqv[:, :, 2:-3] - sqv[:, :, 3:-2]) -
													   5.0 * (sqv[:, :, 1:-4] - sqv[:, :, 4:-1]) +
													   1.0 * (sqv[:, :, :-5] - sqv[:, :, 5:]))
	tnd_qv = - (flux_sqv[:, :, :-1] - flux_sqv[:, :, 1:]) / (dz * s[:, :, 3:-3])
	assert mfwv in tendencies
	assert np.allclose(tnd_qv, tendencies[mfwv][:, :, 3:-3])
	tendencies.pop(mfwv)

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (sqc[:, :, 2:-3] + sqc[:, :, 3:-2]) -
									 8.0 * (sqc[:, :, 1:-4] + sqc[:, :, 4:-1]) +
									 1.0 * (sqc[:, :, :-5] + sqc[:, :, 5:]))
	flux_sqc = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (sqc[:, :, 2:-3] - sqc[:, :, 3:-2]) -
													   5.0 * (sqc[:, :, 1:-4] - sqc[:, :, 4:-1]) +
													   1.0 * (sqc[:, :, :-5] - sqc[:, :, 5:]))
	tnd_qc = - (flux_sqc[:, :, :-1] - flux_sqc[:, :, 1:]) / (dz * s[:, :, 3:-3])
	assert mfcw in tendencies
	assert np.allclose(tnd_qc, tendencies[mfcw][:, :, 3:-3])
	tendencies.pop(mfcw)

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (sqr[:, :, 2:-3] + sqr[:, :, 3:-2]) -
									 8.0 * (sqr[:, :, 1:-4] + sqr[:, :, 4:-1]) +
									 1.0 * (sqr[:, :, :-5] + sqr[:, :, 5:]))
	flux_sqr = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (sqr[:, :, 2:-3] - sqr[:, :, 3:-2]) -
													  5.0 * (sqr[:, :, 1:-4] - sqr[:, :, 4:-1]) +
													  1.0 * (sqr[:, :, :-5] - sqr[:, :, 5:]))
	tnd_qr = - (flux_sqr[:, :, :-1] - flux_sqr[:, :, 1:]) / (dz * s[:, :, 3:-3])
	assert mfpw in tendencies
	assert np.allclose(tnd_qr, tendencies[mfpw][:, :, 3:-3])
	tendencies.pop(mfpw)

	assert tendencies == {}

	assert diagnostics == {}


if __name__ == '__main__':
	pytest.main([__file__])
