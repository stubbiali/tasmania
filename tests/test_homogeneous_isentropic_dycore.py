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
from copy import deepcopy
from datetime import timedelta
import numpy as np
import pytest
from sympl import DiagnosticComponent

import gridtools as gt
from tasmania.python.core.physics_composite import \
	ConcurrentCoupling, TasmaniaDiagnosticComponentComposite
from tasmania.python.dynamics.homogeneous_isentropic_dycore \
	import HomogeneousIsentropicDynamicalCore
from tasmania.python.physics.coriolis import ConservativeIsentropicCoriolis
from tasmania.python.physics.isentropic_diagnostics import IsentropicDiagnostics
from tasmania.python.physics.isentropic_tendencies import \
	ConservativeIsentropicPressureGradient
from tasmania.python.physics.microphysics import \
	Kessler, SaturationAdjustmentKessler, RaindropFallVelocity, Sedimentation


# Convenient shortcuts
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'
dtype = np.float64


def test1_dry(isentropic_dry_data):
	"""
	- No slow tendencies
	- No intermediate tendencies
	- No intermediate diagnostics
	- No sub-stepping
	"""
	grid, states = isentropic_dry_data
	state = states[-1]
	state_dc = deepcopy(state)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s',
		intermediate_tendencies=None, intermediate_diagnostics=None,
		fast_tendencies=None, fast_diagnostics=None,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	assert 'air_isentropic_density' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	assert len(dycore.input_properties) == 5

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 3

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	assert len(dycore.output_properties) == 5

	dt = timedelta(seconds=10)
	state_new = dycore(state, {}, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert 'time' in state_new
	assert state_new['time'] == state['time'] + dt
	assert 'air_isentropic_density' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	assert len(state_new) == 6


def test1_moist(isentropic_moist_data):
	"""
	- No slow tendencies
	- No intermediate tendencies
	- No intermediate diagnostics
	- No sub-stepping
	"""
	grid, states = isentropic_moist_data
	state = states[-1]
	state_dc = deepcopy(state)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=None, intermediate_diagnostics=None,
		fast_tendencies=None, fast_diagnostics=None,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	assert 'air_isentropic_density' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	assert mfwv in dycore.input_properties
	assert mfcw in dycore.input_properties
	assert mfpw in dycore.input_properties
	assert len(dycore.input_properties) == 8

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	assert mfwv in dycore.tendency_properties
	assert mfcw in dycore.tendency_properties
	assert mfpw in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 6

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	assert mfwv in dycore.output_properties
	assert mfcw in dycore.output_properties
	assert mfpw in dycore.output_properties
	assert len(dycore.output_properties) == 8

	dt = timedelta(seconds=10)
	state_new = dycore(state, {}, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert 'time' in state_new
	assert state_new['time'] == state['time'] + dt
	assert 'air_isentropic_density' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	assert mfwv in state_new
	assert mfcw in state_new
	assert mfpw in state_new
	assert len(state_new) == 9


def test2(isentropic_moist_data):
	"""
	- Yes slow tendencies
	- No intermediate tendencies
	- No intermediate diagnostics
	- No sub-stepping
	"""
	grid, states = isentropic_moist_data
	state = states[-1]
	state_dc = deepcopy(state)

	coriolis = ConservativeIsentropicCoriolis(grid)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=None, intermediate_diagnostics=None,
		fast_tendencies=None, fast_diagnostics=None,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	assert 'air_isentropic_density' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	assert mfwv in dycore.input_properties
	assert mfcw in dycore.input_properties
	assert mfpw in dycore.input_properties
	assert len(dycore.input_properties) == 8

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	assert mfwv in dycore.tendency_properties
	assert mfcw in dycore.tendency_properties
	assert mfpw in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 6

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	assert mfwv in dycore.output_properties
	assert mfcw in dycore.output_properties
	assert mfpw in dycore.output_properties
	assert len(dycore.output_properties) == 8

	dt = timedelta(seconds=10)
	slow_tends, _ = coriolis(state)
	state_new = dycore(state, slow_tends, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert 'time' in state_new
	assert state_new['time'] == state['time'] + dt
	assert 'air_isentropic_density' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	assert mfwv in state_new
	assert mfcw in state_new
	assert mfpw in state_new
	assert len(state_new) == 9


def test3(isentropic_moist_data):
	"""
	- Yes slow tendencies
	- Yes intermediate tendencies
	- Yes intermediate diagnostics
	- No sub-stepping
	"""
	grid, states = isentropic_moist_data
	state = states[-1]
	state_dc = deepcopy(state)

	coriolis = ConservativeIsentropicCoriolis(grid)

	pg = ConservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=gt.mode.NUMPY, dtype=dtype
	)
	inter_tends = ConcurrentCoupling(pg)

	dv = IsentropicDiagnostics(
		grid, True, state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=gt.mode.NUMPY, dtype=dtype,
	)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=inter_tends, intermediate_diagnostics=dv,
		fast_tendencies=None, fast_diagnostics=None,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	assert 'air_isentropic_density' in dycore.input_properties
	assert 'montgomery_potential' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	assert mfwv in dycore.input_properties
	assert mfcw in dycore.input_properties
	assert mfpw in dycore.input_properties
	assert len(dycore.input_properties) == 9

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	assert mfwv in dycore.tendency_properties
	assert mfcw in dycore.tendency_properties
	assert mfpw in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 6

	assert 'air_density' in dycore.output_properties
	assert 'air_isentropic_density' in dycore.output_properties
	assert 'air_pressure_on_interface_levels' in dycore.output_properties
	assert 'air_temperature' in dycore.output_properties
	assert 'exner_function_on_interface_levels' in dycore.output_properties
	assert 'height_on_interface_levels' in dycore.output_properties
	assert 'montgomery_potential' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	assert mfwv in dycore.output_properties
	assert mfcw in dycore.output_properties
	assert mfpw in dycore.output_properties
	assert len(dycore.output_properties) == 14

	dt = timedelta(seconds=10)
	slow_tends, _ = coriolis(state)
	state_new = dycore(state, slow_tends, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert 'time' in state_new
	assert state_new['time'] == state['time'] + dt
	assert 'air_density' in state_new
	assert 'air_isentropic_density' in state_new
	assert 'air_pressure_on_interface_levels' in state_new
	assert 'air_temperature' in state_new
	assert 'exner_function_on_interface_levels' in state_new
	assert 'height_on_interface_levels' in state_new
	assert 'montgomery_potential' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	assert mfwv in state_new
	assert mfcw in state_new
	assert mfpw in state_new
	assert len(state_new) == 15


class IdentityDiagnostic(DiagnosticComponent):
	def __init__(self, grid, moist):
		self._grid = grid
		self._moist = moist
		super().__init__()

	@property
	def input_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		if self._moist:
			return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		if self._moist:
			return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	def array_call(self, state):
		return {
			'air_isentropic_density': state['air_isentropic_density'],
			'x_momentum_isentropic': state['x_momentum_isentropic'],
			'y_momentum_isentropic': state['y_momentum_isentropic'],
			mfwv: state[mfwv],
			mfcw: state[mfcw],
			mfpw: state[mfpw],
		}


def test4(isentropic_moist_data):
	"""
	- Moist
	- Yes slow tendencies
	- Yes intermediate tendencies
	- Yes intermediate diagnostics
	- Yes sub-stepping
	- No fast tendencies
	- Yes fast diagnostics
	"""
	grid, states = isentropic_moist_data
	state = states[-1]
	state_dc = deepcopy(state)

	coriolis = ConservativeIsentropicCoriolis(grid)

	slow_tends, _ = coriolis(state)

	pg = ConservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=gt.mode.NUMPY, dtype=dtype
	)
	inter_tends = ConcurrentCoupling(pg)

	dv = IsentropicDiagnostics(
		grid, True, state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=gt.mode.NUMPY, dtype=dtype,
	)

	dt = timedelta(seconds=10)

	#
	# Substepping OFF
	#
	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=inter_tends, intermediate_diagnostics=dv,
		substeps=0, fast_tendencies=None, fast_diagnostics=None,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	state_ref = dycore(state, slow_tends, dt)

	#
	# Substepping ON, no fast diagnostics
	#
	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=inter_tends, intermediate_diagnostics=dv,
		substeps=6, fast_tendencies=None, fast_diagnostics=None,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	state1 = dycore(state, slow_tends, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	for key in state1:
		if key == 'time':
			assert state1['time'] == state_dc['time'] + dt
		else:
			assert key in state_ref
			assert np.allclose(state1[key], state_ref[key])

	assert len(state1) == 15

	#
	# Substepping ON, with fast diagnostics
	#
	fd = IdentityDiagnostic(grid, True)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=inter_tends, intermediate_diagnostics=dv,
		substeps=6, fast_tendencies=None, fast_diagnostics=fd,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	state2 = dycore(state, slow_tends, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	for key in state2:
		if key == 'time':
			assert state2['time'] == state_dc['time'] + dt
		else:
			assert key in state_ref
			assert np.allclose(state2[key], state_ref[key])

	assert len(state2) == 15


#def isentropic_moist_sedimentation_data():
#	from tasmania.python.utils.storage_utils import load_netcdf_dataset
#	return load_netcdf_dataset('baseline_datasets/isentropic_moist_sedimentation.nc')


def test5(isentropic_moist_sedimentation_data):
	"""
	- Moist
	- Yes slow tendencies
	- Yes intermediate tendencies
	- Yes intermediate diagnostics
	- Yes sub-stepping
	- Yes fast tendencies
	- No fast diagnostics
	"""
	grid, states = isentropic_moist_sedimentation_data
	state = states[-1]
	if len(state['accumulated_precipitation'].shape) == 3:
		state['accumulated_precipitation'] = \
			state['accumulated_precipitation'][:, :, 0]
	state_dc = deepcopy(state)

	coriolis = ConservativeIsentropicCoriolis(grid)
	slow_tends, _ = coriolis(state)

	pg = ConservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=gt.mode.NUMPY, dtype=dtype
	)
	ke = Kessler(
		grid, air_pressure_on_interface_levels=True,
		rain_evaporation=False, backend=gt.mode.NUMPY
	)
	inter_tends = ConcurrentCoupling(pg, ke, execution_policy='serial')

	dv = IsentropicDiagnostics(
		grid, True, state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=gt.mode.NUMPY, dtype=dtype,
	)
	sa = SaturationAdjustmentKessler(
		grid, air_pressure_on_interface_levels=True, backend=gt.mode.NUMPY
	)
	inter_diags = TasmaniaDiagnosticComponentComposite(
		dv, sa, execution_policy='serial'
	)

	rfv = RaindropFallVelocity(grid, backend=gt.mode.NUMPY)
	sd = Sedimentation(
		grid, sedimentation_flux_scheme='second_order_upwind',
		backend=gt.mode.NUMPY
	)
	fast_tends = ConcurrentCoupling(
		rfv, sd, execution_policy='serial'
	)

	dt = timedelta(seconds=100)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=inter_tends, intermediate_diagnostics=inter_diags,
		substeps=6, fast_tendencies=fast_tends, fast_diagnostics=None,
		time_integration_scheme='rk3',
		horizontal_flux_scheme='third_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	assert 'air_density' in dycore.input_properties
	assert 'air_isentropic_density' in dycore.input_properties
	assert 'air_pressure_on_interface_levels' in dycore.input_properties
	assert 'air_temperature' in dycore.input_properties
	assert 'exner_function_on_interface_levels' in dycore.input_properties
	assert 'height_on_interface_levels' in dycore.input_properties
	assert 'montgomery_potential' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	assert mfwv in dycore.input_properties
	assert mfcw in dycore.input_properties
	assert mfpw in dycore.input_properties
	assert 'accumulated_precipitation' in dycore.input_properties
	assert len(dycore.input_properties) == 15

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	assert mfwv in dycore.tendency_properties
	assert mfcw in dycore.tendency_properties
	assert mfpw in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 6

	assert 'air_density' in dycore.output_properties
	assert 'air_isentropic_density' in dycore.output_properties
	assert 'air_pressure_on_interface_levels' in dycore.output_properties
	assert 'air_temperature' in dycore.output_properties
	assert 'exner_function_on_interface_levels' in dycore.output_properties
	assert 'height_on_interface_levels' in dycore.output_properties
	assert 'montgomery_potential' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	assert mfwv in dycore.output_properties
	assert mfcw in dycore.output_properties
	assert mfpw in dycore.output_properties
	assert 'accumulated_precipitation' in dycore.output_properties
	assert len(dycore.output_properties) == 15

	state_new = dycore(state, slow_tends, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert len(state) == len(state_dc)

	assert state_new['time'] == state['time'] + dt

	for key in dycore.output_properties:
		assert key in state_new

	assert len(state_new) == len(dycore.output_properties) + 1


def test6(isentropic_moist_sedimentation_data):
	"""
	- Moist
	- Yes slow tendencies
	- Yes intermediate tendencies
	- Yes intermediate diagnostics
	- Yes sub-stepping
	- Yes fast tendencies
	- Yes fast diagnostics
	"""
	grid, states = isentropic_moist_sedimentation_data
	state = states[-1]
	if len(state['accumulated_precipitation'].shape) == 3:
		state['accumulated_precipitation'] = \
			state['accumulated_precipitation'][:, :, 0]
	state_dc = deepcopy(state)

	coriolis = ConservativeIsentropicCoriolis(grid)
	slow_tends, _ = coriolis(state)

	pg = ConservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=gt.mode.NUMPY, dtype=dtype
	)
	ke = Kessler(
		grid, air_pressure_on_interface_levels=True,
		rain_evaporation=False, backend=gt.mode.NUMPY
	)
	inter_tends = ConcurrentCoupling(pg, ke, execution_policy='serial')

	dv = IsentropicDiagnostics(
		grid, True, state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=gt.mode.NUMPY, dtype=dtype,
	)
	sa = SaturationAdjustmentKessler(
		grid, air_pressure_on_interface_levels=True, backend=gt.mode.NUMPY
	)
	inter_diags = TasmaniaDiagnosticComponentComposite(
		dv, sa, execution_policy='serial'
	)

	rfv = RaindropFallVelocity(grid, backend=gt.mode.NUMPY)
	sd = Sedimentation(
		grid, sedimentation_flux_scheme='second_order_upwind',
		backend=gt.mode.NUMPY
	)
	fast_tends = ConcurrentCoupling(
		rfv, sd, execution_policy='serial'
	)

	fast_diags = TasmaniaDiagnosticComponentComposite(
		dv, execution_policy='serial'
	)

	dt = timedelta(seconds=100)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, time_units='s', moist=True,
		intermediate_tendencies=inter_tends, intermediate_diagnostics=inter_diags,
		substeps=6, fast_tendencies=fast_tends, fast_diagnostics=fast_diags,
		time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind',
		horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='second_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='second_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	assert 'air_density' in dycore.input_properties
	assert 'air_isentropic_density' in dycore.input_properties
	assert 'air_pressure_on_interface_levels' in dycore.input_properties
	assert 'air_temperature' in dycore.input_properties
	assert 'exner_function_on_interface_levels' in dycore.input_properties
	assert 'height_on_interface_levels' in dycore.input_properties
	assert 'montgomery_potential' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	assert mfwv in dycore.input_properties
	assert mfcw in dycore.input_properties
	assert mfpw in dycore.input_properties
	assert 'accumulated_precipitation' in dycore.input_properties
	assert len(dycore.input_properties) == 15

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	assert mfwv in dycore.tendency_properties
	assert mfcw in dycore.tendency_properties
	assert mfpw in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 6

	assert 'air_density' in dycore.output_properties
	assert 'air_isentropic_density' in dycore.output_properties
	assert 'air_pressure_on_interface_levels' in dycore.output_properties
	assert 'air_temperature' in dycore.output_properties
	assert 'exner_function_on_interface_levels' in dycore.output_properties
	assert 'height_on_interface_levels' in dycore.output_properties
	assert 'montgomery_potential' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	assert mfwv in dycore.output_properties
	assert mfcw in dycore.output_properties
	assert mfpw in dycore.output_properties
	assert 'accumulated_precipitation' in dycore.output_properties
	assert len(dycore.output_properties) == 15

	state_new = dycore(state, slow_tends, dt)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert len(state) == len(state_dc)

	assert state_new['time'] == state['time'] + dt

	for key in dycore.output_properties:
		assert key in state_new

	assert len(state_new) == len(dycore.output_properties) + 1


if __name__ == '__main__':
	pytest.main([__file__])
	#test5_forward_euler(isentropic_moist_sedimentation_data())
