from datetime import timedelta
import numpy as np
import pytest

import gridtools as gt
from tasmania.dynamics.homogeneous_isentropic_dycore \
	import HomogeneousIsentropicDynamicalCore


def test_dry(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	dycore = HomogeneousIsentropicDynamicalCore(
				 grid, False, 'centered', 'centered', 'relaxed',
				 intermediate_parameterizations=None,
				 damp_on=True, damp_type='rayleigh', damp_depth=15,
				 damp_max=0.0002, damp_at_every_stage=True,
				 smooth_on=True, smooth_type='first_order', smooth_damp_depth=10,
				 smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
				 smooth_moist_on=False, smooth_moist_type='first_order',
				 smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
				 smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
				 backend=gt.mode.NUMPY, dtype=np.float64)

	dt = timedelta(seconds=10)

	state_new = dycore(state, {}, dt)

	assert 'time' in state_new.keys()
	assert state_new['time'] == state['time'] + dt
	state_new.pop('time')

	assert 'air_isentropic_density' in state_new.keys()
	state_new.pop('air_isentropic_density')

	assert 'x_momentum_isentropic' in state_new.keys()
	state_new.pop('x_momentum_isentropic')

	assert 'x_velocity_at_u_locations' in state_new.keys()
	state_new.pop('x_velocity_at_u_locations')

	assert 'y_momentum_isentropic' in state_new.keys()
	state_new.pop('y_momentum_isentropic')

	assert 'y_velocity_at_v_locations' in state_new.keys()
	state_new.pop('y_velocity_at_v_locations')

	assert state_new == {}


def test_moist(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, True, 'rk3', 'third_order_upwind', 'relaxed',
		intermediate_parameterizations=None,
		damp_on=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth_on=True, smooth_type='first_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist_on=False, smooth_moist_type='first_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=np.float64)

	dt = timedelta(seconds=10)

	state_new = dycore(state, {}, dt)

	assert 'time' in state_new.keys()
	assert state_new['time'] == state['time'] + dt
	state_new.pop('time')

	assert 'air_isentropic_density' in state_new.keys()
	state_new.pop('air_isentropic_density')

	assert 'x_momentum_isentropic' in state_new.keys()
	state_new.pop('x_momentum_isentropic')

	assert 'x_velocity_at_u_locations' in state_new.keys()
	state_new.pop('x_velocity_at_u_locations')

	assert 'y_momentum_isentropic' in state_new.keys()
	state_new.pop('y_momentum_isentropic')

	assert 'y_velocity_at_v_locations' in state_new.keys()
	state_new.pop('y_velocity_at_v_locations')

	assert 'mass_fraction_of_water_vapor_in_air' in state_new.keys()
	state_new.pop('mass_fraction_of_water_vapor_in_air')

	assert 'mass_fraction_of_cloud_liquid_water_in_air' in state_new.keys()
	state_new.pop('mass_fraction_of_cloud_liquid_water_in_air')

	assert 'mass_fraction_of_precipitation_water_in_air' in state_new.keys()
	state_new.pop('mass_fraction_of_precipitation_water_in_air')

	assert state_new == {}


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_moist_data
	#test_moist(isentropic_moist_data())
