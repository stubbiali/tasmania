from datetime import timedelta
import numpy as np
import pytest
from sympl import units_are_same

import gridtools as gt
from tasmania.core.time_steppers import ForwardEuler, RungeKutta2, \
									    RungeKutta3COSMO, RungeKutta3
from tasmania.physics.isentropic_tendencies import ConservativeIsentropicPressureGradient
from tasmania.utils.data_utils import make_state


def test_forward_euler(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	dt = timedelta(seconds=20)
	su = state['x_momentum_isentropic'].values
	sv = state['y_momentum_isentropic'].values

	pg = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed',
												backend=backend, dtype=dtype)

	fe = ForwardEuler(pg, grid=grid)

	assert 'x_momentum_isentropic' in fe.output_properties
	assert units_are_same(fe.output_properties['x_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')
	assert 'y_momentum_isentropic' in fe.output_properties
	assert units_are_same(fe.output_properties['y_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')

	_, out_state = fe(state, dt)

	tendencies, _ = pg(state)
	su_new = su + dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	sv_new = sv + dt.total_seconds() * tendencies['y_momentum_isentropic'].values

	assert np.allclose(su_new, out_state['x_momentum_isentropic'])
	assert np.allclose(sv_new, out_state['y_momentum_isentropic'])


def test_rk2(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	dt = timedelta(seconds=20)
	su = state['x_momentum_isentropic'].values
	sv = state['y_momentum_isentropic'].values
	units = {'x_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
			 'y_momentum_isentropic': 'kg m^-1 K^-1 s^-1'}

	pg = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed',
												backend=backend, dtype=dtype)

	rk2 = RungeKutta2(pg, grid=grid)

	assert 'x_momentum_isentropic' in rk2.output_properties
	assert units_are_same(rk2.output_properties['x_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')
	assert 'y_momentum_isentropic' in rk2.output_properties
	assert units_are_same(rk2.output_properties['y_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')

	_, out_state = rk2(state, dt)

	tendencies, _ = pg(state)
	su1 = su + 0.5 * dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	sv1 = sv + 0.5 * dt.total_seconds() * tendencies['y_momentum_isentropic'].values
	raw_state_1 = {'time': state['time'],
				   'x_momentum_isentropic': su1,
				   'y_momentum_isentropic': sv1}
	state_1 = make_state(raw_state_1, grid, units)
	state_1['air_isentropic_density'] = state['air_isentropic_density']
	state_1['montgomery_potential'] = state['montgomery_potential']

	tendencies, _ = pg(state_1)
	su2 = su + dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	sv2 = sv + dt.total_seconds() * tendencies['y_momentum_isentropic'].values

	assert np.allclose(su2, out_state['x_momentum_isentropic'])
	assert np.allclose(sv2, out_state['y_momentum_isentropic'])


def test_rk3cosmo(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	dt = timedelta(seconds=20)
	su = state['x_momentum_isentropic'].values
	sv = state['y_momentum_isentropic'].values
	units = {'x_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
			 'y_momentum_isentropic': 'kg m^-1 K^-1 s^-1'}

	pg = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed',
												backend=backend, dtype=dtype)

	rk3 = RungeKutta3COSMO(pg, grid=grid)

	assert 'x_momentum_isentropic' in rk3.output_properties
	assert units_are_same(rk3.output_properties['x_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')
	assert 'y_momentum_isentropic' in rk3.output_properties
	assert units_are_same(rk3.output_properties['y_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')

	_, out_state = rk3(state, dt)

	tendencies, _ = pg(state)
	su1 = su + 1./3. * dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	sv1 = sv + 1./3. * dt.total_seconds() * tendencies['y_momentum_isentropic'].values
	raw_state_1 = {'time': state['time'],
				   'x_momentum_isentropic': su1,
				   'y_momentum_isentropic': sv1}
	state_1 = make_state(raw_state_1, grid, units)
	state_1['air_isentropic_density'] = state['air_isentropic_density']
	state_1['montgomery_potential'] = state['montgomery_potential']

	tendencies, _ = pg(state_1)
	su2 = su + 1./2. * dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	sv2 = sv + 1./2. * dt.total_seconds() * tendencies['y_momentum_isentropic'].values
	raw_state_2 = {'time': state['time'],
				   'x_momentum_isentropic': su2,
				   'y_momentum_isentropic': sv2}
	state_2 = make_state(raw_state_2, grid, units)
	state_2['air_isentropic_density'] = state['air_isentropic_density']
	state_2['montgomery_potential'] = state['montgomery_potential']

	tendencies, _ = pg(state_2)
	su3 = su + dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	sv3 = sv + dt.total_seconds() * tendencies['y_momentum_isentropic'].values

	assert np.allclose(su3, out_state['x_momentum_isentropic'])
	assert np.allclose(sv3, out_state['y_momentum_isentropic'])


def test_rk3(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	dt = timedelta(seconds=20)
	su = state['x_momentum_isentropic'].values
	sv = state['y_momentum_isentropic'].values
	units = {'x_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
			 'y_momentum_isentropic': 'kg m^-1 K^-1 s^-1'}

	pg = ConservativeIsentropicPressureGradient(grid, 2, 'relaxed',
												backend=backend, dtype=dtype)

	rk3 = RungeKutta3(pg, grid=grid)
	a1, a2 = rk3._alpha1, rk3._alpha2
	b21 = rk3._beta21
	g0, g1, g2 = rk3._gamma0, rk3._gamma1, rk3._gamma2

	assert 'x_momentum_isentropic' in rk3.output_properties
	assert units_are_same(rk3.output_properties['x_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')
	assert 'y_momentum_isentropic' in rk3.output_properties
	assert units_are_same(rk3.output_properties['y_momentum_isentropic']['units'],
						  'kg m^-1 K^-1 s^-1')

	_, out_state = rk3(state, dt)

	tendencies, _ = pg(state)
	k0_su = dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	su1 = su + a1 * k0_su
	k0_sv = dt.total_seconds() * tendencies['y_momentum_isentropic'].values
	sv1 = sv + a1 * k0_sv
	raw_state_1 = {'time': state['time'],
				   'x_momentum_isentropic': su1,
				   'y_momentum_isentropic': sv1}
	state_1 = make_state(raw_state_1, grid, units)
	state_1['air_isentropic_density'] = state['air_isentropic_density']
	state_1['montgomery_potential'] = state['montgomery_potential']

	tendencies, _ = pg(state_1)
	k1_su = dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	su2 = su + b21 * k0_su + (a2 - b21) * k1_su
	k1_sv = dt.total_seconds() * tendencies['y_momentum_isentropic'].values
	sv2 = sv + b21 * k0_sv + (a2 - b21) * k1_sv
	raw_state_2 = {'time': state['time'],
				   'x_momentum_isentropic': su2,
				   'y_momentum_isentropic': sv2}
	state_2 = make_state(raw_state_2, grid, units)
	state_2['air_isentropic_density'] = state['air_isentropic_density']
	state_2['montgomery_potential'] = state['montgomery_potential']

	tendencies, _ = pg(state_2)
	k2_su = dt.total_seconds() * tendencies['x_momentum_isentropic'].values
	su3 = su + g0 * k0_su + g1 * k1_su + g2 * k2_su
	k2_sv = dt.total_seconds() * tendencies['y_momentum_isentropic'].values
	sv3 = sv + g0 * k0_sv + g1 * k1_sv + g2 * k2_sv

	assert np.allclose(su3, out_state['x_momentum_isentropic'])
	assert np.allclose(sv3, out_state['y_momentum_isentropic'])


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data
	#test_rk2(isentropic_dry_data())
