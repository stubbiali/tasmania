import numpy as np
import pytest

import gridtools as gt
from tasmania.physics.isentropic_tendencies import VerticalIsentropicAdvection
from tasmania.utils.data_utils import make_data_array_3d


mf_wv  = 'mass_fraction_of_water_vapor_in_air'
mf_clw = 'mass_fraction_of_cloud_liquid_water_in_air'
mf_pw  = 'mass_fraction_of_precipitation_water_in_air'


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
	assert mf_wv in fluxer.input_properties
	assert mf_clw in fluxer.input_properties
	assert mf_pw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mf_wv in fluxer.tendency_properties
	assert mf_clw in fluxer.tendency_properties
	assert mf_pw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz)
	wm = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
	state['tendency_of_air_potential_temperature'] = make_data_array_3d(w, grid, 'K s^-1')

	s  = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv = state[mf_wv].to_units('g g^-1').values[...]
	qc = state[mf_clw].to_units('g g^-1').values[...]
	qr = state[mf_pw].to_units('g g^-1').values[...]

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

	flux_qv = (wm > 0) * wm * qv[:, :, 1:] - (wm < 0) * wm * qv[:, :, :-1]
	tnd_qv = - (flux_qv[:, :, :-1] - flux_qv[:, :, 1:]) / dz
	assert mf_wv in tendencies
	assert np.allclose(tnd_qv, tendencies[mf_wv][:, :, 1:-1])
	tendencies.pop(mf_wv)

	flux_qc = (wm > 0) * wm * qc[:, :, 1:] - (wm < 0) * wm * qc[:, :, :-1]
	tnd_qc = - (flux_qc[:, :, :-1] - flux_qc[:, :, 1:]) / dz
	assert mf_clw in tendencies
	assert np.allclose(tnd_qc, tendencies[mf_clw][:, :, 1:-1])
	tendencies.pop(mf_clw)

	flux_qr = (wm > 0) * wm * qr[:, :, 1:] - (wm < 0) * wm * qr[:, :, :-1]
	tnd_qr = - (flux_qr[:, :, :-1] - flux_qr[:, :, 1:]) / dz
	assert mf_pw in tendencies
	assert np.allclose(tnd_qr, tendencies[mf_pw][:, :, 1:-1])
	tendencies.pop(mf_pw)

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
	assert mf_wv in fluxer.input_properties
	assert mf_clw in fluxer.input_properties
	assert mf_pw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mf_wv in fluxer.tendency_properties
	assert mf_clw in fluxer.tendency_properties
	assert mf_pw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_data_array_3d(w, grid, 'K s^-1')

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

	flux_qv = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * qv[:, :, 1:] - \
			  (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * qv[:, :, :-1]
	tnd_qv = - (flux_qv[:, :, :-1] - flux_qv[:, :, 1:]) / dz
	assert mf_wv in tendencies
	assert np.allclose(tnd_qv, tendencies[mf_wv][:, :, 1:-1])
	tendencies.pop(mf_wv)

	flux_qc = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * qc[:, :, 1:] - \
			  (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * qc[:, :, :-1]
	tnd_qc = - (flux_qc[:, :, :-1] - flux_qc[:, :, 1:]) / dz
	assert mf_clw in tendencies
	assert np.allclose(tnd_qc, tendencies[mf_clw][:, :, 1:-1])
	tendencies.pop(mf_clw)

	flux_qr = (w[:, :, 1:-1] > 0) * w[:, :, 1:-1] * qr[:, :, 1:] - \
			  (w[:, :, 1:-1] < 0) * w[:, :, 1:-1] * qr[:, :, :-1]
	tnd_qr = - (flux_qr[:, :, :-1] - flux_qr[:, :, 1:]) / dz
	assert mf_pw in tendencies
	assert np.allclose(tnd_qr, tendencies[mf_pw][:, :, 1:-1])
	tendencies.pop(mf_pw)

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
	assert mf_wv in fluxer.input_properties
	assert mf_clw in fluxer.input_properties
	assert mf_pw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mf_wv in fluxer.tendency_properties
	assert mf_clw in fluxer.tendency_properties
	assert mf_pw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz)
	wm = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
	state['tendency_of_air_potential_temperature'] = make_data_array_3d(w, grid, 'K s^-1')

	s  = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv = state[mf_wv].to_units('g g^-1').values[...]
	qc = state[mf_clw].to_units('g g^-1').values[...]
	qr = state[mf_pw].to_units('g g^-1').values[...]

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

	flux_qv = wm * 0.5 * (qv[:, :, 1:] + qv[:, :, :-1])
	tnd_qv = - (flux_qv[:, :, :-1] - flux_qv[:, :, 1:]) / dz
	assert mf_wv in tendencies
	assert np.allclose(tnd_qv, tendencies[mf_wv][:, :, 1:-1])
	tendencies.pop(mf_wv)

	flux_qc = wm * 0.5 * (qc[:, :, 1:] + qc[:, :, :-1])
	tnd_qc = - (flux_qc[:, :, :-1] - flux_qc[:, :, 1:]) / dz
	assert mf_clw in tendencies
	assert np.allclose(tnd_qc, tendencies[mf_clw][:, :, 1:-1])
	tendencies.pop(mf_clw)

	flux_qr = wm * 0.5 * (qr[:, :, 1:] + qr[:, :, :-1])
	tnd_qr = - (flux_qr[:, :, :-1] - flux_qr[:, :, 1:]) / dz
	assert mf_pw in tendencies
	assert np.allclose(tnd_qr, tendencies[mf_pw][:, :, 1:-1])
	tendencies.pop(mf_pw)

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
	assert mf_wv in fluxer.input_properties
	assert mf_clw in fluxer.input_properties
	assert mf_pw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mf_wv in fluxer.tendency_properties
	assert mf_clw in fluxer.tendency_properties
	assert mf_pw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_data_array_3d(w, grid, 'K s^-1')

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

	flux_qv = w[:, :, 1:-1] * 0.5 * (qv[:, :, 1:] + qv[:, :, :-1])
	tnd_qv = - (flux_qv[:, :, :-1] - flux_qv[:, :, 1:]) / dz
	assert mf_wv in tendencies
	assert np.allclose(tnd_qv, tendencies[mf_wv][:, :, 1:-1])
	tendencies.pop(mf_wv)

	flux_qc = w[:, :, 1:-1] * 0.5 * (qc[:, :, 1:] + qc[:, :, :-1])
	tnd_qc = - (flux_qc[:, :, :-1] - flux_qc[:, :, 1:]) / dz
	assert mf_clw in tendencies
	assert np.allclose(tnd_qc, tendencies[mf_clw][:, :, 1:-1])
	tendencies.pop(mf_clw)

	flux_qr = w[:, :, 1:-1] * 0.5 * (qr[:, :, 1:] + qr[:, :, :-1])
	tnd_qr = - (flux_qr[:, :, :-1] - flux_qr[:, :, 1:]) / dz
	assert mf_pw in tendencies
	assert np.allclose(tnd_qr, tendencies[mf_pw][:, :, 1:-1])
	tendencies.pop(mf_pw)

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
	assert mf_wv in fluxer.input_properties
	assert mf_clw in fluxer.input_properties
	assert mf_pw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mf_wv in fluxer.tendency_properties
	assert mf_clw in fluxer.tendency_properties
	assert mf_pw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_data_array_3d(w, grid, 'K s^-1')

	s  = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv = state[mf_wv].to_units('g g^-1').values[...]
	qc = state[mf_clw].to_units('g g^-1').values[...]
	qr = state[mf_pw].to_units('g g^-1').values[...]

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

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (qv[:, :, 1:-2] + qv[:, :, 2:-1]) -
									1.0 * (qv[:, :, :-3] + qv[:, :, 3:]))
	flux_qv = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (qv[:, :, 1:-2] - qv[:, :, 2:-1]) -
													  1.0 * (qv[:, :, :-3] - qv[:, :, 3:]))
	tnd_qv = - (flux_qv[:, :, :-1] - flux_qv[:, :, 1:]) / dz
	assert mf_wv in tendencies
	assert np.allclose(tnd_qv, tendencies[mf_wv][:, :, 2:-2])
	tendencies.pop(mf_wv)

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (qc[:, :, 1:-2] + qc[:, :, 2:-1]) -
									1.0 * (qc[:, :, :-3] + qc[:, :, 3:]))
	flux_qc = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (qc[:, :, 1:-2] - qc[:, :, 2:-1]) -
													  1.0 * (qc[:, :, :-3] - qc[:, :, 3:]))
	tnd_qc = - (flux_qc[:, :, :-1] - flux_qc[:, :, 1:]) / dz
	assert mf_clw in tendencies
	assert np.allclose(tnd_qc, tendencies[mf_clw][:, :, 2:-2])
	tendencies.pop(mf_clw)

	flux4 = w[:, :, 2:-2] / 12.0 * (7.0 * (qr[:, :, 1:-2] + qr[:, :, 2:-1]) -
									1.0 * (qr[:, :, :-3] + qr[:, :, 3:]))
	flux_qr = flux4 - np.abs(w[:, :, 2:-2]) / 12.0 * (3.0 * (qr[:, :, 1:-2] - qr[:, :, 2:-1]) -
													  1.0 * (qr[:, :, :-3] - qr[:, :, 3:]))
	tnd_qr = - (flux_qr[:, :, :-1] - flux_qr[:, :, 1:]) / dz
	assert mf_pw in tendencies
	assert np.allclose(tnd_qr, tendencies[mf_pw][:, :, 2:-2])
	tendencies.pop(mf_pw)

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
	assert mf_wv in fluxer.input_properties
	assert mf_clw in fluxer.input_properties
	assert mf_pw in fluxer.input_properties
	assert len(fluxer.input_properties) == 7

	assert 'air_isentropic_density' in fluxer.tendency_properties
	assert 'x_momentum_isentropic' in fluxer.tendency_properties
	assert 'y_momentum_isentropic' in fluxer.tendency_properties
	assert mf_wv in fluxer.tendency_properties
	assert mf_clw in fluxer.tendency_properties
	assert mf_pw in fluxer.tendency_properties
	assert len(fluxer.tendency_properties) == 6

	assert fluxer.diagnostic_properties == {}

	w  = np.random.rand(grid.nx, grid.ny, grid.nz+1)
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_data_array_3d(w, grid, 'K s^-1')

	s  = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	sv = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	qv = state[mf_wv].to_units('g g^-1').values[...]
	qc = state[mf_clw].to_units('g g^-1').values[...]
	qr = state[mf_pw].to_units('g g^-1').values[...]

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

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (qv[:, :, 2:-3] + qv[:, :, 3:-2]) -
									 8.0 * (qv[:, :, 1:-4] + qv[:, :, 4:-1]) +
									 1.0 * (qv[:, :, :-5] + qv[:, :, 5:]))
	flux_qv = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (qv[:, :, 2:-3] - qv[:, :, 3:-2]) -
													   5.0 * (qv[:, :, 1:-4] - qv[:, :, 4:-1]) +
													   1.0 * (qv[:, :, :-5] - qv[:, :, 5:]))
	tnd_qv = - (flux_qv[:, :, :-1] - flux_qv[:, :, 1:]) / dz
	assert mf_wv in tendencies
	assert np.allclose(tnd_qv, tendencies[mf_wv][:, :, 3:-3])
	tendencies.pop(mf_wv)

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (qc[:, :, 2:-3] + qc[:, :, 3:-2]) -
									 8.0 * (qc[:, :, 1:-4] + qc[:, :, 4:-1]) +
									 1.0 * (qc[:, :, :-5] + qc[:, :, 5:]))
	flux_qc = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (qc[:, :, 2:-3] - qc[:, :, 3:-2]) -
													   5.0 * (qc[:, :, 1:-4] - qc[:, :, 4:-1]) +
													   1.0 * (qc[:, :, :-5] - qc[:, :, 5:]))
	tnd_qc = - (flux_qc[:, :, :-1] - flux_qc[:, :, 1:]) / dz
	assert mf_clw in tendencies
	assert np.allclose(tnd_qc, tendencies[mf_clw][:, :, 3:-3])
	tendencies.pop(mf_clw)

	flux6 = w[:, :, 3:-3] / 60.0 * (37.0 * (qr[:, :, 2:-3] + qr[:, :, 3:-2]) -
									 8.0 * (qr[:, :, 1:-4] + qr[:, :, 4:-1]) +
									 1.0 * (qr[:, :, :-5] + qr[:, :, 5:]))
	flux_qr = flux6 - np.abs(w[:, :, 3:-3]) / 60.0 * (10.0 * (qr[:, :, 2:-3] - qr[:, :, 3:-2]) -
													  5.0 * (qr[:, :, 1:-4] - qr[:, :, 4:-1]) +
													  1.0 * (qr[:, :, :-5] - qr[:, :, 5:]))
	tnd_qr = - (flux_qr[:, :, :-1] - flux_qr[:, :, 1:]) / dz
	assert mf_pw in tendencies
	assert np.allclose(tnd_qr, tendencies[mf_pw][:, :, 3:-3])
	tendencies.pop(mf_pw)

	assert tendencies == {}

	assert diagnostics == {}


if __name__ == '__main__':
	pytest.main([__file__])
