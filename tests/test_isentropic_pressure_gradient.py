from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import pytest
from sympl import DataArray

import gridtools as gt
from tasmania.physics.isentropic_tendencies import NonconservativeIsentropicPressureGradient, \
												   ConservativeIsentropicPressureGradient, \
												   VerticalIsentropicAdvection, \
												   PrescribedSurfaceHeating
from tasmania.utils.data_utils import make_data_array_3d
from tasmania.utils.utils import equal_to


def test_nonconservative_relaxed_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = NonconservativeIsentropicPressureGradient(
		grid, 2, 'relaxed', backend=backend, dtype=dtype)
	pg4 = NonconservativeIsentropicPressureGradient(
		grid, 4, 'relaxed', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	mtg = state['montgomery_potential'].values

	u_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	u_tnd2[1:-1, 1:-1, :] = - (mtg[2:, 1:-1, :] - mtg[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_velocity' in tendencies2.keys()
	assert np.allclose(u_tnd2, tendencies2['x_velocity'])

	v_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	v_tnd2[1:-1, 1:-1, :] = - (mtg[1:-1, 2:, :] - mtg[1:-1, :-2, :]) / (2. * dy)
	assert 'y_velocity' in tendencies2.keys()
	assert np.allclose(v_tnd2, tendencies2['y_velocity'])

	u_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	u_tnd4[2:-2, 2:-2, :] = - (mtg[:-4, 2:-2, :] - 8. * mtg[1:-3, 2:-2, :] +
						       8. * mtg[3:-1, 2:-2, :] - mtg[4:, 2:-2, :]) / (12. * dx)
	assert 'x_velocity' in tendencies4.keys()
	assert np.allclose(u_tnd4, tendencies4['x_velocity'])

	v_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	v_tnd4[2:-2, 2:-2, :] = - (mtg[2:-2, :-4, :] - 8. * mtg[2:-2, 1:-3, :] +
							   8. * mtg[2:-2, 3:-1, :] - mtg[2:-2, 4:, :]) / (12. * dy)
	assert 'y_velocity' in tendencies4.keys()
	assert np.allclose(v_tnd4, tendencies4['y_velocity'])


def test_nonconservative_periodic_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = NonconservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=backend, dtype=dtype)
	pg4 = NonconservativeIsentropicPressureGradient(
		grid, 4, 'periodic', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	mtg = state['montgomery_potential'].values
	_mtg2 = np.concatenate((mtg[-2:-1, :, :], mtg, mtg[1:2, :, :]), axis=0)
	mtg2  = np.concatenate((_mtg2[:, -2:-1, :], _mtg2, _mtg2[:, 1:2, :]), axis=1)
	_mtg4 = np.concatenate((mtg[-3:-1, :, :], mtg, mtg[1:3, :, :]), axis=0)
	mtg4  = np.concatenate((_mtg4[:, -3:-1, :], _mtg4, _mtg4[:, 1:3, :]), axis=1)

	u_tnd2 = - (mtg2[2:, 1:-1, :] - mtg2[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_velocity' in tendencies2.keys()
	assert np.allclose(u_tnd2, tendencies2['x_velocity'])

	v_tnd2 = - (mtg2[1:-1, 2:, :] - mtg2[1:-1, :-2, :]) / (2. * dy)
	assert 'y_velocity' in tendencies2.keys()
	assert np.allclose(v_tnd2, tendencies2['y_velocity'])

	u_tnd4 = - (mtg4[:-4, 2:-2, :] - 8. * mtg4[1:-3, 2:-2, :] +
			    8. * mtg4[3:-1, 2:-2, :] - mtg4[4:, 2:-2, :]) / (12. * dx)
	assert 'x_velocity' in tendencies4.keys()
	assert np.allclose(u_tnd4, tendencies4['x_velocity'])

	v_tnd4 = - (mtg4[2:-2, :-4, :] - 8. * mtg4[2:-2, 1:-3, :] +
			    8. * mtg4[2:-2, 3:-1, :] - mtg4[2:-2, 4:, :]) / (12. * dy)
	assert 'y_velocity' in tendencies4.keys()
	assert np.allclose(v_tnd4, tendencies4['y_velocity'])


def test_conservative_relaxed_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = ConservativeIsentropicPressureGradient(
		grid, 2, 'relaxed', backend=backend, dtype=dtype)
	pg4 = ConservativeIsentropicPressureGradient(
		grid, 4, 'relaxed', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	s   = state['air_isentropic_density'].values
	mtg = state['montgomery_potential'].values

	su_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	su_tnd2[1:-1, 1:-1, :] = - s[1:-1, 1:-1, :] * \
							 (mtg[2:, 1:-1, :] - mtg[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(su_tnd2, tendencies2['x_momentum_isentropic'])

	sv_tnd2 = np.zeros((nx, ny, nz), dtype=dtype)
	sv_tnd2[1:-1, 1:-1, :] = - s[1:-1, 1:-1, :] * \
							 (mtg[1:-1, 2:, :] - mtg[1:-1, :-2, :]) / (2. * dy)
	assert 'y_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(sv_tnd2, tendencies2['y_momentum_isentropic'])

	su_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	su_tnd4[2:-2, 2:-2, :] = - s[2:-2, 2:-2, :] * \
							 (mtg[:-4, 2:-2, :] - 8. * mtg[1:-3, 2:-2, :] +
							  8. * mtg[3:-1, 2:-2, :] - mtg[4:, 2:-2, :]) / (12. * dx)
	assert 'x_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(su_tnd4, tendencies4['x_momentum_isentropic'])

	sv_tnd4 = np.zeros((nx, ny, nz), dtype=dtype)
	sv_tnd4[2:-2, 2:-2, :] = -s[2:-2, 2:-2, :] * \
							 (mtg[2:-2, :-4, :] - 8. * mtg[2:-2, 1:-3, :] +
							  8. * mtg[2:-2, 3:-1, :] - mtg[2:-2, 4:, :]) / (12. * dy)
	assert 'y_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(sv_tnd4, tendencies4['y_momentum_isentropic'])


def test_conservative_periodic_bcs(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	backend = gt.mode.NUMPY
	dtype = np.float32

	pg2 = ConservativeIsentropicPressureGradient(
		grid, 2, 'periodic', backend=backend, dtype=dtype)
	pg4 = ConservativeIsentropicPressureGradient(
		grid, 4, 'periodic', backend=backend, dtype=dtype)

	tendencies2, _ = pg2(state)
	tendencies4, diagnostics = pg4(state)

	assert diagnostics == {}

	s   = state['air_isentropic_density'].values
	mtg = state['montgomery_potential'].values

	_s2   = np.concatenate((s[-2:-1, :, :], s, s[1:2, :, :]), axis=0)
	s2    = np.concatenate((_s2[:, -2:-1, :], _s2, _s2[:, 1:2, :]), axis=1)
	_s4   = np.concatenate((s[-3:-1, :, :], s, s[1:3, :, :]), axis=0)
	s4    = np.concatenate((_s4[:, -3:-1, :], _s4, _s4[:, 1:3, :]), axis=1)
	_mtg2 = np.concatenate((mtg[-2:-1, :, :], mtg, mtg[1:2, :, :]), axis=0)
	mtg2  = np.concatenate((_mtg2[:, -2:-1, :], _mtg2, _mtg2[:, 1:2, :]), axis=1)
	_mtg4 = np.concatenate((mtg[-3:-1, :, :], mtg, mtg[1:3, :, :]), axis=0)
	mtg4  = np.concatenate((_mtg4[:, -3:-1, :], _mtg4, _mtg4[:, 1:3, :]), axis=1)

	su_tnd2 = - s2[1:-1, 1:-1, :] * (mtg2[2:, 1:-1, :] - mtg2[:-2, 1:-1, :]) / (2. * dx)
	assert 'x_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(su_tnd2, tendencies2['x_momentum_isentropic'])

	sv_tnd2 = - s2[1:-1, 1:-1, :] * (mtg2[1:-1, 2:, :] - mtg2[1:-1, :-2, :]) / (2. * dy)
	assert 'y_momentum_isentropic' in tendencies2.keys()
	assert np.allclose(sv_tnd2, tendencies2['y_momentum_isentropic'])

	su_tnd4 = - s4[2:-2, 2:-2, :] * \
			  (mtg4[:-4, 2:-2, :] - 8. * mtg4[1:-3, 2:-2, :] +
			   8. * mtg4[3:-1, 2:-2, :] - mtg4[4:, 2:-2, :]) / (12. * dx)
	assert 'x_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(su_tnd4, tendencies4['x_momentum_isentropic'])

	sv_tnd4 = - s4[2:-2, 2:-2, :] * \
			  (mtg4[2:-2, :-4, :] - 8. * mtg4[2:-2, 1:-3, :] +
			   8. * mtg4[2:-2, 3:-1, :] - mtg4[2:-2, 4:, :]) / (12. * dy)
	assert 'y_momentum_isentropic' in tendencies4.keys()
	assert np.allclose(sv_tnd4, tendencies4['y_momentum_isentropic'])


def test_isentropic_vertical_flux_dry(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	dz = grid.dz.to_units('K').values.item()
	s  = state['air_isentropic_density'].values
	su = state['x_momentum_isentropic'].values
	sv = state['y_momentum_isentropic'].values

	placeholder = state['air_pressure_on_interface_levels'].values.copy()
	state.pop('air_pressure_on_interface_levels')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')

	backend = gt.mode.NUMPY

	#
	# tendency_of_air_potential_temperature
	#
	ivf = VerticalIsentropicAdvection(grid, moist_on=False, backend=backend)

	assert isinstance(ivf, VerticalIsentropicAdvection)

	state['tendency_of_air_potential_temperature'] = \
		make_data_array_3d(placeholder[:, :, :-1], grid, 'K s^-1')
	dtheta_dt = placeholder[:, :, :-1]

	tendencies, diagnostics = ivf(state)

	t_s = 0.5 * (dtheta_dt[:, :, :-2] * s[:, :, :-2] - dtheta_dt[:, :, 2:] * s[:, :, 2:]) / dz
	assert 'air_isentropic_density' in tendencies.keys()
	assert np.allclose(t_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	t_su = 0.5 * (dtheta_dt[:, :, :-2] * su[:, :, :-2] - dtheta_dt[:, :, 2:] * su[:, :, 2:]) / dz
	assert 'x_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	t_sv = 0.5 * (dtheta_dt[:, :, :-2] * sv[:, :, :-2] - dtheta_dt[:, :, 2:] * sv[:, :, 2:]) / dz
	assert 'y_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	assert tendencies == {}
	assert diagnostics == {}

	#
	# tendency_of_air_potential_temperature_on_interface_levels
	#
	ivf = VerticalIsentropicAdvection(grid, moist_on=False, backend=backend,
								 tendency_of_air_potential_temperature_on_interface_levels=True)

	assert isinstance(ivf, VerticalIsentropicAdvection)

	state.pop('tendency_of_air_potential_temperature')
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_data_array_3d(placeholder, grid, 'K s^-1')
	dtheta_dt = placeholder

	tendencies, diagnostics = ivf(state)

	t_s = 0.5 * (dtheta_dt[:, :, 1:-2] * (s[:, :, :-2] + s[:, :, 1:-1]) -
				 dtheta_dt[:, :, 2:-1] * (s[:, :, 1:-1] + s[:, :, 2:])) / dz
	assert 'air_isentropic_density' in tendencies.keys()
	assert np.allclose(t_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	t_su = 0.5 * (dtheta_dt[:, :, 1:-2] * (su[:, :, :-2] + su[:, :, 1:-1]) -
				  dtheta_dt[:, :, 2:-1] * (su[:, :, 1:-1] + su[:, :, 2:])) / dz
	assert 'x_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	t_sv = 0.5 * (dtheta_dt[:, :, 1:-2] * (sv[:, :, :-2] + sv[:, :, 1:-1]) -
				  dtheta_dt[:, :, 2:-1] * (sv[:, :, 1:-1] + sv[:, :, 2:])) / dz
	assert 'y_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	assert tendencies == {}
	assert diagnostics == {}


def test_isentropic_vertical_flux_moist(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	dz = grid.dz.to_units('K').values.item()
	s  = state['air_isentropic_density'].values
	su = state['x_momentum_isentropic'].values
	sv = state['y_momentum_isentropic'].values
	qv = state['mass_fraction_of_water_vapor_in_air'].values
	qc = state['mass_fraction_of_cloud_liquid_water_in_air'].values
	qr = state['mass_fraction_of_precipitation_water_in_air'].values

	placeholder = state['air_pressure_on_interface_levels'].values.copy()
	state.pop('air_pressure_on_interface_levels')
	state.pop('exner_function_on_interface_levels')
	state.pop('height_on_interface_levels')
	state.pop('montgomery_potential')

	backend = gt.mode.NUMPY

	#
	# tendency_of_air_potential_temperature
	#
	ivf = IsentropicVerticalFlux(grid, moist_on=True, backend=backend,
								 tendency_of_air_potential_temperature_on_interface_levels=False)

	assert isinstance(ivf, IsentropicVerticalFlux)

	state['tendency_of_air_potential_temperature'] = \
		make_data_array_3d(placeholder[:, :, :-1], grid, 'K s^-1')
	dtheta_dt = placeholder[:, :, :-1]

	tendencies, diagnostics = ivf(state)

	t_s = 0.5 * (dtheta_dt[:, :, :-2] * s[:, :, :-2] - dtheta_dt[:, :, 2:] * s[:, :, 2:]) / dz
	assert 'air_isentropic_density' in tendencies.keys()
	assert np.allclose(t_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	t_su = 0.5 * (dtheta_dt[:, :, :-2] * su[:, :, :-2] - dtheta_dt[:, :, 2:] * su[:, :, 2:]) / dz
	assert 'x_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	t_sv = 0.5 * (dtheta_dt[:, :, :-2] * sv[:, :, :-2] - dtheta_dt[:, :, 2:] * sv[:, :, 2:]) / dz
	assert 'y_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	t_qv = 0.5 * (dtheta_dt[:, :, :-2] * qv[:, :, :-2] - dtheta_dt[:, :, 2:] * qv[:, :, 2:]) / dz
	assert 'mass_fraction_of_water_vapor_in_air' in tendencies.keys()
	assert np.allclose(t_qv, tendencies['mass_fraction_of_water_vapor_in_air'][:, :, 1:-1])
	tendencies.pop('mass_fraction_of_water_vapor_in_air')

	t_qc = 0.5 * (dtheta_dt[:, :, :-2] * qc[:, :, :-2] - dtheta_dt[:, :, 2:] * qc[:, :, 2:]) / dz
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in tendencies.keys()
	assert np.allclose(t_qc, tendencies['mass_fraction_of_cloud_liquid_water_in_air'][:, :, 1:-1])
	tendencies.pop('mass_fraction_of_cloud_liquid_water_in_air')

	t_qr = 0.5 * (dtheta_dt[:, :, :-2] * qr[:, :, :-2] - dtheta_dt[:, :, 2:] * qr[:, :, 2:]) / dz
	assert 'mass_fraction_of_precipitation_water_in_air' in tendencies.keys()
	assert np.allclose(t_qr, tendencies['mass_fraction_of_precipitation_water_in_air'][:, :, 1:-1])
	tendencies.pop('mass_fraction_of_precipitation_water_in_air')

	assert tendencies == {}
	assert diagnostics == {}

	#
	# tendency_of_air_potential_temperature_on_interface_levels
	#
	ivf = IsentropicVerticalFlux(grid, moist_on=True, backend=backend,
								 tendency_of_air_potential_temperature_on_interface_levels=True)

	assert isinstance(ivf, IsentropicVerticalFlux)

	state.pop('tendency_of_air_potential_temperature')
	state['tendency_of_air_potential_temperature_on_interface_levels'] = \
		make_data_array_3d(placeholder, grid, 'K s^-1')
	dtheta_dt = placeholder

	tendencies, diagnostics = ivf(state)

	t_s = 0.5 * (dtheta_dt[:, :, 1:-2] * (s[:, :, :-2] + s[:, :, 1:-1]) -
				 dtheta_dt[:, :, 2:-1] * (s[:, :, 1:-1] + s[:, :, 2:])) / dz
	assert 'air_isentropic_density' in tendencies.keys()
	assert np.allclose(t_s, tendencies['air_isentropic_density'][:, :, 1:-1])
	tendencies.pop('air_isentropic_density')

	t_su = 0.5 * (dtheta_dt[:, :, 1:-2] * (su[:, :, :-2] + su[:, :, 1:-1]) -
				  dtheta_dt[:, :, 2:-1] * (su[:, :, 1:-1] + su[:, :, 2:])) / dz
	assert 'x_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_su, tendencies['x_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('x_momentum_isentropic')

	t_sv = 0.5 * (dtheta_dt[:, :, 1:-2] * (sv[:, :, :-2] + sv[:, :, 1:-1]) -
				  dtheta_dt[:, :, 2:-1] * (sv[:, :, 1:-1] + sv[:, :, 2:])) / dz
	assert 'y_momentum_isentropic' in tendencies.keys()
	assert np.allclose(t_sv, tendencies['y_momentum_isentropic'][:, :, 1:-1])
	tendencies.pop('y_momentum_isentropic')

	t_qv = 0.5 * (dtheta_dt[:, :, 1:-2] * (qv[:, :, :-2] + qv[:, :, 1:-1]) -
				  dtheta_dt[:, :, 2:-1] * (qv[:, :, 1:-1] + qv[:, :, 2:])) / dz
	assert 'mass_fraction_of_water_vapor_in_air' in tendencies.keys()
	assert np.allclose(t_qv, tendencies['mass_fraction_of_water_vapor_in_air'][:, :, 1:-1])
	tendencies.pop('mass_fraction_of_water_vapor_in_air')

	t_qc = 0.5 * (dtheta_dt[:, :, 1:-2] * (qc[:, :, :-2] + qc[:, :, 1:-1]) -
				  dtheta_dt[:, :, 2:-1] * (qc[:, :, 1:-1] + qc[:, :, 2:])) / dz
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in tendencies.keys()
	assert np.allclose(t_qc, tendencies['mass_fraction_of_cloud_liquid_water_in_air'][:, :, 1:-1])
	tendencies.pop('mass_fraction_of_cloud_liquid_water_in_air')

	t_qr = 0.5 * (dtheta_dt[:, :, 1:-2] * (qr[:, :, :-2] + qr[:, :, 1:-1]) -
				  dtheta_dt[:, :, 2:-1] * (qr[:, :, 1:-1] + qr[:, :, 2:])) / dz
	assert 'mass_fraction_of_precipitation_water_in_air' in tendencies.keys()
	assert np.allclose(t_qr, tendencies['mass_fraction_of_precipitation_water_in_air'][:, :, 1:-1])
	tendencies.pop('mass_fraction_of_precipitation_water_in_air')

	assert tendencies == {}
	assert diagnostics == {}


def test_prescribed_surface_heating(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	amplitude_during_daytime = DataArray(0.8, attrs={'units': 'kW m^-2'})
	amplitude_at_night = DataArray(-75000.0, attrs={'units': 'mW m^-2'})
	attenuation_coefficient_during_daytime = DataArray(1.0/6.0, attrs={'units': 'hm^-1'})
	attenuation_coefficient_at_night = DataArray(1.0/75.0, attrs={'units': 'm^-1'})
	characteristic_length = DataArray(25.0, attrs={'units': 'km'})

	#
	# tendencies_in_diagnostics=False
	#
	psh = PrescribedSurfaceHeating(grid, tendencies_in_diagnostics=False,
								   air_pressure_on_interface_levels=True)

	assert 'air_pressure_on_interface_levels' in psh.input_properties
	assert 'height_on_interface_levels' in psh.input_properties
	assert len(psh.input_properties) == 2

	assert 'air_potential_temperature_on_interface_levels' in psh.tendency_properties
	assert len(psh.tendency_properties) == 1

	assert psh.diagnostic_properties == {}

	#
	# tendencies_in_diagnostics=True
	#
	psh = PrescribedSurfaceHeating(grid, tendencies_in_diagnostics=True,
								   air_pressure_on_interface_levels=False)

	assert 'air_pressure' in psh.input_properties
	assert 'height_on_interface_levels' in psh.input_properties
	assert len(psh.input_properties) == 2

	assert 'air_potential_temperature' in psh.tendency_properties
	assert len(psh.tendency_properties) == 1

	assert 'tendency_of_air_potential_temperature' in psh.diagnostic_properties
	assert len(psh.diagnostic_properties) == 1

	#
	# air_pressure_on_interface_levels=True
	#
	state['time'] = datetime(year=1992, month=2, day=20, hour=15)
	starting_time = state['time'] - timedelta(hours=2)

	psh = PrescribedSurfaceHeating(
		grid, tendencies_in_diagnostics=True,
		air_pressure_on_interface_levels=True,
		amplitude_during_daytime=amplitude_during_daytime,
		amplitude_at_night=amplitude_at_night,
		attenuation_coefficient_during_daytime=attenuation_coefficient_during_daytime,
		attenuation_coefficient_at_night=attenuation_coefficient_at_night,
		characteristic_length=characteristic_length,
		starting_time=starting_time,
	)

	assert equal_to(psh._f0d, 800.0)
	assert equal_to(psh._f0n, -75.0)
	assert equal_to(psh._ad, 1.0/600.0)
	assert equal_to(psh._an, 1.0/75.0)
	assert equal_to(psh._cl, 25000.0)

	tendencies, diagnostics = psh(state)

	assert 'air_potential_temperature_on_interface_levels' in tendencies
	assert len(tendencies) == 1

	assert 'tendency_of_air_potential_temperature_on_interface_levels' in diagnostics
	assert len(diagnostics) == 1

	print('here')

	#
	# air_pressure_on_interface_levels=False
	#
	state['time'] = datetime(year=1992, month=2, day=20, hour=3)
	starting_time = state['time'] - timedelta(hours=2)
	p = state['air_pressure_on_interface_levels'].values
	state['air_pressure'] = make_data_array_3d(0.5 * (p[:, :, :-1] + p[:, :, 1:]), grid, 'Pa')
	state.pop('air_pressure_on_interface_levels')

	psh = PrescribedSurfaceHeating(
		grid, tendencies_in_diagnostics=False,
		air_pressure_on_interface_levels=False,
		amplitude_during_daytime=amplitude_during_daytime,
		amplitude_at_night=amplitude_at_night,
		attenuation_coefficient_during_daytime=attenuation_coefficient_during_daytime,
		attenuation_coefficient_at_night=attenuation_coefficient_at_night,
		characteristic_length=characteristic_length,
		starting_time=starting_time,
	)

	assert equal_to(psh._f0d, 800.0)
	assert equal_to(psh._f0n, -75.0)
	assert equal_to(psh._ad, 1.0/600.0)
	assert equal_to(psh._an, 1.0/75.0)
	assert equal_to(psh._cl, 25000.0)

	tendencies, diagnostics = psh(state)

	assert 'air_potential_temperature' in tendencies
	assert len(tendencies) == 1

	assert len(diagnostics) == 0

	print('here')


if __name__ == '__main__':
	#pytest.main([__file__])
	from conftest import isentropic_dry_data
	test_prescribed_surface_heating(isentropic_dry_data())
