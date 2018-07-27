from datetime import timedelta
import numpy as np
import pytest

import gridtools as gt
from tasmania.dynamics.diagnostics import IsentropicDiagnostics
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.dynamics._isentropic_prognostic import _Centered, _ForwardEuler
from tasmania.physics.microphysics import RaindropFallVelocity
from tasmania.utils.data_utils import get_raw_state


def test_factory(grid):
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('relaxed', grid, 1)
	rfv = RaindropFallVelocity(grid, backend)

	ip_centered = IsentropicPrognostic.factory('centered', grid, True, backend, diags, hb,
											   horizontal_flux_scheme='centered',
											   physics_dynamics_coupling_on=False,
											   sedimentation_on=True,
											   sedimentation_flux_scheme='second_order_upwind',
											   sedimentation_substeps=2,
											   raindrop_fall_velocity_diagnostic=rfv)
	ip_euler = IsentropicPrognostic.factory('forward_euler', grid, True, backend, diags, hb,
											horizontal_flux_scheme='upwind',
											physics_dynamics_coupling_on=False,
											sedimentation_on=True,
											sedimentation_flux_scheme='first_order_upwind',
											sedimentation_substeps=1,
											raindrop_fall_velocity_diagnostic=rfv)

	assert isinstance(ip_centered, _Centered)
	assert isinstance(ip_euler, _ForwardEuler)


def test_leapfrog(grid_and_state):
	grid = grid_and_state[0]
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('periodic', grid, 1)
	rfv = RaindropFallVelocity(grid, backend)

	ip_centered = IsentropicPrognostic.factory('centered', grid, True, backend, diags, hb,
											   horizontal_flux_scheme='centered',
											   sedimentation_on=True,
											   sedimentation_flux_scheme='first_order_upwind',
											   raindrop_fall_velocity_diagnostic=rfv,
											   dtype=np.float64)

	dt = timedelta(seconds=10)

	raw_state = get_raw_state(grid_and_state[1])
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state['mass_fraction_of_water_vapor_in_air']
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state['mass_fraction_of_cloud_liquid_water_in_air']
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state['mass_fraction_of_precipitation_water_in_air']

	raw_tendencies = {}

	raw_state_prv = ip_centered.step_neglecting_vertical_advection(0, dt, raw_state, raw_tendencies)

	assert 'time' in raw_state_prv.keys()
	assert raw_state_prv['time'] == raw_state['time'] + dt

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	mtg = raw_state['montgomery_potential']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']

	s_prv = np.zeros_like(s, dtype=np.float64)
	s_prv[1:-1, :, :] = s[1:-1, :, :] - \
						dt.seconds * (su[2:, :, :] - su[:-2, :, :]) / dx
	assert 'air_isentropic_density' in raw_state_prv.keys()
	assert np.allclose(s_prv[1:-1, :, :], raw_state_prv['air_isentropic_density'][1:-1, :, :])

	su_prv = np.zeros_like(su, dtype=np.float64)
	su_prv[1:-1, :, :] = su[1:-1, :, :] - \
						 dt.seconds / dx * (u[2:-1, :, :] * (su[2:, :, :] + su[1:-1, :, :]) -
											u[1:-2, :, :] * (su[1:-1, :, :] + su[:-2, :, :])) - \
						 dt.seconds / dx * s[1:-1, :, :] * (mtg[2:, :, :] - mtg[:-2, :, :])
	assert 'x_momentum_isentropic' in raw_state_prv.keys()
	assert np.allclose(su_prv[1:-1, :, :], raw_state_prv['x_momentum_isentropic'][1:-1, :, :])

	assert 'y_momentum_isentropic' in raw_state_prv.keys()
	assert np.allclose(sv, raw_state_prv['y_momentum_isentropic'])

	sqv_prv = np.zeros_like(sqv, dtype=np.float64)
	sqv_prv[1:-1, :, :] = sqv[1:-1, :, :] - \
						  dt.seconds / dx * (u[2:-1, :, :] * (sqv[2:, :, :] + sqv[1:-1, :, :]) -
											 u[1:-2, :, :] * (sqv[1:-1, :, :] + sqv[:-2, :, :]))
	assert 'isentropic_density_of_water_vapor' in raw_state_prv.keys()
	assert np.allclose(sqv_prv[1:-1, :, :],
					   raw_state_prv['isentropic_density_of_water_vapor'][1:-1, :, :])

	sqc_prv = np.zeros_like(sqc, dtype=np.float64)
	sqc_prv[1:-1, :, :] = sqc[1:-1, :, :] - \
						  dt.seconds / dx * (u[2:-1, :, :] * (sqc[2:, :, :] + sqc[1:-1, :, :]) -
											 u[1:-2, :, :] * (sqc[1:-1, :, :] + sqc[:-2, :, :]))
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_prv.keys()
	assert np.allclose(sqc_prv[1:-1, :, :],
					   raw_state_prv['isentropic_density_of_cloud_liquid_water'][1:-1, :, :])

	sqr_prv = np.zeros_like(sqr, dtype=np.float64)
	sqr_prv[1:-1, :, :] = sqr[1:-1, :, :] - \
						  dt.seconds / dx * (u[2:-1, :, :] * (sqr[2:, :, :] + sqr[1:-1, :, :]) -
											 u[1:-2, :, :] * (sqr[1:-1, :, :] + sqr[:-2, :, :]))
	assert 'isentropic_density_of_precipitation_water' in raw_state_prv.keys()
	assert np.allclose(sqr_prv[1:-1, :, :],
					   raw_state_prv['isentropic_density_of_precipitation_water'][1:-1, :, :])

	raw_state_new = ip_centered.step_integrating_sedimentation_flux(0, dt, raw_state,
																	raw_state_prv)

	assert 'time' in raw_state_new.keys()
	assert raw_state_new['time'] == raw_state['time'] + dt

	assert np.allclose(s_prv[1:-1, :, :], raw_state_prv['air_isentropic_density'][1:-1, :, :])


def test_upwind(grid_and_state):
	grid = grid_and_state[0]
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('periodic', grid, 1)
	rfv = RaindropFallVelocity(grid, backend)

	ip_euler = IsentropicPrognostic.factory('forward_euler', grid, True, backend, diags, hb,
											horizontal_flux_scheme='upwind',
											sedimentation_on=True,
											sedimentation_flux_scheme='first_order_upwind',
											raindrop_fall_velocity_diagnostic=rfv,
											dtype=np.float64)

	dt = timedelta(seconds=10)

	raw_state = get_raw_state(grid_and_state[1])
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state['mass_fraction_of_water_vapor_in_air']
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state['mass_fraction_of_cloud_liquid_water_in_air']
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state['mass_fraction_of_precipitation_water_in_air']

	raw_tendencies = {}

	raw_state_prv = ip_euler.step_neglecting_vertical_advection(0, dt, raw_state, raw_tendencies)

	assert 'time' in raw_state_prv.keys()
	assert raw_state_prv['time'] == raw_state['time'] + dt

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	pt  = raw_state['air_pressure_on_interface_levels'][0, 0, 0]
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * s[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * s[1:, :, :]
	flux_s_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	s_prv = s - dt.seconds / dx * (flux_s_x[1:, :, :] - flux_s_x[:-1, :, :])
	assert 'air_isentropic_density' in raw_state_prv.keys()
	assert np.allclose(s_prv, raw_state_prv['air_isentropic_density'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * su[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * su[1:, :, :]
	flux_su_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	su_tmp = su - dt.seconds / dx * (flux_su_x[1:, :, :] - flux_su_x[:-1, :, :])
	_, _, mtg_prv, _ = diags.get_diagnostic_variables(s_prv, pt)
	su_prv = su_tmp[1:-1, :, :] - \
			 0.5 * dt.seconds / dx * s_prv[1:-1, :, :] * (mtg_prv[2:, :, :] - mtg_prv[:-2, :, :])
	assert 'x_momentum_isentropic' in raw_state_prv.keys()
	assert np.allclose(su_prv, raw_state_prv['x_momentum_isentropic'][1:-1, :, :])

	assert np.allclose(np.zeros_like(sv, dtype=np.float64),
					   raw_state_prv['y_momentum_isentropic'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * sqv[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * sqv[1:, :, :]
	flux_sqv_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	sqv_prv = sqv - dt.seconds / dx * (flux_sqv_x[1:, :, :] - flux_sqv_x[:-1, :, :])
	assert 'isentropic_density_of_water_vapor' in raw_state_prv.keys()
	assert np.allclose(sqv_prv, raw_state_prv['isentropic_density_of_water_vapor'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * sqc[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * sqc[1:, :, :]
	flux_sqc_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	sqc_prv = sqc - dt.seconds / dx * (flux_sqc_x[1:, :, :] - flux_sqc_x[:-1, :, :])
	assert 'isentropic_density_of_cloud_liquid_water' in raw_state_prv.keys()
	assert np.allclose(sqc_prv, raw_state_prv['isentropic_density_of_cloud_liquid_water'])

	flux = (u[1:-1, :, :] > 0) * u[1:-1, :, :] * sqr[:-1, :, :] + \
		   (u[1:-1, :, :] < 0) * u[1:-1, :, :] * sqr[1:, :, :]
	flux_sqr_x = np.concatenate((flux[-1:, :, :], flux, flux[:1, :, :]), axis=0)
	sqr_prv = sqr - dt.seconds / dx * (flux_sqr_x[1:, :, :] - flux_sqr_x[:-1, :, :])
	assert 'isentropic_density_of_precipitation_water' in raw_state_prv.keys()
	assert np.allclose(sqr_prv, raw_state_prv['isentropic_density_of_precipitation_water'])

	raw_state_new = ip_euler.step_integrating_sedimentation_flux(0, dt, raw_state, raw_state_prv)

	assert 'time' in raw_state_new.keys()
	assert raw_state_new['time'] == raw_state['time'] + dt

	assert np.allclose(s_prv, raw_state_prv['air_isentropic_density'])


if __name__ == '__main__':
	pytest.main([__file__])
