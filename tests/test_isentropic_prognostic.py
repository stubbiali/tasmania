from datetime import timedelta
import numpy as np
import pytest

import gridtools as gt
from tasmania.dynamics.diagnostics import IsentropicDiagnostics, HorizontalVelocity
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.dynamics._isentropic_prognostic import _Centered, _ForwardEuler, \
													 _RK2, _RK3COSMO, _RK3
from tasmania.physics.microphysics import RaindropFallVelocity
from tasmania.utils.data_utils import make_raw_state


mf_wv  = 'mass_fraction_of_water_vapor_in_air'
mf_clw = 'mass_fraction_of_cloud_liquid_water_in_air'
mf_pw  = 'mass_fraction_of_precipitation_water_in_air'


def test_factory(grid):
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('relaxed', grid, 1)
	rfv = RaindropFallVelocity(grid, backend)

	ip_centered = IsentropicPrognostic.factory('centered', grid, True, diags, hb,
											   horizontal_flux_scheme='centered',
											   adiabatic_flow=True,
											   sedimentation_on=True,
											   sedimentation_flux_scheme='second_order_upwind',
											   sedimentation_substeps=2,
											   raindrop_fall_velocity_diagnostic=rfv,
											   backend=backend)
	ip_euler = IsentropicPrognostic.factory('forward_euler', grid, True, diags, hb,
											horizontal_flux_scheme='upwind',
											adiabatic_flow=True,
											sedimentation_on=True,
											sedimentation_flux_scheme='first_order_upwind',
											sedimentation_substeps=1,
											raindrop_fall_velocity_diagnostic=rfv,
											backend=backend)
	ip_rk2 = IsentropicPrognostic.factory('rk2', grid, True, diags, hb,
										  horizontal_flux_scheme='third_order_upwind',
										  adiabatic_flow=True,
										  sedimentation_on=True,
										  sedimentation_flux_scheme='first_order_upwind',
										  sedimentation_substeps=1,
										  raindrop_fall_velocity_diagnostic=rfv,
										  backend=backend)
	ip_rk3c = IsentropicPrognostic.factory('rk3cosmo', grid, True, diags, hb,
										   horizontal_flux_scheme='fifth_order_upwind',
										   adiabatic_flow=True,
										   sedimentation_on=True,
										   sedimentation_flux_scheme='first_order_upwind',
										   sedimentation_substeps=1,
										   raindrop_fall_velocity_diagnostic=rfv,
										   backend=backend)
	ip_rk3 = IsentropicPrognostic.factory('rk3', grid, True, diags, hb,
										  horizontal_flux_scheme='fifth_order_upwind',
										  adiabatic_flow=True,
										  sedimentation_on=True,
										  sedimentation_flux_scheme='first_order_upwind',
										  sedimentation_substeps=1,
										  raindrop_fall_velocity_diagnostic=rfv,
										  backend=backend)

	assert isinstance(ip_centered, _Centered)
	assert isinstance(ip_euler, _ForwardEuler)
	assert isinstance(ip_rk2, _RK2)
	assert isinstance(ip_rk3c, _RK3COSMO)
	assert isinstance(ip_rk3, _RK3)


def test_leapfrog(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('periodic', grid, 1)
	rfv = RaindropFallVelocity(grid, backend)

	ip_centered = IsentropicPrognostic.factory('centered', grid, True, diags, hb,
											   horizontal_flux_scheme='centered',
											   sedimentation_on=True,
											   sedimentation_flux_scheme='first_order_upwind',
											   raindrop_fall_velocity_diagnostic=rfv,
											   backend=backend, dtype=np.float64)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_wv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_clw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_pw]

	raw_tendencies = {}

	raw_state_prv = ip_centered.step_neglecting_vertical_motion(0, dt, raw_state, raw_tendencies)

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

	raw_state_prv[mf_pw] = \
		raw_state_prv['isentropic_density_of_precipitation_water'] / \
		raw_state_prv['air_isentropic_density']
	raw_state_new = ip_centered.step_integrating_sedimentation_flux(0, dt, raw_state,
																	raw_state_prv)

	#assert 'time' in raw_state_new.keys()
	#assert raw_state_new['time'] == raw_state['time'] + dt

	assert np.allclose(s_prv[1:-1, :, :], raw_state_prv['air_isentropic_density'][1:-1, :, :])


def test_upwind(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('periodic', grid, 1)
	rfv = RaindropFallVelocity(grid, backend)

	ip_euler = IsentropicPrognostic.factory('forward_euler', grid, True, diags, hb,
											horizontal_flux_scheme='upwind',
											sedimentation_on=True,
											sedimentation_flux_scheme='first_order_upwind',
											raindrop_fall_velocity_diagnostic=rfv,
											backend=backend, dtype=np.float64)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_wv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_clw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_pw]

	raw_tendencies = {}

	raw_state_prv = ip_euler.step_neglecting_vertical_motion(0, dt, raw_state, raw_tendencies)

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

	raw_state_prv[mf_pw] = \
		raw_state_prv['isentropic_density_of_precipitation_water'] / \
		raw_state_prv['air_isentropic_density']
	raw_state_new = ip_euler.step_integrating_sedimentation_flux(0, dt, raw_state, raw_state_prv)

	#assert 'time' in raw_state_new.keys()
	#assert raw_state_new['time'] == raw_state['time'] + dt

	assert np.allclose(s_prv, raw_state_prv['air_isentropic_density'])


def test_rk2(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('periodic', grid)
	hv = HorizontalVelocity(grid, dtype=np.float64)
	rfv = RaindropFallVelocity(grid, backend)

	ip_rk2 = IsentropicPrognostic.factory('rk2', grid, True, diags, hb,
										  horizontal_flux_scheme='third_order_upwind',
										  sedimentation_on=True,
										  sedimentation_flux_scheme='second_order_upwind',
										  raindrop_fall_velocity_diagnostic=rfv,
										  backend=backend, dtype=np.float64)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_wv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_clw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_pw]

	raw_tendencies = {}

	raw_state_1 = ip_rk2.step_neglecting_vertical_motion(0, dt, raw_state, raw_tendencies)

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	mtg = raw_state['montgomery_potential']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']
	pt  = raw_state['air_pressure_on_interface_levels'][0, 0, 0]

	s_   = hb.from_physical_to_computational_domain(s)
	u_   = hb.from_physical_to_computational_domain(u)
	mtg_ = hb.from_physical_to_computational_domain(mtg)
	su_  = hb.from_physical_to_computational_domain(su)
	sqv_ = hb.from_physical_to_computational_domain(sqv)
	sqc_ = hb.from_physical_to_computational_domain(sqc)
	sqr_ = hb.from_physical_to_computational_domain(sqr)

	flux4 = u_[2:-2, :, :] / 12. * (7. * (s_[2:-1, :, :] + s_[1:-2, :, :]) -
									1. * (s_[3:, :, :] + s_[:-3, :, :]))
	flux = flux4 - np.abs(u_[2:-2, :, :]) / 12. * (3. * (s_[2:-1, :, :] - s_[1:-2, :, :]) -
												   1. * (s_[3:, :, :] - s_[:-3, :, :]))
	s1 = s - 1./2. * dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :])
	assert 'air_isentropic_density' in raw_state_1.keys()
	assert np.allclose(s1, raw_state_1['air_isentropic_density'])

	flux4 = u_[2:-2, :, :] / 12. * (7. * (su_[2:-1, :, :] + su_[1:-2, :, :]) -
									1. * (su_[3:, :, :] + su_[:-3, :, :]))
	flux = flux4 - np.abs(u_[2:-2, :, :]) / 12. * (3. * (su_[2:-1, :, :] - su_[1:-2, :, :]) -
												   1. * (su_[3:, :, :] - su_[:-3, :, :]))
	su1 = su - 1./2. * dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :]) \
		  	 - 1./2. * dt.seconds * s_[2:-2, 2:-2, :] * (mtg_[3:-1, 2:-2, :] -
														 mtg_[1:-3, 2:-2, :]) / (2. * dx)
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
	_, _, mtg1, _ = diags.get_diagnostic_variables(s1, pt)
	raw_state_1['x_velocity_at_u_locations'] = u1
	raw_state_1['y_velocity_at_v_locations'] = v1
	raw_state_1['montgomery_potential'] = mtg1

	raw_state_2 = ip_rk2.step_neglecting_vertical_motion(1, dt, raw_state_1, raw_tendencies)

	s1_   = hb.from_physical_to_computational_domain(s1)
	u1_   = hb.from_physical_to_computational_domain(u1)
	mtg1_ = hb.from_physical_to_computational_domain(mtg1)
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
	su2 = su - dt.seconds / dx * (flux[1:, 2:-2, :] - flux[:-1, 2:-2, :]) \
		  	 - dt.seconds * s1_[2:-2, 2:-2, :] * (mtg1_[3:-1, 2:-2, :] -
												  mtg1_[1:-3, 2:-2, :]) / (2. * dx)
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
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('periodic', grid)
	hv = HorizontalVelocity(grid, dtype=np.float64)
	rfv = RaindropFallVelocity(grid, backend)

	ip_rk3 = IsentropicPrognostic.factory('rk3cosmo', grid, True, diags, hb,
										  horizontal_flux_scheme='fifth_order_upwind',
										  sedimentation_on=True,
										  sedimentation_flux_scheme='second_order_upwind',
										  raindrop_fall_velocity_diagnostic=rfv,
										  backend=backend, dtype=np.float64)

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_wv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_clw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_pw]

	raw_tendencies = {}

	raw_state_1 = ip_rk3.step_neglecting_vertical_motion(0, dt, raw_state, raw_tendencies)

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	mtg = raw_state['montgomery_potential']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']
	pt  = raw_state['air_pressure_on_interface_levels'][0, 0, 0]

	s_   = hb.from_physical_to_computational_domain(s)
	u_   = hb.from_physical_to_computational_domain(u)
	mtg_ = hb.from_physical_to_computational_domain(mtg)
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
	su1 = su - 1./3. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :]) \
		     - 1./3. * dt.seconds * s_[3:-3, 3:-3, :] * (1./12. * mtg_[1:-5, 3:-3, :] -
													  	 8./12. * mtg_[2:-4, 3:-3, :] +
													  	 8./12. * mtg_[4:-2, 3:-3, :] -
													  	 1./12. * mtg_[5:-1, 3:-3, :]) / dx
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
	_, _, mtg1, _ = diags.get_diagnostic_variables(s1, pt)
	raw_state_1['x_velocity_at_u_locations'] = u1
	raw_state_1['y_velocity_at_v_locations'] = v1
	raw_state_1['montgomery_potential'] = mtg1

	raw_state_2 = ip_rk3.step_neglecting_vertical_motion(1, dt, raw_state_1, raw_tendencies)

	s1_   = hb.from_physical_to_computational_domain(s1)
	u1_   = hb.from_physical_to_computational_domain(u1)
	mtg1_ = hb.from_physical_to_computational_domain(mtg1)
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
	su2 = su - 1./2. * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :]) \
		     - 1./2. * dt.seconds * s1_[3:-3, 3:-3, :] * (1./12. * mtg1_[1:-5, 3:-3, :] -
												   		  8./12. * mtg1_[2:-4, 3:-3, :] +
												   		  8./12. * mtg1_[4:-2, 3:-3, :] -
												   		  1./12. * mtg1_[5:-1, 3:-3, :]) / dx
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
	_, _, mtg2, _ = diags.get_diagnostic_variables(s2, pt)
	raw_state_2['x_velocity_at_u_locations'] = u2
	raw_state_2['y_velocity_at_v_locations'] = v2
	raw_state_2['montgomery_potential'] = mtg2

	raw_state_3 = ip_rk3.step_neglecting_vertical_motion(2, dt, raw_state_2, raw_tendencies)

	s2_   = hb.from_physical_to_computational_domain(s2)
	u2_   = hb.from_physical_to_computational_domain(u2)
	mtg2_ = hb.from_physical_to_computational_domain(mtg2)
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
	su3 = su - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :]) \
		  	 - dt.seconds * s2_[3:-3, 3:-3, :] * (1./12. * mtg2_[1:-5, 3:-3, :] -
												  8./12. * mtg2_[2:-4, 3:-3, :] +
												  8./12. * mtg2_[4:-2, 3:-3, :] -
												  1./12. * mtg2_[5:-1, 3:-3, :]) / dx
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
	backend = gt.mode.NUMPY

	diags = IsentropicDiagnostics(grid)
	hb = HorizontalBoundary.factory('periodic', grid)
	hv = HorizontalVelocity(grid, dtype=np.float64)
	rfv = RaindropFallVelocity(grid, backend)

	ip_rk3 = IsentropicPrognostic.factory('rk3', grid, True, diags, hb,
										  horizontal_flux_scheme='fifth_order_upwind',
										  sedimentation_on=True,
										  sedimentation_flux_scheme='second_order_upwind',
										  raindrop_fall_velocity_diagnostic=rfv,
										  backend=backend, dtype=np.float64)
	a1, a2 = ip_rk3._alpha1, ip_rk3._alpha2
	b21 = ip_rk3._beta21
	g0, g1, g2 = ip_rk3._gamma0, ip_rk3._gamma1, ip_rk3._gamma2

	dt = timedelta(seconds=10)

	raw_state = make_raw_state(state)
	raw_state['isentropic_density_of_water_vapor'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_wv]
	raw_state['isentropic_density_of_cloud_liquid_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_clw]
	raw_state['isentropic_density_of_precipitation_water'] = \
		raw_state['air_isentropic_density'] * raw_state[mf_pw]

	raw_tendencies = {}

	raw_state_1 = ip_rk3.step_neglecting_vertical_motion(0, dt, raw_state, raw_tendencies)

	dx  = grid.dx.values.item()
	s   = raw_state['air_isentropic_density']
	u   = raw_state['x_velocity_at_u_locations']
	mtg = raw_state['montgomery_potential']
	su  = raw_state['x_momentum_isentropic']
	sv  = raw_state['y_momentum_isentropic']
	sqv = raw_state['isentropic_density_of_water_vapor']
	sqc = raw_state['isentropic_density_of_cloud_liquid_water']
	sqr = raw_state['isentropic_density_of_precipitation_water']
	pt  = raw_state['air_pressure_on_interface_levels'][0, 0, 0]

	s_   = hb.from_physical_to_computational_domain(s)
	u_   = hb.from_physical_to_computational_domain(u)
	mtg_ = hb.from_physical_to_computational_domain(mtg)
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
	k0_su = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :]) \
		    - dt.seconds * s_[3:-3, 3:-3, :] * (1./12. * mtg_[1:-5, 3:-3, :] -
												8./12. * mtg_[2:-4, 3:-3, :] +
												8./12. * mtg_[4:-2, 3:-3, :] -
												1./12. * mtg_[5:-1, 3:-3, :]) / dx
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
	_, _, mtg1, _ = diags.get_diagnostic_variables(s1, pt)
	raw_state_1['x_velocity_at_u_locations'] = u1
	raw_state_1['y_velocity_at_v_locations'] = v1
	raw_state_1['montgomery_potential'] = mtg1

	raw_state_2 = ip_rk3.step_neglecting_vertical_motion(1, dt, raw_state_1, raw_tendencies)

	s1_   = hb.from_physical_to_computational_domain(s1)
	u1_   = hb.from_physical_to_computational_domain(u1)
	mtg1_ = hb.from_physical_to_computational_domain(mtg1)
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
	k1_su = - dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :]) \
		    - dt.seconds * s1_[3:-3, 3:-3, :] * (1./12. * mtg1_[1:-5, 3:-3, :] -
												 8./12. * mtg1_[2:-4, 3:-3, :] +
												 8./12. * mtg1_[4:-2, 3:-3, :] -
												 1./12. * mtg1_[5:-1, 3:-3, :]) / dx
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
	_, _, mtg2, _ = diags.get_diagnostic_variables(s2, pt)
	raw_state_2['x_velocity_at_u_locations'] = u2
	raw_state_2['y_velocity_at_v_locations'] = v2
	raw_state_2['montgomery_potential'] = mtg2

	raw_state_3 = ip_rk3.step_neglecting_vertical_motion(2, dt, raw_state_2, raw_tendencies)

	s2_   = hb.from_physical_to_computational_domain(s2)
	u2_   = hb.from_physical_to_computational_domain(u2)
	mtg2_ = hb.from_physical_to_computational_domain(mtg2)
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
		  - g2 * dt.seconds / dx * (flux[1:, 3:-3, :] - flux[:-1, 3:-3, :]) \
		  - g2 * dt.seconds * s2_[3:-3, 3:-3, :] * (1./12. * mtg2_[1:-5, 3:-3, :] -
											   		8./12. * mtg2_[2:-4, 3:-3, :] +
											   		8./12. * mtg2_[4:-2, 3:-3, :] -
											   		1./12. * mtg2_[5:-1, 3:-3, :]) / dx
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


if __name__ == '__main__':
	pytest.main([__file__])

