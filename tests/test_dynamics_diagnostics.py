import numpy as np
import pytest

from tasmania.dynamics.diagnostics import IsentropicDiagnostics, \
										  WaterConstituent, \
										  HorizontalVelocity


def test_isentropic_diagnostics(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	s  = state['air_isentropic_density'].values[:, :, :]
	pt = state['air_pressure_on_interface_levels'].values[0, 0, 0]

	ids = IsentropicDiagnostics(grid, dtype=np.float64)

	# Test get_diagnostic_variables
	p, exn, mtg, h = ids.get_diagnostic_variables(s, pt)
	assert np.allclose(p,   state['air_pressure_on_interface_levels'].values[:, :, :])
	assert np.allclose(exn, state['exner_function_on_interface_levels'].values[:, :, :])
	#assert np.allclose(mtg, state['montgomery_potential'].values[:, :, :])
	#assert np.allclose(h,   state['height_on_interface_levels'].values[:, :, :])

	# Test get_height
	#h = ids.get_height(s, pt)
	#assert np.allclose(h, state['height_on_interface_levels'].values[:, :, :])

	# Test get_air_density
	rho = ids.get_air_density(s, h)
	assert np.allclose(rho, state['air_density'].values[:, :, :])

	# Test get_air_temperature
	temp = ids.get_air_temperature(exn)
	assert np.allclose(temp, state['air_temperature'].values[:, :, :])


def test_water_constituent(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]

	s  = state['air_isentropic_density'].values[:, :, :]
	qv = state['mass_fraction_of_water_vapor_in_air'].values[:, :, :]
	qc = state['mass_fraction_of_cloud_liquid_water_in_air'].values[:, :, :]

	wc = WaterConstituent(grid)

	sqv = np.zeros((grid.nx, grid.ny, grid.nz), dtype=s.dtype)
	sqv_clip = np.zeros_like(sqv)
	qv_new = np.zeros_like(sqv)
	qv_new_clip = np.zeros_like(sqv)
	sqc = np.zeros_like(sqv)
	sqc_clip = np.zeros_like(sqv)
	qc_new = np.zeros_like(sqv)
	qc_new_clip = np.zeros_like(sqv)

	wc.get_density_of_water_constituent(s, qv, sqv, clipping=False)
	assert np.allclose(sqv, s*qv)
	wc.get_density_of_water_constituent(s, qv, sqv_clip, clipping=True)
	assert np.allclose(sqv, s*qv)
	assert np.allclose(sqv_clip, s*qv)

	wc.get_mass_fraction_of_water_constituent_in_air(s, sqv, qv_new, clipping=False)
	assert np.allclose(qv_new, qv)
	wc.get_mass_fraction_of_water_constituent_in_air(s, sqv, qv_new_clip, clipping=True)
	assert np.allclose(qv_new, qv)
	assert np.allclose(qv_new_clip, qv)

	wc.get_density_of_water_constituent(s, qc, sqc, clipping=False)
	assert np.allclose(sqc, s*qc)
	wc.get_density_of_water_constituent(s, qc, sqc_clip, clipping=True)
	assert np.allclose(sqc, s*qc)
	assert np.allclose(sqc_clip, s*qc)

	wc.get_mass_fraction_of_water_constituent_in_air(s, sqc, qc_new, clipping=False)
	assert np.allclose(qc_new, qc)
	wc.get_mass_fraction_of_water_constituent_in_air(s, sqc, qc_new_clip, clipping=True)
	assert np.allclose(qc_new, qc)
	assert np.allclose(qc_new_clip, qc)

	assert np.allclose(sqv, s*qv)
	assert np.allclose(sqv_clip, s*qv)
	assert np.allclose(qv_new, qv)
	assert np.allclose(qv_new_clip, qv)


def test_horizontal_velocity(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]

	s  = state['air_isentropic_density'].values[:, :, :]
	u  = state['x_velocity_at_u_locations'].values[:, :, :]
	v  = state['y_velocity_at_v_locations'].values[:, :, :]
	su = state['x_momentum_isentropic'].values[:, :, :]
	sv = state['y_momentum_isentropic'].values[:, :, :]

	hv = HorizontalVelocity(grid, dtype=s.dtype)

	u_new, v_new = hv.get_velocity_components(s, su, sv)
	assert np.allclose(u_new[1:-1, :, :], u[1:-1, :, :])
	assert np.allclose(v_new[:, 1:-1, :], v[:, 1:-1, :])

	#su_new, sv_new = hv.get_momenta(s, u, v)
	#assert np.allclose(su_new, su)
	#assert np.allclose(sv_new, sv)


if __name__ == '__main__':
	pytest.main([__file__])
