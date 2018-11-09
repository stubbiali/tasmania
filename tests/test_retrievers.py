import numpy as np
import pytest

from tasmania.plot.retrievers import DataRetriever, DataRetrieverComposite


def test_field(isentropic_dry_data, isentropic_moist_sedimentation_data):
	#
	# dry
	#
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	field_name  = 'x_velocity_at_u_locations'
	field_units = 'km hr^-1'
	x = slice(20, None)
	y = slice(15, 40)

	dr = DataRetriever(grid, field_name, field_units, x=x, y=y)
	data = dr(state)

	assert data.shape[0] == 32
	assert data.shape[1] == 25
	assert data.shape[2] == 50
	assert np.allclose(data, state['x_velocity_at_u_locations'][20:, 15:40, :].values * 3.6)

	#
	# moist
	#
	grid, states = isentropic_moist_sedimentation_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	field_name  = 'mass_fraction_of_water_vapor_in_air'
	field_units = 'g kg^-1'
	x = slice(20, 30)
	z = slice(0, 30)

	dr = DataRetriever(grid, field_name, field_units, x=x, z=z)
	data = dr(state)

	assert data.shape[0] == 10
	assert data.shape[1] == 1
	assert data.shape[2] == 30
	assert np.allclose(data, state['mass_fraction_of_water_vapor_in_air'][20:30, :, :30].values * 1e3)


def test_horizontal_velocity(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	field_name  = 'horizontal_velocity'
	field_units = 'm s^-1'
	x = slice(20, None)
	y = slice(15, 40)
	z = slice(0, None)

	dr = DataRetriever(grid, field_name, field_units, x=x, y=y, z=z)
	data = dr(state)

	assert data.shape[0] == 31
	assert data.shape[1] == 25
	assert data.shape[2] == 50

	s  = state['air_isentropic_density'][x, y, :].values
	su = state['x_momentum_isentropic'][x, y, :].values
	sv = state['y_momentum_isentropic'][x, y, :].values
	assert np.allclose(data, np.sqrt((su/s)**2 + (sv/s)**2))


def test_height(isentropic_dry_data):
	#
	# height_on_interface_levels
	#
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	field_name  = 'height_on_interface_levels'
	field_units = 'km'

	dr = DataRetriever(grid, field_name, field_units)
	data = dr(state)

	refdata = state['height_on_interface_levels'].values * 1e-3

	assert data.shape[0] == refdata.shape[0]
	assert data.shape[1] == refdata.shape[1]
	assert data.shape[2] == refdata.shape[2]
	assert np.allclose(data, refdata)

	#
	# height
	#
	field_name  = 'height'
	field_units = 'cm'
	x = slice(10, 11)
	y = slice(-1, None)
	z = slice(0, None)

	dr = DataRetriever(grid, field_name, field_units, x=x, y=y, z=z)
	data = dr(state)

	tmp = state['height_on_interface_levels'][10:11, -1:, :].values * 1e2
	refdata = 0.5 * (tmp[:, :, :-1] + tmp[:, :, 1:])

	assert data.shape[0] == refdata.shape[0]
	assert data.shape[1] == refdata.shape[1]
	assert data.shape[2] == refdata.shape[2]
	assert np.allclose(data, refdata)


def test_composite(isentropic_dry_data, isentropic_moist_sedimentation_data):
	grid_d, states_d = isentropic_dry_data
	state_d = states_d[0]
	grid_d.update_topography(state_d['time'] - states_d[0]['time'])

	grid_m, states_m = isentropic_moist_sedimentation_data
	state_m = states_m[0]
	grid_m.update_topography(state_m['time'] - states_m[0]['time'])

	field_name_d  = 'horizontal_velocity'
	field_units_d = None
	xd = slice(10, 11)
	yd = slice(-1, None, None)
	zd = slice(0, None)

	field_name_m  = 'mass_fraction_of_precipitation_water_in_air'
	field_units_m = 'g kg^-1'
	xm = slice(10, 21)
	ym = slice(0, None)
	zm = slice(0, None)

	#
	# One input state
	#
	drc = DataRetrieverComposite(grid_d, field_name_d, field_units_d, xd, yd, zd)
	data = drc(state_d)

	assert data[0][0].shape[0] == 1
	assert data[0][0].shape[1] == 1
	assert data[0][0].shape[2] == 50

	s  = state_d['air_isentropic_density'][10:11, -1:, :].values
	su = state_d['x_momentum_isentropic'][10:11, -1:, :].values
	sv = state_d['y_momentum_isentropic'][10:11, -1:, :].values
	assert np.allclose(data, np.sqrt((su/s)**2 + (sv/s)**2))

	#
	# Two input states
	#
	drc = DataRetrieverComposite((grid_d, grid_m),
								 ((field_name_d, ), (field_name_m, )),
								 ((field_units_d, ), (field_units_m, )),
								 ((xd, ), (xm, )),
								 ((yd, ), (ym, )),
								 ((zd, ), (zm, )),)
	data = drc((state_d, state_m))

	assert data[0][0].shape[0] == 1
	assert data[0][0].shape[1] == 1
	assert data[0][0].shape[2] == 50

	s  = state_d['air_isentropic_density'][10:11, -1:, :].values
	su = state_d['x_momentum_isentropic'][10:11, -1:, :].values
	sv = state_d['y_momentum_isentropic'][10:11, -1:, :].values
	assert np.allclose(data[0][0], np.sqrt((su/s)**2 + (sv/s)**2))

	assert data[1][0].shape[0] == 11
	assert data[1][0].shape[1] == 1
	assert data[1][0].shape[2] == 60

	assert np.allclose(data[1][0],
		state_m['mass_fraction_of_precipitation_water_in_air'][10:21, :, :].values * 1e3)


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data, isentropic_moist_sedimentation_data
	#test_composite(isentropic_dry_data(), isentropic_moist_sedimentation_data())
