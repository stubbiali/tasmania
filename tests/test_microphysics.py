import pytest


def test_kessler(isentropic_moist_sedimentation_evaporation_data):
	from tasmania.physics.microphysics import Kessler
	grid, states = isentropic_moist_sedimentation_evaporation_data
	state = states[-1]

	kp = Kessler(grid)

	assert kp._a == 0.001
	assert kp._k1 == 0.001
	assert kp._k2 == 2.2

	assert hasattr(kp, 'input_properties')
	assert hasattr(kp, 'tendency_properties')
	assert hasattr(kp, 'diagnostic_properties')

	tendencies, diagnostics = kp(state)

	tnd_name_1 = 'tendency_of_air_potential_temperature'
	tnd_name_2 = 'tendency_of_mass_fraction_of_water_vapor_in_air'
	tnd_name_3 = 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air'
	tnd_name_4 = 'tendency_of_mass_fraction_of_precipitation_water_in_air'

	assert tnd_name_1 in tendencies.keys()
	assert tnd_name_2 in tendencies.keys()
	assert tnd_name_3 in tendencies.keys()
	assert tnd_name_4 in tendencies.keys()

	assert tendencies[tnd_name_1].attrs['units'] == 'K s^-1'
	assert tendencies[tnd_name_2].attrs['units'] == 'g g^-1 s^-1'
	assert tendencies[tnd_name_3].attrs['units'] == 'g g^-1 s^-1'
	assert tendencies[tnd_name_4].attrs['units'] == 'g g^-1 s^-1'

	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert tendencies[tnd_name_1].dims == dims
	assert tendencies[tnd_name_2].dims == dims
	assert tendencies[tnd_name_3].dims == dims
	assert tendencies[tnd_name_4].dims == dims

	assert diagnostics == {}


def test_saturation_adjustment_kessler(isentropic_moist_sedimentation_evaporation_data):
	from tasmania.physics.microphysics import SaturationAdjustmentKessler
	grid, states = isentropic_moist_sedimentation_evaporation_data
	state = states[-1]

	sakp = SaturationAdjustmentKessler(grid)

	assert hasattr(sakp, 'input_properties')
	assert hasattr(sakp, 'diagnostic_properties')

	diagnostics = sakp(state)

	diag_name_1 = 'mass_fraction_of_water_vapor_in_air'
	diag_name_2 = 'mass_fraction_of_cloud_liquid_water_in_air'

	assert diag_name_1 in diagnostics.keys()
	assert diag_name_2 in diagnostics.keys()

	assert diagnostics[diag_name_1].attrs['units'] == 'g g^-1'
	assert diagnostics[diag_name_2].attrs['units'] == 'g g^-1'

	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert diagnostics[diag_name_1].dims == dims
	assert diagnostics[diag_name_2].dims == dims


def test_raindrop_fall_velocity(isentropic_moist_sedimentation_evaporation_data):
	from tasmania.physics.microphysics import RaindropFallVelocity
	grid, states = isentropic_moist_sedimentation_evaporation_data
	state = states[-1]

	rfvp = RaindropFallVelocity(grid)

	assert hasattr(rfvp, 'input_properties')
	assert hasattr(rfvp, 'diagnostic_properties')

	diagnostics = rfvp(state)

	diag_name = 'raindrop_fall_velocity'

	assert diag_name in diagnostics.keys()

	assert diagnostics[diag_name].attrs['units'] == 'm s^-1'

	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert diagnostics[diag_name].dims == dims


if __name__ == '__main__':
	pytest.main([__file__])
