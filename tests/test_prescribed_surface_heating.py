from copy import deepcopy
from datetime import datetime, timedelta
import pytest
from sympl import DataArray

from tasmania.physics.isentropic_tendencies import PrescribedSurfaceHeating
from tasmania.utils.data_utils import make_data_array_3d
from tasmania.utils.utils import equal_to


def test(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	amplitude_during_daytime = DataArray(0.8, attrs={'units': 'kW m^-2'})
	amplitude_at_night = DataArray(-75000.0, attrs={'units': 'mW m^-2'})
	attenuation_coefficient_during_daytime = DataArray(1.0/6.0, attrs={'units': 'hm^-1'})
	attenuation_coefficient_at_night = DataArray(1.0/75.0, attrs={'units': 'm^-1'})
	characteristic_length = DataArray(25.0, attrs={'units': 'km'})

	#
	# tendency_of_air_potential_temperature_in_diagnostics=False
	#
	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=False,
		air_pressure_on_interface_levels=True
	)

	assert 'air_pressure_on_interface_levels' in psh.input_properties
	assert 'height_on_interface_levels' in psh.input_properties
	assert len(psh.input_properties) == 2

	assert 'air_potential_temperature_on_interface_levels' in psh.tendency_properties
	assert len(psh.tendency_properties) == 1

	assert psh.diagnostic_properties == {}

	#
	# tendency_of_air_potential_temperature_in_diagnostics=True
	#
	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=True,
		air_pressure_on_interface_levels=False
	)

	assert 'air_pressure' in psh.input_properties
	assert 'height_on_interface_levels' in psh.input_properties
	assert len(psh.input_properties) == 2

	assert psh.tendency_properties == {}

	assert 'tendency_of_air_potential_temperature' in psh.diagnostic_properties
	assert len(psh.diagnostic_properties) == 1

	#
	# air_pressure_on_interface_levels=True
	#
	state['time'] = datetime(year=1992, month=2, day=20, hour=15)
	starting_time = state['time'] - timedelta(hours=2)

	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=True,
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

	assert tendencies == {}

	assert 'tendency_of_air_potential_temperature_on_interface_levels' in diagnostics
	assert len(diagnostics) == 1

	#
	# air_pressure_on_interface_levels=False
	#
	state['time'] = datetime(year=1992, month=2, day=20, hour=3)
	starting_time = state['time'] - timedelta(hours=2)
	p = state['air_pressure_on_interface_levels'].values
	state['air_pressure'] = make_data_array_3d(0.5 * (p[:, :, :-1] + p[:, :, 1:]), grid, 'Pa')
	state.pop('air_pressure_on_interface_levels')

	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=False,
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


if __name__ == '__main__':
	pytest.main([__file__])
