import numpy as np
import pytest

from tasmania.core.offline_diagnostics import \
	FakeComponent, OfflineDiagnosticComponent, RMSD, RRMSD


def test_fake_component(isentropic_dry_data):
	grid, _ = isentropic_dry_data
	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	dims_stgx = (grid.x_at_u_locations.dims[0], grid.y.dims[0], grid.z.dims[0])
	dims_stgz = (grid.x.dims[0], grid.y.dims[0], grid.z_on_interface_levels.dims[0])

	properties = {
		'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
		'air_pressure_on_interface_levels': {'dims': dims_stgz, 'units': 'hPa'},
		'x_velocity_at_u_locations': {'dims': dims_stgx, 'units': 'km hr^-1'},
	}

	fc = FakeComponent({'input_properties': properties})
	assert hasattr(fc, 'input_properties')
	assert fc.input_properties == properties

	fc = FakeComponent({'tendency_properties': properties})
	assert hasattr(fc, 'tendency_properties')
	assert fc.tendency_properties == properties

	fc = FakeComponent({'diagnostic_properties': properties})
	assert hasattr(fc, 'diagnostic_properties')
	assert fc.diagnostic_properties == properties

	fc = FakeComponent({'output_properties': properties})
	assert hasattr(fc, 'output_properties')
	assert fc.output_properties == properties


def test_rmsd(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state1, state2 = states[-1], states[0]

	fields = {
		'air_isentropic_density': 'kg m^-2 K^-1',
		'air_pressure_on_interface_levels': 'hPa',
		'x_velocity_at_u_locations': 'km hr^-1',
	}

	rmsd = RMSD(grid, fields)
	assert isinstance(rmsd, OfflineDiagnosticComponent)

	diagnostics = rmsd(state1, state2)

	assert 'rmsd_of_air_isentropic_density' in diagnostics
	assert 'rmsd_of_air_pressure_on_interface_levels' in diagnostics
	assert 'rmsd_of_x_velocity_at_u_locations' in diagnostics
	assert len(diagnostics) == 3


def test_rrmsd(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state1, state2 = states[-1], states[0]

	fields = {
		'air_isentropic_density': 'kg m^-2 K^-1',
		'air_pressure_on_interface_levels': 'hPa',
		'x_velocity_at_u_locations': 'km hr^-1',
	}

	rrmsd = RRMSD(grid, fields)
	assert isinstance(rrmsd, OfflineDiagnosticComponent)

	diagnostics = rrmsd(state1, state2)

	assert 'rrmsd_of_air_isentropic_density' in diagnostics
	assert 'rrmsd_of_air_pressure_on_interface_levels' in diagnostics
	assert 'rrmsd_of_x_velocity_at_u_locations' in diagnostics
	assert len(diagnostics) == 3


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data
	#test_fake_component(isentropic_dry_data())
	#test_rmsd(isentropic_dry_data())
