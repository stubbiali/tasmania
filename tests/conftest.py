from datetime import timedelta
import pytest
from sympl import DataArray

from tasmania.grids.grid_xyz import GridXYZ as Grid
from tasmania.grids.grid_xz import GridXZ
from tasmania.utils.storage_utils import load_netcdf_dataset


@pytest.fixture(scope='module')
def grid():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 101
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 51
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type='gaussian', topo_time=timedelta(seconds=0),
				topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture(scope='module')
def grid_xz():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 101
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 1
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type='gaussian', topo_time=timedelta(seconds=0),
				topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture(scope='module')
def grid_xz_2d():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 101
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return GridXZ(domain_x, nx, domain_z, nz,
				  topo_type='gaussian', topo_time=timedelta(seconds=0),
				  topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							   'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture(scope='module')
def grid_yz():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 1
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 51
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type='gaussian', topo_time=timedelta(seconds=0),
				topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture(scope='module')
def isentropic_dry_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_dry.nc')


@pytest.fixture(scope='module')
def isentropic_moist_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_moist.nc')


@pytest.fixture(scope='module')
def isentropic_moist_sedimentation_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_moist_sedimentation.nc')


@pytest.fixture(scope='module')
def isentropic_moist_sedimentation_evaporation_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_moist_sedimentation_evaporation.nc')


@pytest.fixture(scope='module')
def physical_constants():
	return {
		'air_pressure_at_sea_level':
			DataArray(1e5, attrs={'units': 'Pa'}),
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gravitational_acceleration':
			DataArray(9.81, attrs={'units': 'm s^-2'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}
