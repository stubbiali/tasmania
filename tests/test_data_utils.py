import numpy as np
import pytest
from sympl import DataArray

from tasmania.utils import data_utils as du
from tasmania.utils.exceptions import ConstantNotFoundError


def test_get_constant():
	u = du.get_constant('gravitational_acceleration', 'm s^-2')
	assert u == 9.80665

	v = du.get_constant('foo', 'm',
						default_value=DataArray(10., attrs={'units': 'km'}))
	assert v == 10000.0

	w = du.get_constant('pippo', '1',
						default_value=DataArray(10., attrs={'units': '1'}))
	assert w == 10.

	try:
		_ = du.get_constant('foo', 'K')
	except ValueError:
		assert True

	try:
		_ = du.get_constant('bar', 'K')
	except ConstantNotFoundError:
		assert True


def test_get_physical_constants():
	d_physical_constants = {
		'gravitational_acceleration': DataArray(9.80665e-3, attrs={'units': 'km s^-2'}),
		'gas_constant_of_dry_air': DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gas_constant_of_water_vapor': DataArray(461.52, attrs={'units': 'hJ K^-1 g^-1'}),
		'latent_heat_of_vaporization_of_water':	DataArray(2.5e6, attrs={'units': 'J kg^-1'}),
		'foo_constant': DataArray(1, attrs={'units': '1'}),
	}

	physical_constants = {
		'latent_heat_of_vaporization_of_water': DataArray(1.5e3, attrs={'units': 'kJ kg^-1'}),
	}

	raw_constants = du.get_physical_constants(d_physical_constants, physical_constants)

	assert raw_constants['gravitational_acceleration'] == 9.80665e-3
	assert raw_constants['gas_constant_of_dry_air'] == 287.0
	assert raw_constants['gas_constant_of_water_vapor'] == 461.5e-5
	assert raw_constants['latent_heat_of_vaporization_of_water'] == 1.5e6
	assert raw_constants['foo_constant'] == 1.0


def test_make_data_array_xy(grid):
	grid_xy = grid.xy_grid
	nx, ny = grid_xy.nx, grid_xy.ny

	raw_array_a, units_a = np.random.rand(nx, ny), 'mm h^-1'
	raw_array_b, units_b = np.random.rand(nx+1, ny), 'm s^-1'
	raw_array_c, units_c = np.random.rand(nx, ny+1), 'm s^-1'
	raw_array_d, units_d = np.random.rand(nx+1, ny+1), '1'

	array_a = du.make_data_array_2d(raw_array_a, grid_xy, units_a)
	array_b = du.make_data_array_2d(raw_array_b, grid_xy, units_b)
	array_c = du.make_data_array_2d(raw_array_c, grid_xy, units_c)
	array_d = du.make_data_array_2d(raw_array_d, grid_xy, units_d)

	assert array_a.shape == (nx, ny)
	assert np.allclose(raw_array_a, array_a.values)
	assert array_a.dims == (grid_xy.x.dims[0], grid_xy.y.dims[0])
	assert array_a.attrs['units'] == 'mm h^-1'

	assert array_b.shape == (nx+1, ny)
	assert np.allclose(raw_array_b, array_b.values)
	assert array_b.dims == (grid_xy.x_at_u_locations.dims[0], grid_xy.y.dims[0])
	assert array_b.attrs['units'] == 'm s^-1'

	assert array_c.shape == (nx, ny+1)
	assert np.allclose(raw_array_c, array_c.values)
	assert array_c.dims == (grid_xy.x.dims[0], grid_xy.y_at_v_locations.dims[0])
	assert array_c.attrs['units'] == 'm s^-1'

	assert array_d.shape == (nx+1, ny+1)
	assert np.allclose(raw_array_d, array_d.values)
	assert array_d.dims == (grid_xy.x_at_u_locations.dims[0],
							grid_xy.y_at_v_locations.dims[0])
	assert array_d.attrs['units'] == '1'


def test_make_data_array_xz(grid_xz_2d):
	nx, nz = grid_xz_2d.nx, grid_xz_2d.nz

	raw_array_a, units_a = np.random.rand(nx, nz), 'mm h^-1'
	raw_array_b, units_b = np.random.rand(nx+1, nz), 'm s^-1'
	raw_array_c, units_c = np.random.rand(nx, nz+1), 'm s^-1'
	raw_array_d, units_d = np.random.rand(nx+1, nz+1), '1'

	array_a = du.make_data_array_2d(raw_array_a, grid_xz_2d, units_a)
	array_b = du.make_data_array_2d(raw_array_b, grid_xz_2d, units_b)
	array_c = du.make_data_array_2d(raw_array_c, grid_xz_2d, units_c)
	array_d = du.make_data_array_2d(raw_array_d, grid_xz_2d, units_d)

	assert array_a.shape == (nx, nz)
	assert np.allclose(raw_array_a, array_a.values)
	assert array_a.dims == (grid_xz_2d.x.dims[0], grid_xz_2d.z.dims[0])
	assert array_a.attrs['units'] == 'mm h^-1'

	assert array_b.shape == (nx+1, nz)
	assert np.allclose(raw_array_b, array_b.values)
	assert array_b.dims == (grid_xz_2d.x_at_u_locations.dims[0], grid_xz_2d.z.dims[0])
	assert array_b.attrs['units'] == 'm s^-1'

	assert array_c.shape == (nx, nz+1)
	assert np.allclose(raw_array_c, array_c.values)
	assert array_c.dims == (grid_xz_2d.x.dims[0], grid_xz_2d.z_on_interface_levels.dims[0])
	assert array_c.attrs['units'] == 'm s^-1'

	assert array_d.shape == (nx+1, nz+1)
	assert np.allclose(raw_array_d, array_d.values)
	assert array_d.dims == (grid_xz_2d.x_at_u_locations.dims[0],
							grid_xz_2d.z_on_interface_levels.dims[0])
	assert array_d.attrs['units'] == '1'


def test_make_data_array_3d(grid):
	nx, ny, nz = grid.nx, grid.ny, grid.nz

	raw_array_a, units_a = np.random.rand(nx, ny, nz), 'm'
	raw_array_b, units_b = np.random.rand(nx+1, ny, nz), 'm s^-2'
	raw_array_c, units_c = np.random.rand(nx, ny+1, nz), 'kg m^-3'
	raw_array_d, units_d = np.random.rand(nx, ny, nz+1), 'Pa'
	raw_array_e, units_e = np.random.rand(nx+1, ny, nz+1), 'Pa'

	array_a = du.make_data_array_3d(raw_array_a, grid, units_a)
	array_b = du.make_data_array_3d(raw_array_b, grid, units_b)
	array_c = du.make_data_array_3d(raw_array_c, grid, units_c)
	array_d = du.make_data_array_3d(raw_array_d, grid, units_d)
	array_e = du.make_data_array_3d(raw_array_e, grid, units_e)

	assert array_a.shape == (nx, ny, nz)
	assert np.allclose(raw_array_a, array_a.values)
	assert array_a.dims == (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert array_a.attrs['units'] == 'm'

	assert array_b.shape == (nx+1, ny, nz)
	assert np.allclose(raw_array_b, array_b.values)
	assert array_b.dims == (grid.x_at_u_locations.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert array_b.attrs['units'] == 'm s^-2'

	assert array_c.shape == (nx, ny+1, nz)
	assert np.allclose(raw_array_c, array_c.values)
	assert array_c.dims == (grid.x.dims[0], grid.y_at_v_locations.dims[0], grid.z.dims[0])
	assert array_c.attrs['units'] == 'kg m^-3'

	assert array_d.shape == (nx, ny, nz+1)
	assert np.allclose(raw_array_d, array_d.values)
	assert array_d.dims == (grid.x.dims[0], grid.y.dims[0],
							grid.z_on_interface_levels.dims[0])
	assert array_d.attrs['units'] == 'Pa'

	assert array_e.shape == (nx+1, ny, nz+1)
	assert np.allclose(raw_array_e, array_e.values)
	assert array_e.dims == (grid.x_at_u_locations.dims[0], grid.y.dims[0],
							grid.z_on_interface_levels.dims[0])
	assert array_e.attrs['units'] == 'Pa'

	raw_array_e[0, 0, 0] = 200
	assert np.allclose(raw_array_e, array_e.values)


if __name__ == '__main__':
	pytest.main([__file__])
