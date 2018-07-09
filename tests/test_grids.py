import numpy as np
import pytest


def test_grid_xy():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny, 
			 dims_x=dims_x, units_x=units_x, dims_y=dims_y, units_y=units_y)

	assert g.nx == nx
	assert g.dx == (domain_x[1]-domain_x[0]) / (nx-1)
	assert g.ny == ny
	assert g.dy == (domain_y[1]-domain_y[0]) / (ny-1)

	import tasmania.namelist as nl
	assert g.x[:].dtype == nl.datatype


def test_grid_xz():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_z, nz, dims_z, units_z = [400, 300], 50, 'isentropic_density', 'K'

	from tasmania.grids.grid_xz import GridXZ as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz, 
			 dims_x=dims_x, units_x=units_x, dims_z=dims_z, units_z=units_z,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_max_height=1.e3, topo_width_x=10.e3)

	assert g.nx == nx
	assert g.dx == (domain_x[1]-domain_x[0]) / (nx-1)
	assert g.nz == nz
	assert g.dz == (domain_z[0]-domain_z[1]) / nz
	assert np.max(g.topography_height) == 0.

	g.update_topography(timedelta(seconds=45))

	assert np.max(g.topography_height) == 1.e3


def test_grid_xyz():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'
	domain_z, nz, dims_z, units_z = [400, 300], 50, 'isentropic_density', 'K'

	from tasmania.grids.grid_xyz import GridXYZ as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 dims_x=dims_x, units_x=units_x, 
			 dims_y=dims_y, units_y=units_y, 
			 dims_z=dims_z, units_z=units_z,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_max_height=1.e3, topo_width_x=10.e3, topo_width_y=5.e3)

	assert g.nx == nx
	assert g.dx == (domain_x[1]-domain_x[0]) / (nx-1)
	assert g.ny == ny
	assert g.dy == (domain_y[1]-domain_y[0]) / (ny-1)
	assert g.nz == nz
	assert g.dz == (domain_z[0]-domain_z[1]) / nz
	assert np.max(g.topography_height) == 0

	g.update_topography(timedelta(seconds=45))

	assert np.max(g.topography_height) == 1.e3


if __name__ == '__main__':
	pytest.main([__file__])
