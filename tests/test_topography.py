import numpy as np
import pytest

import sympl
from tasmania.namelist import datatype as dtype


def test_topography_1d_flat():
	domain_x, nx, dims_x, units_x = [0, 100e3], 100, 'x', 'm'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	from tasmania.grids.topography import Topography1d as Topography
	topo = Topography(x, topo_type='flat_terrain')

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_1d_gaussian():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	from tasmania.grids.topography import Topography1d as Topography
	topo = Topography(x, topo_type='gaussian', topo_max_height=1.e3, topo_width_x=10e3)

	topo_ref = 1.e3 * np.exp(-((xv-50e3) / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_1d_update():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	from tasmania.grids.topography import Topography1d as Topography
	from datetime import timedelta
	topo = Topography(x, topo_type='gaussian', topo_time=timedelta(seconds=60), 
					  topo_max_height=1.e3, topo_width_x=10e3)

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 60.
	assert topo.topo_fact == 0.

	topo.update(timedelta(seconds=30))

	topo_ref = 500. * np.exp(-((xv-50e3) / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 0.5

	topo.update(timedelta(seconds=60))

	topo_ref = 1.e3 * np.exp(-((xv-50e3) / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.

	topo.update(timedelta(seconds=90))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.


def test_topography_2d_flat():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny, 
			 dims_x=dims_x, units_x=units_x, dims_y=dims_y, units_y=units_y)

	from tasmania.grids.topography import Topography2d as Topography
	topo = Topography(g, topo_type='flat_terrain')

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_2d_gaussian():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny, 
			 dims_x=dims_x, units_x=units_x, dims_y=dims_y, units_y=units_y)

	from tasmania.grids.topography import Topography2d as Topography
	topo = Topography(g, topo_type='gaussian', topo_max_height=1.e3,
					  topo_width_x=10e3, topo_width_y=10e3)

	xv_ = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		  else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	yv_ = 1000. * np.array([0.5 * (domain_y[0]+domain_y[1])], dtype=dtype) if ny == 1 \
		  else 1000. * np.linspace(domain_y[0], domain_y[1], ny, dtype=dtype)
	xv = np.repeat(xv_[:, np.newaxis], 201, axis=1)
	yv = np.repeat(yv_[np.newaxis, :], 101, axis=0)
	topo_ref = 1.e3 * np.exp(- ((xv-50e3) / 10.e3)**2 - (yv / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_2d_schaer():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny, 
			 dims_x=dims_x, units_x=units_x, dims_y=dims_y, units_y=units_y)

	from tasmania.grids.topography import Topography2d as Topography
	topo = Topography(g, topo_type='gaussian', topo_max_height=1.e3, topo_width_x=10e3, topo_width_y=10e3)

	xv_ = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		  else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	yv_ = 1000. * np.array([0.5 * (domain_y[0]+domain_y[1])], dtype=dtype) if ny == 1 \
		  else 1000. * np.linspace(domain_y[0], domain_y[1], ny, dtype=dtype)
	xv = np.repeat(xv_[:, np.newaxis], 201, axis=1)
	yv = np.repeat(yv_[np.newaxis, :], 101, axis=0)
	topo_ref = 1.e3 * np.exp(- ((xv-50e3) / 10.e3)**2 - (yv / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_2d_update():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny, 
			 dims_x=dims_x, units_x=units_x, dims_y=dims_y, units_y=units_y)

	from tasmania.grids.topography import Topography2d as Topography
	from datetime import timedelta
	topo = Topography(g, topo_type='gaussian', topo_time=timedelta(seconds=60),
					  topo_max_height=1.e3, topo_width_x=10e3, topo_width_y=10e3)

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 60.
	assert topo.topo_fact == 0.

	topo.update(timedelta(seconds=30))

	xv_ = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		  else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	yv_ = 1000. * np.array([0.5 * (domain_y[0]+domain_y[1])], dtype=dtype) if ny == 1 \
		  else 1000. * np.linspace(domain_y[0], domain_y[1], ny, dtype=dtype)
	xv = np.repeat(xv_[:, np.newaxis], 201, axis=1)
	yv = np.repeat(yv_[np.newaxis, :], 101, axis=0)
	topo_ref = 500. * np.exp(- ((xv-50e3) / 10.e3)**2 - (yv / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 0.5

	topo.update(timedelta(seconds=60))
	topo_ref = 1.e3 * np.exp(- ((xv-50e3) / 10.e3)**2 - (yv / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.

	topo.update(timedelta(seconds=90))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.


if __name__ == '__main__':
	pytest.main([__file__])
