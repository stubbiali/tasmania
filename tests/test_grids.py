import numpy as np
import pytest

from tasmania.utils.utils import equal_to as eq


def test_grid_xy():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny, 
			 dims_x=dims_x, units_x=units_x, dims_y=dims_y, units_y=units_y)

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y[1]-domain_y[0]) / (ny-1))

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y[0] + (j-0.5)*g.dy)

	import tasmania.namelist as nl
	assert g.x[:].dtype == nl.datatype
	assert g.y[:].dtype == nl.datatype


def test_grid_xz():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_z, nz, dims_z, units_z = [400, 300], 50, 'isentropic_density', 'K'

	from tasmania.grids.grid_xz import GridXZ as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 dims_x=dims_x, units_x=units_x, dims_z=dims_z, units_z=units_z,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[0]-domain_z[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] - k*g.dz)

	assert np.max(g.topography_height) == 0.
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	assert eq(g.z_interface, 400)


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
			 topo_kwargs={'topo_max_height': 1.e3,
						  'topo_width_x': 10.e3, 'topo_width_y': 5.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y[1]-domain_y[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[0]-domain_z[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] - k*g.dz)

	assert np.max(g.topography_height) == 0
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3


def test_gal_chen_2d():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_z, nz = [100, 0], 50

	from tasmania.grids.gal_chen import GalChen2d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 dims_x=dims_x, units_x=units_x,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[0]-domain_z[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] - k*g.dz)

	assert np.max(g.topography_height) == 0.
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	import copy
	h	 = copy.deepcopy(g.height)
	h_il = copy.deepcopy(g.height_on_interface_levels)
	p	 = copy.deepcopy(g.reference_pressure)
	p_il = copy.deepcopy(g.reference_pressure_on_interface_levels)
	g.update_topography(timedelta(seconds=60))
	assert np.allclose(h.values, g.height.values)
	assert np.allclose(h_il.values, g.height_on_interface_levels.values)
	assert np.allclose(p.values, g.reference_pressure.values)
	assert np.allclose(p_il.values, g.reference_pressure_on_interface_levels.values)


def test_gal_chen_2d_exceptions():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	from tasmania.grids.gal_chen import GalChen2d as Grid
	from datetime import timedelta

	domain_z, nz = [100, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [-2, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [100, 0], 50
	z_interface  = 200
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


def test_gal_chen_3d():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'
	domain_z, nz = [100, 0], 50

	from tasmania.grids.gal_chen import GalChen3d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 dims_x=dims_x, units_x=units_x,
			 dims_y=dims_y, units_y=units_y,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': 1.e3,
						  'topo_width_x': 10.e3, 'topo_width_y': 5.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y[1]-domain_y[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[0]-domain_z[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] - k*g.dz)

	assert np.max(g.topography_height) == 0
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	import copy
	h	 = copy.deepcopy(g.height)
	h_il = copy.deepcopy(g.height_on_interface_levels)
	p	 = copy.deepcopy(g.reference_pressure)
	p_il = copy.deepcopy(g.reference_pressure_on_interface_levels)
	g.update_topography(timedelta(seconds=60))
	assert np.allclose(h.values, g.height.values)
	assert np.allclose(h_il.values, g.height_on_interface_levels.values)
	assert np.allclose(p.values, g.reference_pressure.values)
	assert np.allclose(p_il.values, g.reference_pressure_on_interface_levels.values)


def test_gal_chen_3d_exceptions():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [0, 100e3], 101, 'y', 'm'
	from tasmania.grids.gal_chen import GalChen3d as Grid
	from datetime import timedelta

	domain_z, nz = [100, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [-2, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [100, 0], 50
	z_interface  = 200
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


def test_sigma_2d():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_z, nz = [0.2, 1.], 50

	from tasmania.grids.sigma import Sigma2d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 dims_x=dims_x, units_x=units_x, dtype=float,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[1]-domain_z[0]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] + (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] + k*g.dz)

	assert np.max(g.topography_height) == 0.
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	import copy
	h	 = copy.deepcopy(g.height)
	h_il = copy.deepcopy(g.height_on_interface_levels)
	p	 = copy.deepcopy(g.reference_pressure)
	p_il = copy.deepcopy(g.reference_pressure_on_interface_levels)
	g.update_topography(timedelta(seconds=60))
	assert np.allclose(h.values, g.height.values)
	assert np.allclose(h_il.values, g.height_on_interface_levels.values)
	assert np.allclose(p.values, g.reference_pressure.values)
	assert np.allclose(p_il.values, g.reference_pressure_on_interface_levels.values)

	assert eq(g.z_interface, 0.2)


def test_sigma_2d_exceptions():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	from tasmania.grids.sigma import Sigma2d as Grid
	from datetime import timedelta

	domain_z, nz = [2., 1.], 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = [-0.2, 1.], 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = [0.2, 0.4], 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = [0.2, 1.], 50
	z_interface  = 0
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0.2, 1.0).'


def test_sigma_3d():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50., 50.], 201, 'y', 'km'
	domain_z, nz = [1e-6, 1.], 50

	from tasmania.grids.sigma import Sigma3d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 dims_x=dims_x, units_x=units_x,
			 dims_y=dims_y, units_y=units_y, dtype=float,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': 1.e3,
						  'topo_width_x': 10.e3, 'topo_width_y': 5.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y[1]-domain_y[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[1]-domain_z[0]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] + (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] + k*g.dz)

	assert np.max(g.topography_height) == 0
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	import copy
	h	 = copy.deepcopy(g.height)
	h_il = copy.deepcopy(g.height_on_interface_levels)
	p	 = copy.deepcopy(g.reference_pressure)
	p_il = copy.deepcopy(g.reference_pressure_on_interface_levels)
	g.update_topography(timedelta(seconds=60))
	assert np.allclose(h.values, g.height.values)
	assert np.allclose(h_il.values, g.height_on_interface_levels.values)
	assert np.allclose(p.values, g.reference_pressure.values)
	assert np.allclose(p_il.values, g.reference_pressure_on_interface_levels.values)


def test_sigma_3d_exceptions():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [0, 100e3], 101, 'y', 'm'
	from tasmania.grids.sigma import Sigma3d as Grid
	from datetime import timedelta

	domain_z, nz = [3., 1.], 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 topo_type='gaussian',
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = [-2, 1.], 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = [0.2, 0.4], 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = [0.1, 1.], 50
	z_interface  = 4
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0.1, 1.0).'


def test_sleve_2d():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_z, nz = [100, 0], 50

	from tasmania.grids.sleve import SLEVE2d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 dims_x=dims_x, units_x=units_x,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[0]-domain_z[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] - k*g.dz)

	assert np.max(g.topography_height) == 0.
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	import copy
	h	 = copy.deepcopy(g.height)
	h_il = copy.deepcopy(g.height_on_interface_levels)
	p	 = copy.deepcopy(g.reference_pressure)
	p_il = copy.deepcopy(g.reference_pressure_on_interface_levels)
	g.update_topography(timedelta(seconds=60))
	assert np.allclose(h.values, g.height.values)
	assert np.allclose(h_il.values, g.height_on_interface_levels.values)
	assert np.allclose(p.values, g.reference_pressure.values)
	assert np.allclose(p_il.values, g.reference_pressure_on_interface_levels.values)


def test_sleve_2d_exceptions():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	from tasmania.grids.sleve import SLEVE2d as Grid
	from datetime import timedelta

	domain_z, nz = [100, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [-2, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [100, 0], 50
	z_interface  = 200
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


def test_sleve_3d():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [-50, 50], 201, 'y', 'km'
	domain_z, nz = [100, 0], 50

	from tasmania.grids.sleve import SLEVE3d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 dims_x=dims_x, units_x=units_x,
			 dims_y=dims_y, units_y=units_y,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': 1.e3,
						  'topo_width_x': 10.e3, 'topo_width_y': 5.e3})

	assert g.nx == nx
	assert eq(g.dx, (domain_x[1]-domain_x[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y[1]-domain_y[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z[0]-domain_z[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z[0] - k*g.dz)

	assert np.max(g.topography_height) == 0
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	import copy
	h	 = copy.deepcopy(g.height)
	h_il = copy.deepcopy(g.height_on_interface_levels)
	p	 = copy.deepcopy(g.reference_pressure)
	p_il = copy.deepcopy(g.reference_pressure_on_interface_levels)
	g.update_topography(timedelta(seconds=60))
	assert np.allclose(h.values, g.height.values)
	assert np.allclose(h_il.values, g.height_on_interface_levels.values)
	assert np.allclose(p.values, g.reference_pressure.values)
	assert np.allclose(p_il.values, g.reference_pressure_on_interface_levels.values)


def test_sleve_3d_exceptions():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'
	domain_y, ny, dims_y, units_y = [0, 100e3], 101, 'y', 'm'
	from tasmania.grids.sleve import SLEVE3d as Grid
	from datetime import timedelta

	domain_z, nz = [100, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [-2, 1], 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = [100, 0], 50
	z_interface  = 200
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 dims_x=dims_x, units_x=units_x,
				 dims_y=dims_y, units_y=units_y,
				 z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': 1.e3, 'topo_width_x': 10.e3})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


if __name__ == '__main__':
	pytest.main([__file__])
