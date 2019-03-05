# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version. 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
import pytest
from sympl import DataArray

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import conf
from tasmania.python.utils.utils import equal_to as eq


def test_grid_xy():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201

	from tasmania.python.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny)

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y.values[1]-domain_y.values[0]) / (ny-1))

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y.values[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y.values[0] + (j-0.5)*g.dy)

	assert g.x[:].dtype == conf.datatype[0]
	assert g.y[:].dtype == conf.datatype[0]


def test_grid_xz():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_z, nz = DataArray([400, 300], dims='isentropic_density', attrs={'units': 'K'}), 50

	from tasmania.python.grids.grid_xz import GridXZ as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'})})

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[0]-domain_z.values[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] - k*g.dz)

	assert np.max(g.topography_height) == 0.
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3

	assert eq(g.z_interface, 400)


def test_grid_xyz():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201
	domain_z, nz = DataArray([400, 300], dims='isentropic_density', attrs={'units': 'K'}), 50

	from tasmania.python.grids.grid_xyz import GridXYZ as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
						  'topo_width_y': DataArray(5., attrs={'units': 'km'})})

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y.values[1]-domain_y.values[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[0]-domain_z.values[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y.values[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y.values[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] - k*g.dz)

	assert np.max(g.topography_height) == 0
	g.update_topography(timedelta(seconds=45))
	assert np.max(g.topography_height) == 1.e3


def test_gal_chen_2d():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50

	from tasmania.python.grids.gal_chen import GalChen2d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'})})

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[0]-domain_z.values[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] - k*g.dz)

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
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	from tasmania.python.grids.gal_chen import GalChen2d as Grid
	from datetime import timedelta

	domain_z, nz = DataArray([100, 1], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([-2, 0], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50
	z_interface  = DataArray(200, attrs={'units': 'm'})
	try:
		_ = Grid(domain_x, nx, domain_z, nz, z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


def test_gal_chen_3d():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201
	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50

	from tasmania.python.grids.gal_chen import GalChen3d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
						  'topo_width_y': DataArray(5., attrs={'units': 'km'})})

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y.values[1]-domain_y.values[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[0]-domain_z.values[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y.values[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y.values[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] - k*g.dz)

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
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201
	from tasmania.python.grids.gal_chen import GalChen3d as Grid
	from datetime import timedelta

	domain_z, nz = DataArray([100, 1], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([-2, 0], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Gal-Chen vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50
	z_interface = DataArray(200, attrs={'units': 'm'})
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz, z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


def test_sigma_2d():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_z, nz = DataArray([0.2, 1.], attrs={'units': '1'}), 50

	from tasmania.python.grids.sigma import Sigma2d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'})},
			 dtype=np.float64)

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[1]-domain_z.values[0]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] + (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] + k*g.dz)

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
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	from tasmania.python.grids.sigma import Sigma2d as Grid
	from datetime import timedelta

	domain_z, nz = DataArray([2., 1.], attrs={'units': '1'}), 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = DataArray([-0.2, 1.], attrs={'units': '1'}), 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = DataArray([0.2, 0.4], attrs={'units': '1'}), 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = DataArray([0.2, 1.], attrs={'units': '1'}), 50
	z_interface  = DataArray(0, attrs={'units': '1'})
	try:
		_ = Grid(domain_x, nx, domain_z, nz, z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0.2, 1.0).'


def test_sigma_3d():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201
	domain_z, nz = DataArray([0.2, 1.], attrs={'units': '1'}), 50

	from tasmania.python.grids.sigma import Sigma3d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
						  'topo_width_y': DataArray(5., attrs={'units': 'km'})},
			 dtype=np.float64)

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y.values[1]-domain_y.values[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[1]-domain_z.values[0]) / nz, tol=1e-6)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y.values[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y.values[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] + (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] + k*g.dz)

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
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201
	from tasmania.python.grids.sigma import Sigma3d as Grid
	from datetime import timedelta

	domain_z, nz = DataArray([3., 1.], attrs={'units': '1'}), 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
							  'topo_width_y': DataArray(5., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = DataArray([-2., 1.], attrs={'units': '1'}), 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
							  'topo_width_y': DataArray(5., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = DataArray([0.2, 0.4], attrs={'units': '1'}), 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
							  'topo_width_y': DataArray(5., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'Pressure-based vertical coordinate should be positive, ' \
						 'equal to 1 at the surface, and decreasing with height.'

	domain_z, nz = DataArray([0.1, 1.], attrs={'units': '1'}), 50
	z_interface  = DataArray(4, attrs={'units': '1'})
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz, z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
							  'topo_width_y': DataArray(5., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0.1, 1.0).'


def test_sleve_2d():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50

	from tasmania.python.grids.sleve import SLEVE2d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'})})

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[0]-domain_z.values[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] - k*g.dz)

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
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	from tasmania.python.grids.sleve import SLEVE2d as Grid
	from datetime import timedelta

	domain_z, nz = DataArray([100, 1], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([-2, 0], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50
	z_interface  = DataArray(200, attrs={'units': 'm'})
	try:
		_ = Grid(domain_x, nx, domain_z, nz, z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


def test_sleve_3d():
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201
	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50

	from tasmania.python.grids.sleve import SLEVE3d as Grid
	from datetime import timedelta
	g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 topo_type='gaussian', topo_time=timedelta(seconds=30),
			 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
						  'topo_width_x': DataArray(10., attrs={'units': 'km'}),
						  'topo_width_y': DataArray(5., attrs={'units': 'km'})})

	assert g.nx == nx
	assert eq(g.dx, (domain_x.values[1]-domain_x.values[0]) / (nx-1))
	assert g.ny == ny
	assert eq(g.dy, (domain_y.values[1]-domain_y.values[0]) / (ny-1))
	assert g.nz == nz
	assert eq(g.dz, (domain_z.values[0]-domain_z.values[1]) / nz)

	from random import randint
	i = randint(0, nx-1)
	assert eq(g.x.values[i], domain_x.values[0] + i*g.dx)
	assert eq(g.x_at_u_locations.values[i], domain_x.values[0] + (i-0.5)*g.dx)

	j = randint(0, ny-1)
	assert eq(g.y.values[j], domain_y.values[0] + j*g.dy)
	assert eq(g.y_at_v_locations.values[j], domain_y.values[0] + (j-0.5)*g.dy)

	k = randint(0, nz-1)
	assert eq(g.z.values[k], domain_z.values[0] - (k+0.5)*g.dz)
	assert eq(g.z_on_interface_levels.values[k], domain_z.values[0] - k*g.dz)

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
	domain_x, nx = DataArray([0, 100e3], dims='x', attrs={'units': 'm'}), 101
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 201
	from tasmania.python.grids.sleve import SLEVE3d as Grid
	from datetime import timedelta

	domain_z, nz = DataArray([100, 1], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([-2, 0], attrs={'units': 'm'}), 50
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'SLEVE vertical coordinate should be positive ' \
						 'and vanish at the terrain surface.'

	domain_z, nz = DataArray([100, 0], attrs={'units': 'm'}), 50
	z_interface = DataArray(200, attrs={'units': 'm'})
	try:
		_ = Grid(domain_x, nx, domain_y, ny, domain_z, nz, z_interface=z_interface,
				 topo_type='gaussian', topo_time=timedelta(seconds=30),
				 topo_kwargs={'topo_max_height': DataArray(1., attrs={'units': 'km'}),
							  'topo_width_x': DataArray(10., attrs={'units': 'km'})})
	except ValueError as e:
		assert str(e) == 'z_interface should be in the range (0, 100).'


if __name__ == '__main__':
	pytest.main([__file__])
