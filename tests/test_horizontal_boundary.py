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

from tasmania.dynamics.horizontal_boundary import HorizontalBoundary as HB


def test_periodic(grid):
	nx, ny, nz = grid.nx, grid.ny, grid.nz

	bnd = HB.factory('periodic', grid, 2)

	phi = np.random.rand(nx, ny, nz)
	phi[-1, :, :] = phi[0, :, :]
	phi[:, -1, :] = phi[:, 0, :]

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx + 4
	assert phi_.shape[1] == ny + 4
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_[2:-2, 2:-2, :])
	assert np.allclose(phi_[0, 2:-2, :], phi[-3, :, :])
	assert np.allclose(phi_[1, 2:-2, :], phi[-2, :, :])
	assert np.allclose(phi_[-2, 2:-2, :], phi[1, :, :])
	assert np.allclose(phi_[-1, 2:-2, :], phi[2, :, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx
	assert phi_new.shape[1] == ny
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	phi = np.random.rand(nx+1, ny+1, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx + 5
	assert phi_.shape[1] == ny + 5
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_[2:-2, 2:-2, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx+1
	assert phi_new.shape[1] == ny+1
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new)
	assert np.allclose(phi_new[0, :, :], phi_new[-2, :, :])
	assert np.allclose(phi_new[1, :, :], phi_new[-1, :, :])
	assert np.allclose(phi_new[:, 0, :], phi_new[:, -2, :])
	assert np.allclose(phi_new[:, 1, :], phi_new[:, -1, :])

	bnd.set_outermost_layers_x(phi_new)

	bnd.set_outermost_layers_y(phi_new)

	cgrid = bnd.get_computational_grid()
	assert cgrid.nx == nx+4
	assert cgrid.ny == ny+4
	assert cgrid.nz == nz


def test_periodic_xz(grid_xz):
	nx, ny, nz = grid_xz.nx, grid_xz.ny, grid_xz.nz

	bnd = HB.factory('periodic', grid_xz, 2)

	phi = np.random.rand(nx, ny, nz)
	phi[-1, :, :] = phi[0, :, :]

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx + 4
	assert phi_.shape[1] == ny + 4
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_[2:-2, 2:-2, :])
	assert np.allclose(phi_[0, 2:-2, :], phi[-3, :, :])
	assert np.allclose(phi_[1, 2:-2, :], phi[-2, :, :])
	assert np.allclose(phi_[-2, 2:-2, :], phi[1, :, :])
	assert np.allclose(phi_[-1, 2:-2, :], phi[2, :, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx
	assert phi_new.shape[1] == ny
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	phi = np.random.rand(nx+1, ny+1, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx + 5
	assert phi_.shape[1] == ny + 5
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_[2:-2, 2:-2, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx+1
	assert phi_new.shape[1] == ny+1
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new)
	assert np.allclose(phi_new[0, :, :], phi_new[-2, :, :])
	assert np.allclose(phi_new[1, :, :], phi_new[-1, :, :])

	bnd.set_outermost_layers_x(phi_new)

	bnd.set_outermost_layers_y(phi_new)

	cgrid = bnd.get_computational_grid()
	assert cgrid.nx == nx+4
	assert cgrid.ny == ny+4
	assert cgrid.nz == nz


def test_periodic_yz(grid_yz):
	nx, ny, nz = grid_yz.nx, grid_yz.ny, grid_yz.nz

	bnd = HB.factory('periodic', grid_yz, 2)

	phi = np.random.rand(nx, ny, nz)
	phi[-1, :, :] = phi[0, :, :]

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx + 4
	assert phi_.shape[1] == ny + 4
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_[2:-2, 2:-2, :])
	assert np.allclose(phi_[2:-2,  0, :], phi[:, -3, :])
	assert np.allclose(phi_[2:-2,  1, :], phi[:, -2, :])
	assert np.allclose(phi_[2:-2, -2, :], phi[:,  1, :])
	assert np.allclose(phi_[2:-2, -1, :], phi[:,  2, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx
	assert phi_new.shape[1] == ny
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	phi = np.random.rand(nx+1, ny+1, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx + 5
	assert phi_.shape[1] == ny + 5
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_[2:-2, 2:-2, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx+1
	assert phi_new.shape[1] == ny+1
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new)
	assert np.allclose(phi_new[:, 0, :], phi_new[:, -2, :])
	assert np.allclose(phi_new[:, 1, :], phi_new[:, -1, :])

	bnd.set_outermost_layers_x(phi_new)

	bnd.set_outermost_layers_y(phi_new)

	cgrid = bnd.get_computational_grid()
	assert cgrid.nx == nx+4
	assert cgrid.ny == ny+4
	assert cgrid.nz == nz


def test_relaxed(grid):
	nx, ny, nz = grid.nx, grid.ny, grid.nz

	bnd = HB.factory('relaxed', grid, 2)

	phi = np.random.rand(nx, ny, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx
	assert phi_.shape[1] == ny
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_)

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx
	assert phi_new.shape[1] == ny
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new, phi)
	assert np.allclose(phi_new[bnd.nr:-bnd.nr, bnd.nr:-bnd.nr, :],
					   phi[bnd.nr:-bnd.nr, bnd.nr:-bnd.nr, :])

	phi = np.random.rand(nx+1, ny+1, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx+1
	assert phi_.shape[1] == ny+1
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi, phi_)

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx+1
	assert phi_new.shape[1] == ny+1
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new, phi)
	assert np.allclose(phi[bnd.nr:-bnd.nr, bnd.nr:-bnd.nr, :],
					   phi_new[bnd.nr:-bnd.nr, bnd.nr:-bnd.nr, :])

	bnd.set_outermost_layers_x(phi_new, phi)
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])

	bnd.set_outermost_layers_y(phi_new, phi)
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])

	cgrid = bnd.get_computational_grid()
	assert cgrid.nx == nx
	assert cgrid.ny == ny
	assert cgrid.nz == nz


def test_relaxed_xz(grid_xz):
	nx, ny, nz = grid_xz.nx, grid_xz.ny, grid_xz.nz

	bnd = HB.factory('relaxed', grid_xz, 2)

	phi = np.random.rand(nx, ny, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx
	assert phi_.shape[1] == ny+4
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi[:, 0, :], phi_[:, 0, :])
	assert np.allclose(phi[:, 0, :], phi_[:, 1, :])
	assert np.allclose(phi[:, 0, :], phi_[:, 2, :])
	assert np.allclose(phi[:, 0, :], phi_[:, 3, :])
	assert np.allclose(phi[:, 0, :], phi_[:, 4, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx
	assert phi_new.shape[1] == ny
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new, phi)
	assert np.allclose(phi_new[bnd.nr:-bnd.nr, :, :], phi[bnd.nr:-bnd.nr, :, :])

	phi = np.random.rand(nx+1, ny+1, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx+1
	assert phi_.shape[1] == ny+5
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi[:, 0, :], phi_[:, 0, :])
	assert np.allclose(phi[:, 0, :], phi_[:, 1, :])
	assert np.allclose(phi[:, 0, :], phi_[:, 2, :])
	assert np.allclose(phi[:, 1, :], phi_[:, 3, :])
	assert np.allclose(phi[:, 1, :], phi_[:, 4, :])
	assert np.allclose(phi[:, 1, :], phi_[:, 5, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx+1
	assert phi_new.shape[1] == ny+1
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new, phi)
	assert np.allclose(phi[bnd.nr:-bnd.nr, :, :], phi_new[bnd.nr:-bnd.nr, :, :])

	bnd.set_outermost_layers_x(phi_new, phi)
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])

	cgrid = bnd.get_computational_grid()
	assert cgrid.nx == nx
	assert cgrid.ny == ny+4
	assert cgrid.nz == nz


def test_relaxed_yz(grid_yz):
	nx, ny, nz = grid_yz.nx, grid_yz.ny, grid_yz.nz

	bnd = HB.factory('relaxed', grid_yz, 2)

	phi = np.random.rand(nx, ny, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx+4
	assert phi_.shape[1] == ny
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi[0, :, :], phi_[0, :, :])
	assert np.allclose(phi[0, :, :], phi_[1, :, :])
	assert np.allclose(phi[0, :, :], phi_[2, :, :])
	assert np.allclose(phi[0, :, :], phi_[3, :, :])
	assert np.allclose(phi[0, :, :], phi_[4, :, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx
	assert phi_new.shape[1] == ny
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new, phi)
	assert np.allclose(phi_new[:, bnd.nr:-bnd.nr, :], phi[:, bnd.nr:-bnd.nr, :])

	phi = np.random.rand(nx+1, ny+1, nz)

	phi_ = bnd.from_physical_to_computational_domain(phi)
	assert phi_.shape[0] == nx+5
	assert phi_.shape[1] == ny+1
	assert phi_.shape[2] == nz
	assert phi_.dtype == phi.dtype
	assert np.allclose(phi[0, :, :], phi_[0, :, :])
	assert np.allclose(phi[0, :, :], phi_[1, :, :])
	assert np.allclose(phi[0, :, :], phi_[2, :, :])
	assert np.allclose(phi[1, :, :], phi_[3, :, :])
	assert np.allclose(phi[1, :, :], phi_[4, :, :])
	assert np.allclose(phi[1, :, :], phi_[5, :, :])

	phi_new = bnd.from_computational_to_physical_domain(phi_)
	assert phi_new.shape[0] == nx+1
	assert phi_new.shape[1] == ny+1
	assert phi_new.shape[2] == nz
	assert phi_new.dtype == phi.dtype
	assert np.allclose(phi, phi_new)

	bnd.enforce(phi_new, phi)
	assert np.allclose(phi[:, bnd.nr:-bnd.nr, :], phi_new[:, bnd.nr:-bnd.nr, :])

	bnd.set_outermost_layers_y(phi_new, phi)
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])

	cgrid = bnd.get_computational_grid()
	assert cgrid.nx == nx+4
	assert cgrid.ny == ny
	assert cgrid.nz == nz


def test_relaxed_symmetric_xz(grid):
	# TODO
	pass


def test_relaxed_symmetric_yz(grid):
	# TODO
	pass


if __name__ == '__main__':
	pytest.main([__file__])
