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
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module contains
"""
import numpy as np
import argparse

from domain_decomposition import DomainPreprocess


def prepare_partitioning(nx, ny, nz, sx, sy, sz, nparts, px=0, py=0, pz=0):
    domain = np.array([nx, ny, nz])
    slices = np.array([sx, sy, sz])
    periodic = np.array([px, py, pz])

    ddc = DomainPreprocess(domain=domain, periodic=periodic, subdivs_per_dim=slices)

    # Add Use case specific stencils:
    # e.g. stencil_burgers_forward_backward
    ddc.add_stencil({"unow": [[1], [1], [1], [1], [0], [0]]})
    ddc.add_stencil({"vnow": [[1], [1], [1], [1], [0], [0]]})

    # Once all stencils are added call preprocess and partitioning
    ddc.preprocess()
    ddc.pymetis_partitioning(nparts)


def prepare_initial_condition(nx, ny, nz, dxs, dxe, dys, dye):
        domain = [(dxs, dys), (dxe, dye)]
        datatype = np.float64
        # Create the grid
        x = np.linspace(domain[0][0], domain[1][0], nx)
        y = np.linspace(domain[0][1], domain[1][1], ny)

        # Instatiate the arrays representing the initial conditions
        unew = np.zeros((nx, ny, nz), dtype=datatype)
        vnew = np.zeros((nx, ny, nz), dtype=datatype)

        # Set the initial conditions
        for i in range(nx):
            for j in range(ny):
                if (0.5 <= x[i] and x[i] <= 1.0) and (0.5 <= y[j] and y[j] <= 1.0):
                    unew[i, j, 0], vnew[i, j, 0] = 0.0, 1.0
                else:
                    unew[i, j, 0], vnew[i, j, 0] = 1.0, 0.0

        # Apply the boundary conditions
        unew[0, :, 0], vnew[0, :, 0] = 0., 0.
        unew[-1, :, 0], vnew[-1, :, 0] = 0., 0.
        unew[:, 0, 0], vnew[:, 0, 0] = 0., 0.
        unew[:, -1, 0], vnew[:, -1, 0] = 0., 0.

        np.save("shankar_initial_conditions_unew.npy", unew)
        np.save("shankar_initial_conditions_vnew.npy", vnew)

def prepare_boundary_condition(nx, ny, nz):
    datatype = np.float64

    # Instatiate the arrays representing the boundary conditions
    unew = np.zeros((nx, ny, nz), dtype=datatype)
    vnew = np.zeros((nx, ny, nz), dtype=datatype)

    np.save("shankar_boundary_conditions_unew.npy", unew)
    np.save("shankar_boundary_conditions_vnew.npy", vnew)


if __name__ == "__main__":
    nx = 100
    ny = 100
    nz = 1
    sx = 2
    sy = 2
    sz = 1
    nparts = 4
    px = 0
    py = 0
    pz = 0

    dxs = 0
    dxe = 2
    dys = 0
    dye = 2

    prepare_partitioning(nx, ny, nz, sx, sy, sz, nparts, px, py, pz)
    prepare_initial_condition(nx, ny, nz, dxs, dxe, dys, dye)
    prepare_boundary_condition(nx, ny, nz)

