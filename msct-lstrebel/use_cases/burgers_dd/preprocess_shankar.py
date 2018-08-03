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

from dd_preprocess import DomainPreprocess


def prepare_partitioning(nx, ny, nz, sx, sy, sz, nparts, method, px=0, py=0, pz=0):
    cdomain = np.array([nx, ny, nz])
    slices = np.array([sx, sy, sz])
    periodic = np.array([px, py, pz])

    ddc = DomainPreprocess(domain=cdomain, periodic=periodic, subdivs_per_dim=slices, path=path, prefix=prefix)

    # Add Use case specific stencils:
    if method == 'forward_backward':
        ddc.add_stencil({"unow": [[1], [1], [1], [1], [0], [0]]})
        ddc.add_stencil({"vnow": [[1], [1], [1], [1], [0], [0]]})
    elif method == 'upwind':
        ddc.add_stencil({"unow": [[1], [1], [1], [1], [0], [0]]})
        ddc.add_stencil({"vnow": [[1], [1], [1], [1], [0], [0]]})
    elif method == 'upwind_third_order':
        ddc.add_stencil({"unow": [[2], [2], [2], [2], [0], [0]]})
        ddc.add_stencil({"vnow": [[2], [2], [2], [2], [0], [0]]})

    # Once all stencils are added call preprocess and partitioning
    ddc.preprocess()
    ddc.pymetis_partitioning(nparts)


def prepare_initial_condition(nx, ny, nz, dxs, dxe, dys, dye, nb):
        domain = [(dxs, dys), (dxe, dye)]
        datatype = np.float64

        # Create the grid
        x = np.linspace(domain[0][0], domain[1][0], nx - 2 * nb)
        y = np.linspace(domain[0][1], domain[1][1], ny - 2 * nb)

        # Instatiate the arrays representing the solution
        unew = np.zeros((nx, ny, 1), dtype=datatype)
        vnew = np.zeros((nx, ny, 1), dtype=datatype)

        # Set the initial conditions
        for i in range(nb, nx - nb):
            for j in range(nb, ny - nb):
                if (0.5 <= x[i - nb] and x[i - nb] <= 1.0) and (0.5 <= y[j - nb] and y[j - nb] <= 1.0):
                    unew[i, j, 0], vnew[i, j, 0] = 0.0, 1.0
                else:
                    unew[i, j, 0], vnew[i, j, 0] = 1.0, 0.0

        # Apply the boundary conditions
        unew[0, :, 0], vnew[0, :, 0] = 0., 0.
        unew[-1, :, 0], vnew[-1, :, 0] = 0., 0.
        unew[:, 0, 0], vnew[:, 0, 0] = 0., 0.
        unew[:, -1, 0], vnew[:, -1, 0] = 0., 0.

        np.save(path + prefix + "shankar_initial_conditions_unew.npy", unew)
        np.save(path + prefix + "shankar_initial_conditions_vnew.npy", vnew)

def prepare_boundary_condition(nx, ny, nz, hxm, hxp, hym, hyp, hzm, hzp):
    datatype = np.float64

    # Instatiate the arrays representing the boundary conditions
    unew = np.zeros((nx + hxm + hxp, ny + hym + hyp, nz + hzm + hzp), dtype=datatype)
    vnew = np.zeros((nx + hxm + hxp, ny + hym + hyp, nz + hzm + hzp), dtype=datatype)

    np.save(path + prefix + "shankar_boundary_conditions_unew.npy", unew)
    np.save(path + prefix + "shankar_boundary_conditions_vnew.npy", vnew)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Burger's equation - Zhao setup.")
    parser.add_argument("-nx", default=100, type=int,
                        help="Number of grid points in x direction.")
    parser.add_argument("-ny", default=100, type=int,
                        help="Number of grid points in y direction.")
    parser.add_argument("-nz", default=1, type=int,
                        help="Number of grid points in z direction.")
    parser.add_argument("-dt", default=0.001, type=float,
                        help="Time step size [s].")
    parser.add_argument("-nt", default=100, type=int,
                        help="Number of time steps.")
    parser.add_argument("-eps", default=0.01, type=float,
                        help="Viscosity constant.")
    parser.add_argument("-m", default="forward_backward", type=str,
                        help="Method: 'forward_backward', 'upwind', 'upwind_third_order'")
    parser.add_argument("-sx", default=1, type=int,
                        help="Number of divisions in x direction.")
    parser.add_argument("-sy", default=1, type=int,
                        help="Number of divisions in y direction.")
    parser.add_argument("-sz", default=1, type=int,
                        help="Number of divisions in z direction.")
    parser.add_argument("-np", default=1, type=int,
                        help="Number of partitions.")
    parser.add_argument("-loc", default="", type=str,
                        help="Path to location where files should be saved to.")
    parser.add_argument("-pf", default="", type=str,
                        help="Prefix for file names.")
    args = parser.parse_args()

    nx = args.nx
    ny = args.ny
    nz = args.nz
    dt = args.dt
    nt = args.nt
    eps = args.eps
    method = args.m

    sx = args.sx
    sy = args.sy
    sz = args.sz
    nparts = args.np

    path = args.loc
    prefix = args.pf

    domain = [(0, 0), (2, 2)]
    px = 0
    py = 0
    pz = 0

    # Set stencil's definitions function and computational domain
    nb = None
    if method == 'forward_backward':
        nb = 1
    elif method == 'upwind':
        nb = 1
    elif method == 'upwind_third_order':
        nb = 2

    prepare_partitioning(nx, ny, nz, sx, sy, sz, nparts, method, px, py, pz)
    prepare_initial_condition(nx, ny, nz, domain[0][0], domain[1][0], domain[0][1], domain[1][1], nb)
    # prepare_boundary_condition(nx, ny, nz, 1, 1, 1, 1, 0, 0)

