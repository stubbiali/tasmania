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
from mpi4py import MPI

import gridtools as gt
import stencils
import timer as ti

from domain_decomposition import DomainDecomposition
from dd_postprocess import DomainPostprocess


def run_zhao():
    timer.start(name="Overall Zhao time", level=1)
    timer.start(name="Initialization", level=2)

    # Set stencil's definitions function and computational domain
    if method == 'forward_backward':
        definitions_func_ = stencils.stencil_burgers_forward_backward
        halo = [1, 1, 1, 1, 0, 0]
        nb = 1
    elif method == 'upwind':
        definitions_func_ = stencils.stencil_burgers_upwind
        halo = [1, 1, 1, 1, 0, 0]
        nb = 1
    elif method == 'upwind_third_order':
        definitions_func_ = stencils.stencil_burgers_upwind_third_order
        halo = [2, 2, 2, 2, 0, 0]
        nb = 2

    # Driver
    #
    # # Infer the grid size
    dx = float(domain[1][0] - domain[0][0]) / nx
    dy = float(domain[1][1] - domain[0][1]) / ny
    #
    # # Create the grid
    # x = np.linspace(domain[0][0], domain[1][0], nx)
    # xv = np.repeat(x[:, np.newaxis], ny, axis=1)
    # y = np.linspace(domain[0][1], domain[1][1], ny)
    # yv = np.repeat(y[np.newaxis, :], nx, axis=0)

    # unew_east = np.zeros((nb, ny, nz))
    # unew_west = np.zeros((nb, ny, nz))
    # unew_north = np.zeros((nx, nb, nz))
    # unew_south = np.zeros((nx, nb, nz))
    #
    # vnew_east = np.zeros((nb, ny, nz))
    # vnew_west = np.zeros((nb, ny, nz))
    # vnew_north = np.zeros((nx, nb, nz))
    # vnew_south = np.zeros((nx, nb, nz))

    # Register fields and stencils to the DomainDecomposition class:
    prepared_domain = DomainDecomposition("subdomains_pymetis.dat.part." + str(nparts), "metis",
                                          path=path, prefix=prefix, comm_onesided=onesided)

    prepared_domain.register_field(fieldname="unow",
                                   halo=halo,
                                   field_ic_file=path + prefix + "zhao_initial_conditions_unew",
                                   singlefile=False)
                                   # field_bc_file=path + prefix + "zhao_boundary_conditions_unew.npy")
    prepared_domain.register_field(fieldname="vnow",
                                   halo=halo,
                                   field_ic_file=path + prefix + "zhao_initial_conditions_vnew",
                                   singlefile=False)
                                   # field_bc_file=path + prefix + "zhao_boundary_conditions_vnew.npy")

    prepared_domain.register_field(fieldname="unew",
                                   halo=halo,
                                   field_ic_file=path + prefix + "zhao_initial_conditions_unew",
                                   singlefile=False)
                                   # field_bc_file=path + prefix + "zhao_boundary_conditions_unew.npy")
    prepared_domain.register_field(fieldname="vnew",
                                   halo=halo,
                                   field_ic_file=path + prefix + "zhao_initial_conditions_vnew",
                                   singlefile=False)
                                   # field_bc_file=path + prefix + "zhao_boundary_conditions_vnew.npy")

    prepared_domain.register_field(fieldname="xv",
                                   halo=halo,
                                   field_ic_file=path + prefix + "zhao_initial_conditions_xv",
                                   singlefile=False)

    prepared_domain.register_field(fieldname="yv",
                                   halo=halo,
                                   field_ic_file=path + prefix + "zhao_initial_conditions_yv",
                                   singlefile=False)


    # Convert global inputs to GT4Py Global's
    dt_ = gt.Global(dt)
    dx_ = gt.Global(dx)
    dy_ = gt.Global(dy)
    eps_ = gt.Global(eps)

    stencil = prepared_domain.register_stencil(
        definitions_func=definitions_func_,
        inputs={'in_u': "unow", 'in_v': "vnow"},
        global_inputs={'dt': dt_, 'dx': dx_, 'dy': dy_, 'eps': eps_},
        outputs={'out_u': "unew", 'out_v': "vnew"},
        mode=gt.mode.NUMPY
    )

    timer.stop(name="Initialization")
    timer.start(name="Time integration", level=2)

    if save_freq >= 0:
        prepared_domain.save_fields(["unew", "vnew"], postfix="t_" + str(0))

    # Time integration
    for n in range(nt):
        # Advance the time levels
        prepared_domain.swap_fields("unow", "unew")
        prepared_domain.swap_fields("vnow", "vnew")

        # Apply the boundary conditions
        # Set the boundaries
        t = (n + 1) * float(dt)
        unew_west = {}
        unew_east = {}
        unew_north = {}
        unew_south = {}
        vnew_west = {}
        vnew_east = {}
        vnew_north = {}
        vnew_south = {}
        for sd in prepared_domain.subdivisions:
            unew_west[sd] = (- 2. * eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t)
                                      * np.sin(np.pi * sd.get_interior_field("yv")[:nb, :, 0]))
            unew_east[sd] = (- 2. * eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t)
                                      * np.sin(np.pi * sd.get_interior_field("yv")[-nb:, :, 0]))
            unew_north[sd] = np.zeros((sd.size[0], nb))
            unew_south[sd] = np.zeros((sd.size[0], nb))
            vnew_west[sd] = np.zeros((nb, sd.size[1]))
            vnew_east[sd] = np.zeros((nb, sd.size[1]))
            vnew_north[sd] = (- eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t)
                                       * np.sin(2. * np.pi * sd.get_interior_field("xv")[:, :nb, 0]))
            vnew_south[sd] = (eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t)
                                       * np.sin(2. * np.pi * sd.get_interior_field("xv")[:, -nb:, 0]))

            unew_west[sd] = unew_west[sd].reshape((halo[0], sd.size[1], sd.size[2]))
            unew_east[sd] = unew_east[sd].reshape((halo[1], sd.size[1], sd.size[2]))
            unew_north[sd] = unew_north[sd].reshape((sd.size[0], halo[2], sd.size[2]))
            unew_south[sd] = unew_south[sd].reshape((sd.size[0], halo[3], sd.size[2]))

            vnew_west[sd] = vnew_west[sd].reshape((halo[0], sd.size[1], sd.size[2]))
            vnew_east[sd] = vnew_east[sd].reshape((halo[1], sd.size[1], sd.size[2]))
            vnew_north[sd] = vnew_north[sd].reshape((sd.size[0], halo[2], sd.size[2]))
            vnew_south[sd] = vnew_south[sd].reshape((sd.size[0], halo[3], sd.size[2]))

            sd.set_boundary_condition("unow", 0, unew_west[sd])
            sd.set_boundary_condition("unow", 1, unew_east[sd])
            sd.set_boundary_condition("unow", 2, unew_north[sd])
            sd.set_boundary_condition("unow", 3, unew_south[sd])

            sd.set_boundary_condition("vnow", 0, vnew_west[sd])
            sd.set_boundary_condition("vnow", 1, vnew_east[sd])
            sd.set_boundary_condition("vnow", 2, vnew_north[sd])
            sd.set_boundary_condition("vnow", 3, vnew_south[sd])

        prepared_domain.apply_boundary_condition("unow")
        prepared_domain.apply_boundary_condition("vnow")

        # Step the solution
        timer.start(name="Compute during time integration", level=3)
        stencil.compute()
        timer.stop(name="Compute during time integration")

        # Communicate partition boundaries
        timer.start(name="Communication during time integration", level=3)
        prepared_domain.communicate("unew")
        prepared_domain.communicate("vnew")
        timer.stop(name="Communication during time integration")

        timer.start(name="Saving fields during time integration", level=3)
        # Save
        if ((save_freq > 0) and (n % save_freq == 0)):
            prepared_domain.save_fields(["unew", "vnew"], postfix="t_"+str(n))
        elif (save_freq >= 0 and (n + 1 == nt)):
            prepared_domain.save_fields(["unew", "vnew"], postfix="t_"+str(n + 1))
        timer.stop(name="Saving fields during time integration")

    timer.stop(name="Time integration")

    timer.stop(name="Overall Zhao time")



def postprocess_zhao():
    postproc = DomainPostprocess(path=path, prefix=prefix)
    postproc.create_pickle_dump(nx, ny, nz, nt, domain, dt, eps, save_freq, filename, cleanup=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Burger's equation - Zhao setup.")

    parser.add_argument("-nx", default=100, type=int,
                        help="Number of grid points in x direction.")
    parser.add_argument("-ny", default=100, type=int,
                        help="Number of grid points in y direction.")
    parser.add_argument("-dt", default=0.001, type=float,
                        help="Time step size [s].")
    parser.add_argument("-nt", default=100, type=int,
                        help="Number of time steps.")
    parser.add_argument("-eps", default=0.01, type=float,
                        help="Viscosity constant.")
    parser.add_argument("-m", default="forward_backward", type=str,
                        help="Method: 'forward_backward', 'upwind', 'upwind_third_order'")
    parser.add_argument("-sf", default=0, type=int,
                        help="Save frequency: Number of time steps between fields are saved to file.")
    parser.add_argument("-np", default=1, type=int,
                        help="Number of partitions.")
    parser.add_argument("-loc", default="", type=str,
                        help="Path to location where files should be saved to.")
    parser.add_argument("-pf", default="", type=str,
                        help="Prefix for file names.")
    parser.add_argument("-ow", default=False, type=bool,
                        help="Flag if MPI communication should use one sided communications.")
    args = parser.parse_args()

    domain = [(0, 0), (1, 1)]
    nx = args.nx
    ny = args.ny
    nz = 1
    dt = args.dt
    nt = args.nt
    eps = args.eps
    method = args.m
    datatype = np.float64
    save_freq = args.sf
    filename = 'zhao_' + str(method) + '.pickle'

    nparts = args.np
    path = args.loc
    prefix = args.pf

    onesided = args.ow

    timer = ti.Timings(name="Burger's equation - Zhao setup")

    run_zhao()

    timer.start(name="Post processing time", level=1)
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        if save_freq >= 0:
            postprocess_zhao()

    timer.stop(name="Post processing time")

    if MPI.COMM_WORLD.Get_rank() == 0:
        timer.list_timings()