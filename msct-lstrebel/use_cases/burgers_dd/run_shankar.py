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


def run_shankar():
    timer.start(name="Overall Shankar time", level=1)
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
    # Infer the grid size
    dx = float(domain[1][0] - domain[0][0]) / nx
    dy = float(domain[1][1] - domain[0][1]) / ny

    # Register fields and stencils to the DomainDecomposition class:
    prepared_domain = DomainDecomposition("subdomains_pymetis.dat.part." + str(nparts), "metis",
                                          path=path, prefix=prefix)

    prepared_domain.register_field(fieldname="unow",
                                   halo=halo,
                                   field_ic_file=path + prefix +"shankar_initial_conditions_unew.npy")
                                   # field_bc_file=path + prefix +"shankar_boundary_conditions_unew.npy")
    prepared_domain.register_field(fieldname="vnow",
                                   halo=halo,
                                   field_ic_file=path + prefix +"shankar_initial_conditions_vnew.npy")
                                   # field_bc_file=path + prefix +"shankar_boundary_conditions_vnew.npy")

    prepared_domain.register_field(fieldname="unew",
                                   halo=halo,
                                   field_ic_file=path + prefix +"shankar_initial_conditions_unew.npy")
                                   # field_bc_file=path + prefix +"shankar_boundary_conditions_unew.npy")
    prepared_domain.register_field(fieldname="vnew",
                                   halo=halo,
                                   field_ic_file=path + prefix +"shankar_initial_conditions_vnew.npy")
                                   # field_bc_file=path + prefix +"shankar_boundary_conditions_vnew.npy")

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
        prepared_domain.set_boundary_condition("unow", 0, 1, np.zeros((1, ny, 1)))
        prepared_domain.set_boundary_condition("unow", 1, 1, np.zeros((1, ny, 1)))
        prepared_domain.set_boundary_condition("unow", 2, 1, np.zeros((nx, 1, 1)))
        prepared_domain.set_boundary_condition("unow", 3, 1, np.zeros((nx, 1, 1)))

        prepared_domain.set_boundary_condition("vnow", 0, 1, np.zeros((1, ny, 1)))
        prepared_domain.set_boundary_condition("vnow", 1, 1, np.zeros((1, ny, 1)))
        prepared_domain.set_boundary_condition("vnow", 2, 1, np.zeros((nx, 1, 1)))
        prepared_domain.set_boundary_condition("vnow", 3, 1, np.zeros((nx, 1, 1)))

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
        if ((save_freq > 0) and (n % save_freq == 0)) or (save_freq >= 0 and n + 1 == nt):
            prepared_domain.save_fields(["unew", "vnew"], postfix="t_"+str(n + 1))
        timer.stop(name="Saving fields during time integration")

    timer.stop(name="Time integration")

    timer.stop(name="Overall Shankar time")


def postprocess_shankar():
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
    args = parser.parse_args()

    domain = [(0, 0), (2, 2)]
    nx = args.nx
    ny = args.ny
    nz = 1
    dt = args.dt
    nt = args.nt
    eps = args.eps
    method = args.m
    datatype = np.float64
    save_freq = args.sf
    filename = 'shankar' + str(method) + '.pickle'

    nparts = args.np
    path = args.loc
    prefix = args.pf

    timer = ti.Timings(name="Burger's equation - Shankar setup")

    run_shankar()

    timer.start("Post processing time", level=1)

    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        postprocess_shankar()

    timer.stop("Post processing time")

    if MPI.COMM_WORLD.Get_rank() == 0:
        timer.list_timings()
