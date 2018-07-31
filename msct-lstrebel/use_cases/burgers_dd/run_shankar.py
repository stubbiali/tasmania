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
from mpi4py import MPI

import gridtools as gt
import stencils
import timer as ti
from domain_decomposition import DomainDecomposition, DomainPostprocess

domain = [(0, 0), (2, 2)]
nx = 100
ny = 100
nz = 1
dt = 0.001
nt = 1000
eps = 0.01
method = 'forward_backward'
datatype = np.float64
save_freq = 1
print_freq = 10
file_output = True
filename = 'test_shankar_' + str(method) + '.pickle'


def run_shankar():
    timer = ti.Timings(name="Burger's equation - Shankar setup")
    timer.start(name="Overall Shankar time", level=1)
    timer.start(name="Initialization", level=2)

    # Driver
    #
    # Infer the grid size
    dx = float(domain[1][0] - domain[0][0]) / nx
    dy = float(domain[1][1] - domain[0][1]) / ny

    # Register fields and stencils to the DomainDecomposition class:
    prepared_domain = DomainDecomposition("subdomains_pymetis.dat.part.4", "metis")

    prepared_domain.register_field(fieldname="unow",
                                   halo=[1, 1, 1, 1, 0, 0],
                                   field_ic_file="shankar_initial_conditions_unew.npy")
                                   # field_bc_file="shankar_boundary_conditions_unew.npy")
    prepared_domain.register_field(fieldname="vnow",
                                   halo=[1, 1, 1, 1, 0, 0],
                                   field_ic_file="shankar_initial_conditions_vnew.npy")
                                   # field_bc_file="shankar_boundary_conditions_vnew.npy")

    prepared_domain.register_field(fieldname="unew",
                                   halo=[1, 1, 1, 1, 0, 0],
                                   field_ic_file="shankar_initial_conditions_unew.npy")
                                   # field_bc_file="shankar_boundary_conditions_unew.npy")
    prepared_domain.register_field(fieldname="vnew",
                                   halo=[1, 1, 1, 1, 0, 0],
                                   field_ic_file="shankar_initial_conditions_vnew.npy")
                                   # field_bc_file="shankar_boundary_conditions_vnew.npy")

    # Convert global inputs to GT4Py Global's
    dt_ = gt.Global(dt)
    dx_ = gt.Global(dx)
    dy_ = gt.Global(dy)
    eps_ = gt.Global(eps)

    stencil = prepared_domain.register_stencil(
        definitions_func=stencils.stencil_burgers_forward_backward,
        inputs={'in_u': "unow", 'in_v': "vnow"},
        global_inputs={'dt': dt_, 'dx': dx_, 'dy': dy_, 'eps': eps_},
        outputs={'out_u': "unew", 'out_v': "vnew"},
        mode=gt.mode.NUMPY)

    timer.stop(name="Initialization")
    timer.start(name="Time integration", level=2)

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

        timer.start(name="Checkpointing during time integration", level=3)
        # Save
        if ((save_freq > 0) and (n % save_freq == 0)) or (n + 1 == nt):
            prepared_domain.save_fields(["unew", "vnew"], postfix="t_"+str(n))
        timer.stop(name="Checkpointing during time integration")

    timer.stop(name="Time integration")

    timer.stop(name="Overall Shankar time")

    if MPI.COMM_WORLD.Get_rank() == 0:
        timer.list_timings()


def postprocess_shankar():
    postproc = DomainPostprocess()

    postproc.create_pickle_dump(nx, ny, nz, nt, domain, dt, eps, save_freq, filename, cleanup=True)

    # postproc.combine_output_files(size=[100, 100, 1], fieldname="unew", cleanup=True)
    # postproc.combine_output_files(size=[100, 100, 1], fieldname="vnew", cleanup=True)
    #
    # unew = np.load("unew.npy")
    # vnew = np.load("vnew.npy")
    # print('u max = %5.5f, u min = %5.5f, v max = %5.5f, v min = %5.5f, ||u|| max = %12.12f'
    #       % (unew.max(), unew.min(), vnew.max(), vnew.min(), np.sqrt(unew ** 2 + vnew ** 2).max()))


if __name__ == "__main__":
    run_shankar()
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        postprocess_shankar()
