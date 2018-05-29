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
This module contains a small example to experiment with gridtools and mpi
"""
import numpy as np
import pickle
from mpi4py import MPI

import gridtools as gt


def stencil_simple_mult(const, in_u):
    """
    GT4Py stencil performing a multiplication with a constant on the grid.

    Arguments
    ---------
    const : GT4Py Global
        Constant for multiplication.
    in_u : GT4Py Equation
        The current grid value.

    Returns
    -------
    out_u : GT4Py Equation
        The result of the multiplication.
    """
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = in_u[i, j, k] * const

    return out_u

def stencil_move_right(in_u):
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    out_u = gt.Equation()

    out_u[i, j, k] = in_u[i - 1, j, k]

    return out_u

def stencil_move_down(in_u):
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    out_u = gt.Equation()

    out_u[i, j, k] = in_u[i, j - 1, k]

    return out_u

def stencil_add_neighbors(in_u):
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    out_u = gt.Equation()

    out_u[i, j, k] = in_u[i, j, k] + in_u[i + 1, j, k] + in_u[i - 1, j, k] + in_u[i, j + 1, k] + in_u[i, j - 1, k]

    return out_u


# def stencil_add_neighbors_boundary_ym(in_u):
#     i = gt.Index()
#     j = gt.Index()
#     k = gt.Index()
#
#     out_u = gt.Equation()
#
#     out_u[i, j, k] = in_u[i, j, k] + in_u[i + 1, j, k] + in_u[i - 1, j, k] + in_u[i, j + 1, k]
#
#     return out_u
#
# def stencil_add_neighbors_boundary_yp(in_u):
#     i = gt.Index()
#     j = gt.Index()
#     k = gt.Index()
#
#     out_u = gt.Equation()
#
#     out_u[i, j, k] = in_u[i, j, k] + in_u[i + 1, j, k] + in_u[i - 1, j, k]  + in_u[i, j - 1, k]
#
#     return out_u
#
# def stencil_add_neighbors_boundary_xm(in_u):
#     i = gt.Index()
#     j = gt.Index()
#     k = gt.Index()
#
#     out_u = gt.Equation()
#
#     out_u[i, j, k] = in_u[i, j, k] + in_u[i + 1, j, k] + in_u[i, j - 1, k] + in_u[i, j + 1, k]
#
#     return out_u
#
# def stencil_add_neighbors_boundary_xp(in_u):
#     i = gt.Index()
#     j = gt.Index()
#     k = gt.Index()
#
#     out_u = gt.Equation()
#
#     out_u[i, j, k] = in_u[i, j, k] + in_u[i - 1, j, k] + in_u[i, j + 1, k] + in_u[i, j - 1, k]
#
#     return out_u

class HelloNeighbor:
    def __init__(self, id):

        experiment_id = id

        self.comm = MPI.COMM_WORLD
        self.comm_size = self.comm.Get_size()
        self.comm_rank = self.comm.Get_rank()

        assert(self.comm_size == 2)

        self.domain = [(0, 0), (2, 2)]
        self.nx = 8
        self.ny = 5
        self.nt = 4
        self.const = 1.0
        self.datatype = np.float64
        self.filename = 'hello_neighbor_output.pickle'

        self.lx = self.nx // 2
        self.ly = self.ny

        self.xminusboundary = MPI.DOUBLE.Create_vector(count=self.lx, blocklength=1, stride=self.ly)
        self.xminusboundary.Commit()
        self.xplusboundary = MPI.DOUBLE.Create_vector(count=self.lx, blocklength=1, stride=self.ly)
        self.xplusboundary.Commit()

        self.yplusboundary = MPI.DOUBLE.Create_contiguous(count=self.lx)
        self.yplusboundary.Commit()
        self.yminusboundary = MPI.DOUBLE.Create_contiguous(count=self.lx)
        self.yminusboundary.Commit()

#        if self.comm_rank == 0:
#            self.win_ym = MPI.Win.Allocate_shared(self.ly * MPI.DOUBLE.Get_size(), MPI.DOUBLE.Get_size(), comm=self.comm)
#        else:
#            self.win_ym = MPI.Win.Allocate_shared(0, MPI.DOUBLE.Get_size())

#        self.ymbuf, itemsize = self.win_ym.Shared_query(0)
#        self.ymbufary = np.ndarray(buffer=self.ymbuf, dtype=self.datatype, shape=(self.ly,))

#        if self.comm_rank == 0:
#            self.win_yp = MPI.Win.Allocate_shared(self.ly * MPI.DOUBLE.Get_size(), MPI.DOUBLE.Get_size(), comm=self.comm)
#        else:
#            self.win_yp = MPI.Win.Allocate_shared(0, MPI.DOUBLE.Get_size())

#        self.ypbuf, itemsize = self.win_yp.Shared_query(0)
#        self.ypbufary = np.ndarray(buffer=self.ypbuf, dtype=self.datatype, shape=(self.ly,))

        self.reqs = []

        if self.comm_rank == 0:
            self.start_x = 0
            self.start_y = 0
        else:
            self.start_x = self.lx
            self.start_y = 0

        # Create the grid:
        self.x = np.linspace(self.domain[0][0], self.domain[1][0], self.nx)
        self.y = np.linspace(self.domain[0][1], self.domain[1][1], self.ny)

        # Instantiate the arrays representing the solution
        self.unow = np.zeros((self.lx, self.ly, 1), dtype = self.datatype)
        self.unew = np.zeros((self.lx, self.ly, 1), dtype = self.datatype)

        # Set the initial conditions:
        self.initial_conditions(id=experiment_id)
        self.set_stencil(id=experiment_id)

    def set_stencil(self, id):
        if id == 0:
            definitions_func_ = stencil_simple_mult
            domain_ = gt.domain.Rectangle((1, 1, 0), (self.nx - 2, self.ny - 2, 0))
            # Convert global inputs to GT4Py Global's
            self.const = gt.Global(self.const)

            # Instantiate stencil object
            self.stencil = gt.NGStencil(
                    definitions_func = definitions_func_,
                    inputs = {'in_u': self.unow},
                    global_inputs = {'const': self.const},
                    outputs = {'out_u': self.unew},
                    domain = domain_,
                    mode = gt.mode.NUMPY)
        elif id == 1:
            definitions_func_ = stencil_move_right
            domain_ = gt.domain.Rectangle((1, 0, 0), (self.lx - 1, self.ly, 0))
            self.stencil = gt.NGStencil(
                    definitions_func = definitions_func_,
                    inputs = {'in_u': self.unow},
                    outputs = {'out_u': self.unew},
                    domain = domain_,
                    mode = gt.mode.NUMPY)
        elif id == 2:
            definitions_func_ = stencil_move_down
            domain_ = gt.domain.Rectangle((0, 1, 0), (self.lx, self.ly - 1, 0))
            self.stencil = gt.NGStencil(
                    definitions_func = definitions_func_,
                    inputs = {'in_u': self.unow},
                    outputs = {'out_u': self.unew},
                    domain = domain_,
                    mode = gt.mode.NUMPY)
        elif id == 3:
            definitions_func_ = stencil_add_neighbors # stencil_simple_mult #
            # Convert global inputs to GT4Py Global's
            # self.const = gt.Global(self.const)
            domain_ = gt.domain.Rectangle((1, 1, 0), (self.lx - 2, self.ly - 2, 0))
            self.stencil = gt.NGStencil(
                    definitions_func = definitions_func_,
                    inputs = {'in_u': self.unow},
                    outputs = {'out_u': self.unew},
                    # global_inputs={'const': self.const},
                    domain = domain_,
                    mode = gt.mode.NUMPY)

            # self.ymboundary_stencil = gt.NGStencil(
            #     definitions_func=stencil_add_neighbors_boundary_ym,
            #     inputs={'in_u': self.unow},
            #     outputs={'out_u': self.unew},
            #     domain=gt.domain.Rectangle((1,0,0), (self.lx - 2, 0, 0)),
            #     mode=gt.mode.NUMPY)
            #
            # self.ypboundary_stencil = gt.NGStencil(
            #     definitions_func=stencil_add_neighbors_boundary_yp,
            #     inputs={'in_u': self.unow},
            #     outputs={'out_u': self.unew},
            #     domain=gt.domain.Rectangle((1, self.ly - 1, 0), (self.lx - 2, self.ly - 1, 0)),
            #     mode=gt.mode.NUMPY)
            #
            # self.xmboundary_stencil = gt.NGStencil(
            #     definitions_func=stencil_add_neighbors_boundary_xm,
            #     inputs={'in_u': self.unow},
            #     outputs={'out_u': self.unew},
            #     domain=gt.domain.Rectangle((0, 1, 0), (0, self.ly - 2, 0)),
            #     mode=gt.mode.NUMPY)
            #
            # self.xpboundary_stencil = gt.NGStencil(
            #     definitions_func=stencil_add_neighbors_boundary_xp,
            #     inputs={'in_u': self.unow},
            #     outputs={'out_u': self.unew},
            #     domain=gt.domain.Rectangle((self.lx - 1, 1, 0), (self.lx - 1, self.ly - 2, 0)),
            #     mode=gt.mode.NUMPY)

    def initial_conditions(self, id):
        if id == 0:
            self.ic_midblock()
        elif id == 1:
            self.ic_leftwall()
        elif id == 2:
            self.ic_upwall()
        elif id == 3:
            # self.ic_index()
            self.ic_singlemid1()
        else:
            print("Wrong id for initial condition:" + id + " not available")

        print('Processor ' + str(self.comm_rank) + ' Step 0 (IC)' + ' u: \n' + str(
            self.unew[:, :, 0]) + '\n')

    def ic_index(self):
        for i in range(self.lx):
            for j in range(self.ly):
                self.unew[i, j, 0] = (self.start_x + i) * self.ny + self.start_y + j

    def ic_leftwall(self):
        if self.comm_rank == 0:
            temp = 1.0
        else:
            temp = 0.0

        i = 0
        for j in range(self.ly):
            self.unew[i, j, 0] = temp

    def ic_upwall(self):
        if self.comm_rank == 0:
            temp = 1.0
        else:
            temp = 0.0

        j = 0
        for i in range(self.lx):
            self.unew[i, j, 0] = temp

    def ic_midblock(self):
        if self.comm_rank == 0:
            temp = 1.0
        else:
            temp = 0.0

        for i in range(self.nx):
            for j in range(self.ny):
                if (0.5 <= self.x[i] and self.x[i] <= 1.0) and (0.5 <= self.y[j] and self.y[j] <= 1.0):
                    self.unew[i, j, 0] = temp

    def ic_singlemid1(self):
        if self.comm_rank == 0:
            midx = self.lx // 2
            midy = self.ly // 2
            self.unew[midx, midy, 0] = 1.0


    def communicate(self, id, onesided=False):
        if onesided:
            self.communicate_one_sided()
        else:
            self.communicate_two_way(id)

    def communicate_one_sided(self):
        if self.comm_rank == 0:
            self.win_ym.Fence()
            self.ymbufary[:] = self.unew[-2, :, 0]
            self.win_ym.Fence()

            self.win_yp.Fence()
            self.ypbufary[:] = self.unew[1, :, 0]
            self.win_yp.Fence()
        else:
            self.win_ym.Fence()
            self.win_ym.Get(self.ymbufary, 0)
            self.unew[0, :, 0] = self.ymbufary[:]
            self.win_ym.Fence()

            self.win_yp.Fence()
            self.win_yp.Get(self.ypbufary, 0)
            self.unew[-1, :, 0] = self.ypbufary[:]
            self.win_yp.Fence()

        if self.comm_rank != 0:
            self.win_ym.Fence()
            self.ymbufary[:] = self.unew[-2, :, 0]
            self.win_ym.Put(self.ymbufary, 0)
            self.win_ym.Fence()

            self.win_yp.Fence()
            self.ypbufary[:] = self.unew[1, :, 0]
            self.win_yp.Put(self.ypbufary, 0)
            self.win_yp.Fence()
        else:
            self.win_ym.Fence()
            self.unew[0, :, 0] = self.ymbufary[:]
            self.win_ym.Fence()

            self.win_yp.Fence()
            self.unew[-1, :, 0] = self.ypbufary[:]
            self.win_yp.Fence()


    def communicate_two_way(self, id):
        if id == 0:
            # Swap whole domain
            self.reqs = [None, None]
            if self.comm_rank == 0:
                self.reqs[0] = self.comm.Isend(self.unew, 1, 0)
                self.reqs[1] = self.comm.Irecv(self.unow, 1, 0)
            else:
                self.reqs[0] = self.comm.Irecv(self.unow, 0, 0)
                self.reqs[1] = self.comm.Isend(self.unew, 0, 0)

        elif id == 1:
            # Send last line in x-direction of processor 0 to first line in x-direction of processor 1
            self.reqs = [None]
            if self.comm_rank == 0:
                self.reqs[0] = self.comm.Isend(self.unew[-1, :, :], 1, 0)
            else:
                self.reqs[0] = self.comm.Irecv(self.unow[0, :, :], 0, 0)


        elif id == 2:
            # Send last line in y-direction of processor 0 to first line in y-direction of processor 1
            self.reqs = [None]
            if self.comm_rank == 0:
                self.reqs[0] = self.comm.Isend([self.unew, self.ydirection], 1, 0)
            else:
                self.reqs[0] = self.comm.Irecv([self.unow, self.ydirection], 0, 0)
        elif id == 3:
            # Send from Processor 0 to Processor 1
            # Along x-direction:
            # self.reqs = [None]
            # if self.comm_rank == 0:
            #     self.reqs[0] = self.comm.Isend([self.unew.reshape((self.lx*self.ly*1))[(self.ly - 1):], 1, self.xplusboundary], 1, 0)
            # else:
            #     self.reqs[0] = self.comm.Irecv([self.unow.reshape((self.lx*self.ly*1))[0:], 1, self.xminusboundary], 0, 0)

            # # Along y-direction:
            self.reqs = [None, None, None, None]
            if self.comm_rank == 0:
                self.reqs[0] = self.comm.Isend([self.unew[-2, :, :], 1, self.yminusboundary], 1, 0)
                self.reqs[1] = self.comm.Irecv([self.unew[-1, :, :], 1, self.yminusboundary], 1, 0)
                self.reqs[2] = self.comm.Isend([self.unew[1, :, :], 1, self.yplusboundary], 1, 0)
                self.reqs[3] = self.comm.Irecv([self.unew[0, :, :], 1, self.yplusboundary], 1, 0)
            else:
                self.reqs[0] = self.comm.Irecv([self.unew[0, :, :], 1, self.yplusboundary], 0, 0)
                self.reqs[1] = self.comm.Isend([self.unew[1, :, :], 1, self.yplusboundary], 0, 0)
                self.reqs[2] = self.comm.Irecv([self.unew[-1, :, :], 1, self.yminusboundary], 0, 0)
                self.reqs[3] = self.comm.Isend([self.unew[-2, :, :], 1, self.yminusboundary], 0, 0)


    def time_integration(self, id):
        for n in range(self.nt):
            # Advance fields
            self.unow[:, :, 0] = self.unew[:, :, 0]

            # Exchange with neighbor:
            self.communicate(id)

            # Step the solution
            self.stencil.compute()
            MPI.Request.waitall(self.reqs[:])
            # self.ymboundary_stencil.compute()
            # self.ypboundary_stencil.compute()
            #
            # self.xmboundary_stencil.compute()
            # self.xpboundary_stencil.compute()

            # Print some result
            if id == 0:
                print('Processor %5.i, Step %5.i, u = %5.5f' % (self.comm_rank, n + 1, self.unew.max()))
            elif id == 1:
                print('Processor ' + str(self.comm_rank) + ' Step ' + str(n + 1) + ' u: \n' + str(self.unow[:, :, 0]) + '\n')
            elif id == 2:
                print('Processor ' + str(self.comm_rank) + ' Step ' + str(n + 1) + ' u: \n' + str(self.unow[:, :, 0]) + '\n')
            elif id == 3:
                print('Processor ' + str(self.comm_rank) + ' Step ' + str(n + 1) + ' u: \n' + str(
                    self.unew[:, :, 0]) + '\n')


if __name__ == "__main__":
    experiment_number = 3

    test = HelloNeighbor(id=experiment_number)
    test.time_integration(id=experiment_number)
