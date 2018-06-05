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
This module contains a small example to experiment with conditional stencils in gridtools and mpi
"""
import numpy as np
import pickle
from mpi4py import MPI

import gridtools as gt


def stencil_cond_add_neighbors(in_u, in_condi):
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # condi = in_condi[0]

    out_u = gt.Equation()

    # out_u[i, j, k] = in_u[i, j, k] * in_condi[i, j, k]

    if in_condi[i, j, k] > 1.0:
        out_u[i, j, k] = in_u[i, j, k] + in_u[i - 1, j, k] + in_u[i, j - 1, k] + in_u[i + 1, j, k] + in_u[i, j + 1, k]
    else:
        out_u[i, j, k] = in_u[i, j, k] * 2.0

    return out_u


class HelloNeighbor:
    def __init__(self):

        self.comm = MPI.COMM_WORLD
        self.comm_size = self.comm.Get_size()
        self.comm_rank = self.comm.Get_rank()

        assert(self.comm_size == 2)

        self.domain = [(0, 0), (2, 2)]
        self.nx = 8
        self.ny = 5
        self.nt = 6
        self.const = 0.0
        self.datatype = np.float64
        self.filename = 'conditional_stencil_output.pickle'

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
        self.unow = np.zeros((self.lx, self.ly, 1), dtype=self.datatype)
        self.unew = np.zeros((self.lx, self.ly, 1), dtype=self.datatype)

        self.condi = 0.5 * np.ones((self.lx, self.ly, 1), dtype=self.datatype)

        # Set the initial conditions:
        self.initial_conditions()
        self.set_stencil()

    def set_stencil(self):
            definitions_func_ = stencil_cond_add_neighbors
            # # Convert global inputs to GT4Py Global's
            # self.const = gt.Global(self.const)
            domain_ = gt.domain.Rectangle((1, 1, 0), (self.lx - 2, self.ly - 2, 0))
            self.stencil = gt.NGStencil(
                    definitions_func = definitions_func_,
                    inputs = {'in_u': self.unow, 'in_condi': self.condi},
                    # global_inputs={'const': self.const},
                    outputs = {'out_u': self.unew},
                    domain = domain_,
                    mode = gt.mode.NUMPY)

    def initial_conditions(self):
        self.ic_singlemid1()

        print('Processor ' + str(self.comm_rank) + ' Step 0 (IC)' + ' u: \n' + str(
            self.unew[:, :, 0]) + '\n')

    def ic_index(self):
        for i in range(self.lx):
            for j in range(self.ly):
                self.unew[i, j, 0] = (self.start_x + i) * self.ny + self.start_y + j

    def ic_singlemid1(self):
        if self.comm_rank == 0:
            midx = self.lx // 2
            midy = self.ly // 2
            self.unew[midx, midy, 0] = 1.0

    def communicate(self):
        self.communicate_two_way(id)

    def communicate_two_way(self, id):
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


    def time_integration(self):
        for n in range(self.nt):
            # Advance fields
            self.unow[:, :, 0] = self.unew[:, :, 0]

            # Change time dependent conditional
            if n > (self.nt // 2 - 1):
                print("Second half")
                self.condi[:] = 2.0 * self.condi[:]
            # print("Conditional: \n")
            # print(self.condi)
            print("\n")
            # Print extent
            print("Stencil extent: " + str(self.stencil.get_extent()) + " \n")

            # Exchange with neighbor:
            self.communicate()

            # Step the solution
            self.stencil.compute()
            MPI.Request.waitall(self.reqs[:])

            # Print some result
            print('Processor ' + str(self.comm_rank) + ' Step ' + str(n + 1) + ' u: \n' + str(
                    self.unew[:, :, 0]) + '\n')


if __name__ == "__main__":

    test = HelloNeighbor()
    test.time_integration()
