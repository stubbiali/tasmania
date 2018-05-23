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


class HelloNeighbor:
    def __init__(self):

        self.comm = MPI.COMM_WORLD
        self.comm_size = self.comm.Get_size()
        self.comm_rank = self.comm.Get_rank()

        assert(self.comm_size == 2)

        if self.comm_rank == 0:
            print("{} processes are communicating.".format(self.comm_size))

        self.domain = [(0, 0), (2, 2)]
        self.nx = 10
        self.ny = 10
        self.nt = 2
        self.const = 1.0
        self.datatype = np.float64
        self.filename = 'hello_neighbor_output.pickle'

        self.lx = 5
        self.ly = 10

        self.left_boundary = MPI.DOUBLE.Create_vector(count=self.lx, blocklength=1, stride=self.ly)
        self.left_boundary.Commit()
        self.right_boundary = MPI.DOUBLE.Create_vector(count=self.lx, blocklength=1, stride=self.ly)
        self.right_boundary.Commit()

        self.top_boundary = MPI.DOUBLE.Create_contiguous(count=self.lx)
        self.top_boundary.Commit()
        self.bottom_boundary = MPI.DOUBLE.Create_contiguous(count=self.lx)
        self.bottom_boundary.Commit()

        if self.comm_rank == 0:
            self.start_x = 0
            self.start_y = 0
        else:
            self.start_x = 5
            self.start_y = 0

        # Create the grid:
        self.x = np.linspace(self.domain[0][0], self.domain[1][0], self.nx)
        self.y = np.linspace(self.domain[0][1], self.domain[1][1], self.ny)

        # Instantiate the arrays representing the solution
        self.unow = np.zeros((self.lx, self.ly, 1), dtype=self.datatype)
        self.unew = np.zeros((self.lx, self.ly, 1), dtype=self.datatype)

        # self.testnow = np.zeros((self.lx * self.ly), dtype=self.datatype)
        # self.testnew = np.zeros((self.lx * self.ly), dtype=self.datatype)


        # Set the initial conditions:
        self.initial_conditions()
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

    def initial_conditions(self):
        for i in range(self.lx):
            for j in range(self.ly):
                self.unew[i, j, 0] = (self.start_x + i) * self.ny + self.start_y + j
                # self.testnew[i*self.ly + j] = (self.start_x + i) * self.ny + self.start_y + j

        # Print some result
        print('Processor ' + str(self.comm_rank) + ' Step 0 (IC)' + ' u: \n' + str(
            self.unew[:, :, 0]) + '\n')
        # print('Processor ' + str(self.comm_rank) + ' Step 0 (IC)' + ' u: \n' + str(
        #     self.testnew[:]) + '\n')

    def communicate(self):
            # Send from Processor 0 to Processor 1
            # Along x-direction:
            # reqs = [None]
            # if self.comm_rank == 0:
            #     reqs[0] = self.comm.Isend([self.testnew[(self.ly - 1):], 1, self.right_boundary], 1, 0)
            # else:
            #     reqs[0] = self.comm.Irecv([self.testnow[0:], 1, self.left_boundary], 0, 0)
            # MPI.Request.waitall(reqs)

            reqs = [None]
            if self.comm_rank == 0:
                reqs[0] = self.comm.Isend([self.unew.reshape((self.lx*self.ly*1))[(self.ly - 1):], 1, self.right_boundary], 1, 0)
            else:
                reqs[0] = self.comm.Irecv([self.unow.reshape((self.lx*self.ly*1))[0:], 1, self.left_boundary], 0, 0)
            MPI.Request.waitall(reqs)

            # Along y-direction:
            reqs = [None]
            if self.comm_rank == 0:
                reqs[0] = self.comm.Isend([self.unew[-1, :, :], 1, self.bottom_boundary], 1, 0)
            else:
                reqs[0] = self.comm.Irecv([self.unow[0, :, :], 1, self.top_boundary], 0, 0)
            MPI.Request.waitall(reqs)

    def communicate_with_buffer(self):
            # Send from Processor 0 to Processor 1
            # Along x-direction:
            reqs = [None]
            sendbufx = np.empty(self.ly, dtype=self.datatype)
            recbufx = np.empty(self.ly, dtype=self.datatype)

            if self.comm_rank == 0:
                sendbufx[:] = self.unew[-1, :, 0]
                reqs[0] = self.comm.Isend(sendbufx, 1, 0)
                # reqs[0] = self.comm.Isend([self.unew[-1, :, :], 1, self.right_boundary], 1, 0)
            else:
                reqs[0] = self.comm.Irecv(recbufx, 0, 0)
                # reqs[0] = self.comm.Irecv([self.unow[0, :, :], 1, self.right_boundary], 0, 0)
            MPI.Request.waitall(reqs)

            if self.comm_rank == 1:
                self.unow[0, :, 0] = recbufx[:]

            # Send from Processor 0 to Processor 1
            # Along y-direction:
            reqs = [None]
            sendbufy = np.empty(self.lx, dtype=self.datatype)
            recbufy = np.empty(self.lx, dtype=self.datatype)

            if self.comm_rank == 0:
                sendbufy[:] = self.unew[:, -1, 0]
                reqs[0] = self.comm.Isend(sendbufy, 1, 0)
                # reqs[0] = self.comm.Isend([self.unew[-1, :, :], 1, self.right_boundary], 1, 0)
            else:
                reqs[0] = self.comm.Irecv(recbufy, 0, 0)
                # reqs[0] = self.comm.Irecv([self.unow[0, :, :], 1, self.right_boundary], 0, 0)
            MPI.Request.waitall(reqs)

            if self.comm_rank == 1:
                self.unow[:, 0, 0] = recbufy[:]

    def time_integration(self):
        for n in range(self.nt):
            # Advance fields
            self.unow[:, :, 0] = self.unew[:, :, 0]
            # self.testnow[:] = self.testnew[:]

            # Exchange with neighbor:
            self.communicate()

            # Step the solution
            self.stencil.compute()

            # Print some result
            # print('Processor ' + str(self.comm_rank) + ' Step ' + str(n + 1) + ' u: \n' + str(self.testnow.reshape((5, 10))) + '\n')
            print('Processor ' + str(self.comm_rank) + ' Step ' + str(n + 1) + ' u: \n' + str(self.unow[:, :, 0]) + '\n')

            # print(self.unow[:, 0, :])
            # print(self.unow[0, :, :])

if __name__ == "__main__":

    test = HelloNeighbor()
    test.time_integration()