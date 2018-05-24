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
# from mpi4py import MPI
from pymetis import part_graph

# import gridtools as gt

class DomainSubdivision:
    def __init__(self, id, size, border, gridpoints, neighbors):
        self.id = id
        self.size = size
        self.border = border
        self.gridpoints = gridpoints
        self.neighbors = neighbors

class DomainDecomposition:
    def __init__(self, domain, periodic, subdivs_per_dim, stencil_extend):
        self.domain = domain
        self.periodic = periodic
        self.subdiv_per_dim = subdivs_per_dim
        self.stencil_extend = stencil_extend

        subdiv_size = self.domain / self.subdiv_per_dim
        subdiv_gridpoints = 1
        for e in subdiv_size:
            subdiv_gridpoints *= e

        total_subdivisions = 1
        for e in subdivs_per_dim:
            total_subdivisions *= e

        border_sizes = np.zeros((stencil_extend.size))
        # border_sizes[0] = subdiv_size[1] * subdiv_size[2] * stencil_extend[0]
        # border_sizes[1] = subdiv_size[1] * subdiv_size[2] * stencil_extend[1]
        # border_sizes[2] = subdiv_size[0] * subdiv_size[2] * stencil_extend[2]
        # border_sizes[3] = subdiv_size[0] * subdiv_size[2] * stencil_extend[3]
        # border_sizes[4] = subdiv_size[0] * subdiv_size[1] * stencil_extend[4]
        # border_sizes[5] = subdiv_size[0] * subdiv_size[1] * stencil_extend[5]
        for e in range(stencil_extend.size):
            border_sizes[e] = subdiv_size[((e // 2) - 1) % 3] * subdiv_size[((e // 2) + 1) % 3] * stencil_extend[e]

        self.subdivisions = []

        self.adjncy = []
        self.xadj = [] #np.zeros(total_subdivisions, dtype=np.int)
        self.vweights = [] #np.zeros(total_subdivisions, dtype=np.int)
        self.eweights = []
        self.edgecounter = 0
        self.alist = []

        for i in range(self.subdiv_per_dim[0]):
            for j in range(self.subdiv_per_dim[1]):
                for k in range(self.subdiv_per_dim[2]):
                    ind = (i * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + k
                    # End of Domain in negative X direction
                    if i == 0:
                        if periodic[0]:
                            negx = ((self.subdiv_per_dim[0] - 1) * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + k
                        else:
                            negx = None
                    else:
                        negx = ((i - 1) * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + k

                    # End of Domain in positive X direction
                    if i == self.subdiv_per_dim[0] - 1:
                        if periodic[0]:
                            posx = (0 * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + k
                        else:
                            posx = None
                    else:
                        posx = ((i + 1) * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + k

                    # End of Domain in negative Y direction
                    if j == 0:
                        if periodic[1]:
                            negy = (i * self.subdiv_per_dim[1] + self.subdiv_per_dim[1] - 1) * self.subdiv_per_dim[2] + k
                            print(ind, negy)
                        else:
                            negy = None
                    else:
                        negy = (i * self.subdiv_per_dim[1] + j - 1) * self.subdiv_per_dim[2] + k

                    # End of Domain in positive Y direction
                    if j == self.subdiv_per_dim[1] - 1:
                        if periodic[1]:
                            posy = (i * self.subdiv_per_dim[1] + 0) * self.subdiv_per_dim[2] + k
                        else:
                            posy = None
                    else:
                        posy = (i * self.subdiv_per_dim[1] + j + 1) * self.subdiv_per_dim[2] + k

                    # End of Domain in negative Z direction
                    if k == 0:
                        if periodic[2]:
                            negz = (i * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + self.subdiv_per_dim[2] - 1
                        else:
                            negz = None
                    else:
                        negz = (i * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + k - 1

                    # End of Domain in positive Z direction
                    if k == self.subdiv_per_dim[2] - 1:
                        if periodic[2]:
                            posz = (i * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + 0
                        else:
                            posz = None
                    else:
                        posz = (i * self.subdiv_per_dim[1] + j) * self.subdiv_per_dim[2] + k + 1

                    negx = negx if negx != ind else None
                    posx = posx if posx != ind else None
                    negy = negy if negy != ind else None
                    posy = posy if posy != ind else None
                    negz = negz if negz != ind else None
                    posz = posz if posz != ind else None

                    nindex = [negx, posx, negy, posy, negz, posz]

                    self.subdivisions.append(DomainSubdivision(ind, subdiv_size, border_sizes,
                                                               subdiv_gridpoints, nindex))
                    self.vweights.append(int(subdiv_gridpoints))
                    self.xadj.append(int(self.edgecounter))
                    for e in range(len(nindex)):
                        if nindex[e] is not None:
                            self.edgecounter += 1
                            self.adjncy.append(int(nindex[e]))
                            self.eweights.append(int(border_sizes[e]))

                    # templist = []
                    #
                    # for e in range(len(nindex)):
                    #     if nindex[e] is not None:
                    #         templist.append(int(nindex[e]))
                    #
                    # self.alist.append(np.asarray(templist))


        # print(self.alist)

        print(self.adjncy, self.xadj, self.vweights, self.eweights, self.edgecounter)
        #
        # for i in self.subdivisions:
        #     print(i.id, i.size, i.border, i.gridpoints, i.neighbors)


if __name__ == "__main__":
    domain = np.array([200, 100, 1])
    slices = np.array([20, 1, 1])
    stencil = np.array([1, 1, 1, 1, 0, 0])
    periodic = np.array([0, 0, 0])

    ddc = DomainDecomposition(domain, periodic, slices, stencil)


    # def part_graph(nparts, adjacency=None, xadj=None, adjncy=None,
                   # vweights=None, eweights=None, recursive=None)
    parts = 2

    # print(part_graph(parts, adjacency=ddc.alist))


    print(part_graph(parts, xadj=ddc.xadj, adjncy=ddc.adjncy, vweights=ddc.vweights, eweights=ddc.eweights))