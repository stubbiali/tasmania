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

        self.total_subdivisions = 1
        for e in subdivs_per_dim:
            self.total_subdivisions *= e

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
        self.xadj = []
        self.vweights = []
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
                            negx = (((self.subdiv_per_dim[0] - 1) * self.subdiv_per_dim[1] + j)
                                    * self.subdiv_per_dim[2] + k)
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
                            negy = ((i * self.subdiv_per_dim[1] + self.subdiv_per_dim[1] - 1)
                                    * self.subdiv_per_dim[2] + k)
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
                            negz = ((i * self.subdiv_per_dim[1] + j)
                                    * self.subdiv_per_dim[2] + self.subdiv_per_dim[2] - 1)
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

        self.write_to_file_metis_format(self.adjncy,
                                        self.xadj,
                                        self.vweights,
                                        self.eweights,
                                        self.edgecounter,
                                        "test")
        self.write_to_file_scotch_format(self.adjncy,
                                        self.xadj,
                                        self.vweights,
                                        self.eweights,
                                        self.edgecounter,
                                        "test")

    def prepare_for_pymetis(self):
        self.adjncy.append(int(self.total_subdivisions))
        self.xadj.append(int(self.edgecounter))

    def write_to_file_metis_format(self, adjncy, xadj, vweights, eweights, edgecounter, filename, flag=11):
        """
        Vertex numbering starts with 1 not 0!
        """
        # header line: Number of Vertices" "Number of Edges (counted once)" "3 digit binary flag"
        header = "{0:d} {1:d} {2:03d} \n".format(len(xadj), edgecounter//2, flag)
        # vertex line: s w_1 w_2 ... w_ncon v_1 e_1 v_2 e_2 ... v_k e_k
        # s: size of vertex
        # w_* : weight of vertex
        # v_* : neighbor vertex index
        # e_* : edge weight to neighbor

        vertex_lines = ""
        for i in range(len(xadj)):
            vertex_lines += "{0:d} ".format(vweights[i])
            if i < len(xadj) - 1:
                for j in range(xadj[i], xadj[i + 1]):
                    vertex_lines += "{0:d} {1:d} ".format(adjncy[j] + 1, eweights[j])
            else:
                for j in range(xadj[i], len(adjncy)):
                    vertex_lines += "{0:d} {1:d} ".format(adjncy[j] + 1, eweights[j])
            vertex_lines += "\n"

        content = header + vertex_lines

        with open(filename+"_metis.dat", "w") as f:
            f.writelines(content)

    def write_to_file_scotch_format(self, adjncy, xadj, vweights, eweights, edgecounter, filename, flag=11):
        """ First line: graph file version number
            Second line: number of vertices followed by number of arcs (edge number twice counted)
            Third line: graph base index value (0 or 1) and numeric flag
            //End of Header
            Other lines: [vertex label] [vertex load] vertex_degree [arc_load] arc_end_vertex
        """
        header = "0 \n{0:d} {1:d} \n0 {2:03d}\n".format(len(xadj), edgecounter, flag)
        vertex_lines = ""
        for i in range(len(xadj)):
            vertex_lines += "{0:d} ".format(vweights[i])
            if i < len(xadj) - 1:
                vertex_lines += "{0:d} ".format(xadj[i + 1] - xadj[i])
                for j in range(xadj[i], xadj[i + 1]):
                    vertex_lines += "{0:d} {1:d} ".format(eweights[j], adjncy[j])
            else:
                vertex_lines += "{0:d} ".format(len(adjncy) - xadj[i])
                for j in range(xadj[i], len(adjncy)):
                    vertex_lines += "{0:d} {1:d} ".format(eweights[j], adjncy[j])
            vertex_lines += "\n"

        contents = header + vertex_lines

        with open(filename+"_scotch.src", "w") as f:
            f.writelines(contents)


if __name__ == "__main__":
    domain = np.array([2048, 1024, 512])
    slices = np.array([16, 8, 8])
    stencil = np.array([1, 1, 1, 1, 1, 1])
    periodic = np.array([0, 0, 0])

    ddc = DomainDecomposition(domain, periodic, slices, stencil)

    # def part_graph(nparts, adjacency=None, xadj=None, adjncy=None,
                   # vweights=None, eweights=None, recursive=None)

    ddc.prepare_for_pymetis()

    with open("pymetis_test.dat.part.5", "w") as f:
        partitioning = part_graph(5, xadj=ddc.xadj, adjncy=ddc.adjncy, vweights=ddc.vweights, eweights=ddc.eweights)
        for i in partitioning[1]:
            f.write(str(i) + "\n")
