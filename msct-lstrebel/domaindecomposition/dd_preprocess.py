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
import warnings
import pickle

from pymetis import part_graph

from domain_decomposition import DomainSubdivision


class DomainPreprocess:
    def __init__(self, domain, periodic, subdivs_per_dim, fileoutput="", path="", prefix=""):
        self.domain = domain
        self.periodic = periodic
        self.subdivs_per_dim = subdivs_per_dim
        self.fileout = fileoutput
        self.path = path
        self.prefix = prefix

        self.subdivisions = []
        self.adjncy = []
        self.xadj = []
        self.vweights = []
        self.eweights = []
        self.edgecounter = 0
        self.alist = []

        self.total_subdivisions = 1
        self.subdivisions = []
        self.stencil_field_patterns = {}
        self.stencil_field_accesses = {}

    def add_stencil(self, stencil):
        # stencil is a dictionary of {fieldname: list of 6 lists (one for each direction)
        # containing the access patterns of the stencil patterns, next field : next list ...}
        # Add stencil pattern to the field, either concatenate with already existing pattern or create new one:
        for fieldname, stencil_pattern in stencil.items():
            if fieldname in self.stencil_field_patterns:
                for d in range(0, 6):
                    self.stencil_field_patterns[fieldname][d] = sorted(
                        (self.stencil_field_patterns[fieldname][d]
                         + list(set(stencil_pattern[d]) - set(self.stencil_field_patterns[fieldname][d]))))

                    self.stencil_field_accesses[fieldname][d] = len(self.stencil_field_patterns[fieldname][d])
            else:
                self.stencil_field_patterns[fieldname] = stencil_pattern.copy()

                self.stencil_field_accesses[fieldname] = stencil_pattern.copy()
                for d in range(0, 6):
                    self.stencil_field_accesses[fieldname][d] = len(stencil_pattern[d])

    def combined_accesses(self):
        total_accesses = np.zeros(6)
        for fieldname in self.stencil_field_accesses.keys():
            for d in range(0, 6):
                total_accesses[d] += self.stencil_field_accesses[fieldname][d]

        return total_accesses

    def halo_maximum_extent(self):
        halo_max = np.zeros(6)
        for fieldname in self.stencil_field_patterns.keys():
            for d in range(0, 6):
                halo_max[d] = max(halo_max[d], max(self.stencil_field_patterns[fieldname][d]))

        return halo_max

    def communication_cost_estimation(self, subdiv_size, stencil_extent):
        halo_sizes = np.zeros((stencil_extent.size))
        # halo_sizes[0] = subdiv_size[1] * subdiv_size[2] * stencil_extent[0]
        # halo_sizes[1] = subdiv_size[1] * subdiv_size[2] * stencil_extent[1]
        # halo_sizes[2] = subdiv_size[0] * subdiv_size[2] * stencil_extent[2]
        # halo_sizes[3] = subdiv_size[0] * subdiv_size[2] * stencil_extent[3]
        # halo_sizes[4] = subdiv_size[0] * subdiv_size[1] * stencil_extent[4]
        # halo_sizes[5] = subdiv_size[0] * subdiv_size[1] * stencil_extent[5]
        for e in range(stencil_extent.size):
            halo_sizes[e] = subdiv_size[((e // 2) - 1) % 3] * subdiv_size[((e // 2) + 1) % 3] * stencil_extent[e]

        return halo_sizes

    def computational_cost_estimation(self, subdiv_gridpoints):
        return subdiv_gridpoints

    def preprocess(self):
        subdiv_size = self.domain // self.subdivs_per_dim
        assert (np.alltrue(self.domain % self.subdivs_per_dim == 0)), ("Subdivisions per dimension is not"
                                                                       " a factor of the given domain size.")
        subdiv_gridpoints = 1
        for e in subdiv_size:
            subdiv_gridpoints *= e

        for e in self.subdivs_per_dim:
            self.total_subdivisions *= e

        halo_max = self.halo_maximum_extent()
        for e in range(len(halo_max)):
            if halo_max[e] > subdiv_size[e // 2]:
                warnings.warn("Stencil extents into multiple subdivisions", RuntimeWarning)

        stencil_extent = self.combined_accesses()
        comm_cost = self.communication_cost_estimation(subdiv_size, stencil_extent)

        comp_cost = self.computational_cost_estimation(subdiv_gridpoints)

        for i in range(self.subdivs_per_dim[0]):
            for j in range(self.subdivs_per_dim[1]):
                for k in range(self.subdivs_per_dim[2]):
                    ind = (i * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + k

                    global_range = np.array([i * subdiv_size[0], (i + 1) * subdiv_size[0],
                                             j * subdiv_size[1], (j + 1) * subdiv_size[1],
                                             k * subdiv_size[2], (k + 1) * subdiv_size[2]])

                    # End of Domain in negative X direction
                    if i == 0:
                        if self.periodic[0]:
                            negx = (((self.subdivs_per_dim[0] - 1) * self.subdivs_per_dim[1] + j)
                                    * self.subdivs_per_dim[2] + k)
                        else:
                            negx = None
                    else:
                        negx = ((i - 1) * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + k

                    # End of Domain in positive X direction
                    if i == self.subdivs_per_dim[0] - 1:
                        if self.periodic[0]:
                            posx = (0 * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + k
                        else:
                            posx = None
                    else:
                        posx = ((i + 1) * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + k

                    # End of Domain in negative Y direction
                    if j == 0:
                        if self.periodic[1]:
                            negy = ((i * self.subdivs_per_dim[1] + self.subdivs_per_dim[1] - 1)
                                    * self.subdivs_per_dim[2] + k)
                        else:
                            negy = None
                    else:
                        negy = (i * self.subdivs_per_dim[1] + j - 1) * self.subdivs_per_dim[2] + k

                    # End of Domain in positive Y direction
                    if j == self.subdivs_per_dim[1] - 1:
                        if self.periodic[1]:
                            posy = (i * self.subdivs_per_dim[1] + 0) * self.subdivs_per_dim[2] + k
                        else:
                            posy = None
                    else:
                        posy = (i * self.subdivs_per_dim[1] + j + 1) * self.subdivs_per_dim[2] + k

                    # End of Domain in negative Z direction
                    if k == 0:
                        if self.periodic[2]:
                            negz = ((i * self.subdivs_per_dim[1] + j)
                                    * self.subdivs_per_dim[2] + self.subdivs_per_dim[2] - 1)
                        else:
                            negz = None
                    else:
                        negz = (i * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + k - 1

                    # End of Domain in positive Z direction
                    if k == self.subdivs_per_dim[2] - 1:
                        if self.periodic[2]:
                            posz = (i * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + 0
                        else:
                            posz = None
                    else:
                        posz = (i * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + k + 1

                    if negx == ind or posx == ind or negy == ind or posy == ind or negz == ind or posz == ind:
                        warnings.warn("Due to periodicity at least one subdivision"
                                      " is its own neighbor." + str([negx, posx, negy, posy, negz, posz]),
                                      RuntimeWarning)

                    nindex = [negx, posx, negy, posy, negz, posz]

                    self.subdivisions.append(DomainSubdivision(id=ind,
                                                               pid=0,
                                                               size=subdiv_size,
                                                               global_coords=global_range,
                                                               neighbors_id=nindex))

                    self.vweights.append(int(comp_cost))
                    self.xadj.append(int(self.edgecounter))
                    for e in range(len(nindex)):
                        if nindex[e] is not None:
                            self.edgecounter += 1
                            self.adjncy.append(int(nindex[e]))
                            self.eweights.append(int(comm_cost[e]))

        with open(self.path + self.prefix + "subdivisions.pkl", "wb") as f:
            pickle.dump(self.subdivisions, f)

        if self.fileout == "metis":
            self.write_to_file_metis_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            self.path + self.prefix + "subdomains")
        elif self.fileout == "scotch":
            self.write_to_file_scotch_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            self.path + self.prefix + "subdomains")
        elif self.fileout == "both":
            self.write_to_file_metis_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            self.path + self.prefix + "subdomains")
            self.write_to_file_scotch_format(self.adjncy,
                                             self.xadj,
                                             self.vweights,
                                             self.eweights,
                                             self.edgecounter,
                                             self.path + self.prefix + "subdomains")

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

    def prepare_for_pymetis(self):
        self.adjncy.append(int(self.total_subdivisions))
        self.xadj.append(int(self.edgecounter))

    def pymetis_partitioning(self, nparts, verbose=False):
        self.prepare_for_pymetis()

        partitioning = part_graph(nparts,
                                  xadj=self.xadj,
                                  adjncy=self.adjncy,
                                  vweights=self.vweights,
                                  eweights=self.eweights)

        with open(self.path + self.prefix + "subdomains_pymetis.dat.part." + str(nparts), "w") as f:
            for i in partitioning[1]:
                f.write(str(i) + "\n")

        if verbose:
            print(partitioning[1])