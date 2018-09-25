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
This module contains the domain decomposition pre-process functionality.
Specifically, the creation of a graph partitioning source graph based on stencil patterns,
saving the source graph in a specific format, and calling the PyMetis library to carry out the graph partitioning.

"""

import numpy as np
import warnings
import pickle

from pymetis import part_graph

from domain_decomposition import DomainSubdivision


class DomainPreprocess:
    """ This class contains all the functions needed for the domain decomposition pre-process.

    """
    def __init__(self, domain, periodic, subdivs_per_dim, fileoutput="", path="", prefix=""):
        """ Initialize the DomainPreprocess class with the necessary parameters and options.

        :param domain: Array or List with 3 entries corresponding to the domain size
        in x-direction, y-direction, and z-direction.
        :param periodic: Array or List with 3 entries corresponding to the periodicity
        in x-direction, y-direction, and z-direction. An entry of 0 means non-periodic and an entry of 1 means periodic.
        :param subdivs_per_dim: Array or List with 3 entries corresponding to the number of subdivisions
        in x-direction, y-direction, and z-direction.
        The number of subdivisions needs to be a factor of the corresponding domain size.
        :param fileoutput: Optional: Can be "metis", "scotch", or "both".
        If provided saves the created source graph in the corresponding file format.
        :param path: Optional: Path where all the output should be saved.
        :param prefix: Optional: Prefix for all output file names.
        """
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
        """ Function to add stencil patterns to the DomainPreprocess class needed to generate the correct source graph.

        Concatenates the access pattern with previously added stencils.

        :param stencil: A dictionary of {fieldname: list of 6 lists (one for each direction)
        containing the access patterns of the stencil patterns, next field : next list ...}
        :return: None
        """
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
        """ Helper function used to collect and combine all access patterns.

        :return: total_accesses: Array of size 6 containing the total accesses in each direction.
        """
        total_accesses = np.zeros(6)
        for fieldname in self.stencil_field_accesses.keys():
            for d in range(0, 6):
                total_accesses[d] += self.stencil_field_accesses[fieldname][d]

        return total_accesses

    def halo_maximum_extent(self):
        """ Helper function to determine the maximum of the stencil patterns in each direction.

        :return: halo_max: Array of size 6 containing the maximum extent of the stencils in each direction.
        """
        halo_max = np.zeros(6)
        for fieldname in self.stencil_field_patterns.keys():
            for d in range(0, 6):
                halo_max[d] = max(halo_max[d], max(self.stencil_field_patterns[fieldname][d]))

        return halo_max

    def communication_cost_estimation(self, subdiv_size, stencil_extent):
        """ Communication cost estimation function.

        :param subdiv_size: List or Array of size 3 containing the size of the subdivisions in each dimension.
        :param stencil_extent: List or Array of size 6 containing the maximum extent of the stencils in each direction.
        :return: halo_sizes: Array of size 6 containing the size of the halo for each direction.
        """
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
        """ Computational cost estimation function.

        For simplicity computational cost estimation is proportional to number of grid points in the subdivision.

        :param subdiv_gridpoints: Number of grid points in each subdivision.
        :return: Computational cost estimation.
        """
        return subdiv_gridpoints

    def preprocess(self):
        """ Main pre-process function.
        Uses the initialized parameters and options to generate a graph partitioning source graph.

        Source graph is generated by iterating through the subdivisions in each direction,
        determine the neighbor index,
        or setting it to "None" if the subdivision is on the global boundary and non-periodic.
        Vertex and edge weights are determined by the estimation functions.

        Saves the subdivisions in the classes subdivisions list,
        and serializes them using pickle into the subdivisions.pkl file.
        Saves the source graph in the classes adjncy, xadj, vweights, and eweights lists,
        adhering to the PyMetis naming convention.
        If initialized with the option saves the source graph in "metis", "scotch", or "both" formats to file.

        :return: None
        """
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

        # Since all subdivisions are by design uniform normalize the computational cost to 1.
        # Needed for large subdivisions to avoid integer overflow
        # of total vertex weight (max for 32 bit PyMetis 2147483647)
        # Total edge weight has same limit but edge weights should not get as large as the computational cost
        comp_cost = 1 #self.computational_cost_estimation(subdiv_gridpoints)

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
        """ Helper function to write the source graph to file in the metis format:
        Header:
        One Line: "Number of Vertices" "Number of Edges (counted once)" "3 digit binary flag"

        Body:
        Each vertex line: s w_1 w_2 ... w_ncon v_1 e_1 v_2 e_2 ... v_k e_k
        s: size of vertex
        w_* : weight of vertex
        v_* : neighbor vertex index
        e_* : edge weight to neighbor

        Vertex numbering starts with 1 not 0!

        :return: None
        """
        header = "{0:d} {1:d} {2:03d} \n".format(len(xadj), edgecounter//2, flag)

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
        """ Helper function to write the source graph to file in the scotch format:
        Header:
        First line: graph file version number
        Second line: number of vertices followed by number of arcs (edge number twice counted)
        Third line: graph base index value (0 or 1) and numeric flag

        Body:
        Each vertex line: [vertex label] [vertex load] vertex_degree [arc_load] arc_end_vertex

        :return: None
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
        """ Small helper function to add end point to adjacency lists needed by PyMetis.

        :return: None
        """
        self.adjncy.append(int(self.total_subdivisions))
        self.xadj.append(int(self.edgecounter))

    def pymetis_partitioning(self, nparts, verbose=False):
        """ Partioning function calls PyMetis library to graph partition prepared source graph.

        :param nparts: Number of partitions the graph should be partitioned into.
        Corresponds to number of processing units that will carry out the computations.
        :param verbose: Optional: binary flag if True shows the partitioning results in the standard output.
        :return: None
        """
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