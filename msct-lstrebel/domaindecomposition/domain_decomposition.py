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

# from mpi4py import MPI
from pymetis import part_graph

import gridtools as gt
from gridtools.user_interface.mode import Mode
from gridtools.user_interface.vertical_direction import VerticalDirection


class DomainPartitions:
    domain_partitions = None

    @staticmethod
    def load_partitions(fileinput, fileformat):
        if fileformat == "metis":
            DomainPartitions.domain_partitions = DomainPartitions.load_from_metis_file(fileinput)
        elif fileformat == "scotch":
            DomainPartitions.domain_partitions = DomainPartitions.load_from_scotch_file(fileinput)
        else:
            print("Only 'metis' or 'scotch' as fileformat accepted.")

    @staticmethod
    def load_from_metis_file(fileinput):
        return np.loadtxt(fileinput, dtype=int)

    @staticmethod
    def load_from_scotch_file(fileinput):
        return np.loadtxt(fileinput, dtype=int, skiprows=1, usecols=1)

    @staticmethod
    def print_partitions():
        print(DomainPartitions.domain_partitions)


class DomainSubdivision:
    def __init__(self, id, pid, size, halo, gridpoints, neighbors):
        self.id = id
        self.partitions_id = pid
        # self.global_coords = global_coords
        self.size = size
        self.halo = halo
        self.gridpoints = gridpoints
        self.neighbors = neighbors
        self.sub_stencils = []
        self.fields = {}

    def register_stencil(self, **kwargs):
        # Set default values
        definitions_func = inputs = outputs = domain = None
        constant_inputs = global_inputs = {}
        mode = Mode.DEBUG
        vertical_direction = VerticalDirection.PARALLEL

        # Read keyword arguments
        for key in kwargs:
            if key == "definitions_func":
                definitions_func = kwargs[key]
            elif key == "inputs":
                inputs = kwargs[key]
            elif key == "constant_inputs":
                constant_inputs = kwargs[key]
            elif key == "global_inputs":
                global_inputs = kwargs[key]
            elif key == "outputs":
                outputs = kwargs[key]
            elif key == "domain":
                domain = kwargs[key]
            elif key == "mode":
                mode = kwargs[key]
            elif key == "vertical_direction":
                vertical_direction = kwargs[key]
            else:
                raise ValueError("\n  NGStencil accepts the following keyword arguments: \n"
                                 "  - definitions_func, \n"
                                 "  - inputs, \n"
                                 "  - constant_inputs [default: {}], \n"
                                 "  - global_inputs [default: {}], \n"
                                 "  - outputs, \n"
                                 "  - domain, \n"
                                 "  - mode [default: DEBUG], \n"
                                 "  - vertical_direction [default: PARALLEL]. \n"
                                 "  The order does not matter.")
        fields_in = {}
        fields_out = {}

        for k, v in inputs.items():
            fields_in[k] = self.fields[v]
        for k, v in outputs.items():
            fields_out[k] = self.fields[v]

        stencil = gt.NGStencil(
            definitions_func=definitions_func,
            inputs=fields_in,
            constant_inputs=constant_inputs,
            global_inputs=global_inputs,
            outputs=fields_out,
            domain=domain,
            mode=mode,
            vertical_direction=vertical_direction)

        self.sub_stencils.append(stencil)

    def communicate(self):
        for n in range(len(self.neighbors)):
            # Check if neighbor is local or external
            if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors[n]]:
                # local exchange
                # Exchange halos for each stencil
                for f in self.fields.values():
                    pass
            else:
                # external exchange
                # Exchange halos for each field
                for f in self.fields.values():
                    pass

    def exchange_locally(self):
        pass

    def exchange_externally(self):
        pass

    def communicate_two_way(self):
        pass

    def communicate_one_way(self):
        pass

    def compute(self):
        for s in self.sub_stencils:
            s.compute()


class DomainPreprocess:
    def __init__(self, domain, periodic, subdivs_per_dim, stencil_extent, fileoutput=""):
        self.domain = domain
        self.periodic = periodic
        self.subdivs_per_dim = subdivs_per_dim
        self.stencil_extent = stencil_extent
        self.subdivisions = []

        self.preprocess(fileoutput)

    def communication_cost_estimation(self, subdiv_size, stencil_extent):
        halo_sizes = np.zeros((self.stencil_extent.size))
        # halo_sizes[0] = subdiv_size[1] * subdiv_size[2] * stencil_extent[0]
        # halo_sizes[1] = subdiv_size[1] * subdiv_size[2] * stencil_extent[1]
        # halo_sizes[2] = subdiv_size[0] * subdiv_size[2] * stencil_extent[2]
        # halo_sizes[3] = subdiv_size[0] * subdiv_size[2] * stencil_extent[3]
        # halo_sizes[4] = subdiv_size[0] * subdiv_size[1] * stencil_extent[4]
        # halo_sizes[5] = subdiv_size[0] * subdiv_size[1] * stencil_extent[5]
        for e in range(self.stencil_extent.size):
            halo_sizes[e] = subdiv_size[((e // 2) - 1) % 3] * subdiv_size[((e // 2) + 1) % 3] * stencil_extent[e]

        return halo_sizes

    def computational_cost_estimation(self, subdiv_gridpoints):
        return subdiv_gridpoints

    def preprocess(self, fileoutput=""):
        subdiv_size = self.domain / self.subdivs_per_dim
        subdiv_gridpoints = 1
        for e in subdiv_size:
            subdiv_gridpoints *= e

        self.total_subdivisions = 1
        for e in self.subdivs_per_dim:
            self.total_subdivisions *= e

        comm_cost = self.communication_cost_estimation(subdiv_size, self.stencil_extent)

        comp_cost = self.computational_cost_estimation(subdiv_gridpoints)
        halos = comp_cost

        for e in range(self.stencil_extent.size):
            if self.stencil_extent[e] > subdiv_size[e // 2]:
                warnings.warn("Stencil extents into multiple subdivisions", RuntimeWarning)

        self.subdivisions = []

        self.adjncy = []
        self.xadj = []
        self.vweights = []
        self.eweights = []
        self.edgecounter = 0
        self.alist = []

        for i in range(self.subdivs_per_dim[0]):
            for j in range(self.subdivs_per_dim[1]):
                for k in range(self.subdivs_per_dim[2]):
                    ind = (i * self.subdivs_per_dim[1] + j) * self.subdivs_per_dim[2] + k
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
                        warnings.warn("Due to periodicity at least one subdivision is its own neighbor.", RuntimeWarning)

                    nindex = [negx, posx, negy, posy, negz, posz]

                    self.subdivisions.append(DomainSubdivision(ind, ind, subdiv_size, halos,
                                                               subdiv_gridpoints, nindex))

                    self.vweights.append(int(comp_cost))
                    self.xadj.append(int(self.edgecounter))
                    for e in range(len(nindex)):
                        if nindex[e] is not None:
                            self.edgecounter += 1
                            self.adjncy.append(int(nindex[e]))
                            self.eweights.append(int(comm_cost[e]))

        with open("subdivisions.pkl", "wb") as f:
            pickle.dump(self.subdivisions, f)

        if fileoutput == "metis":
            self.write_to_file_metis_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            "subdomains")
        elif fileoutput == "scotch":
            self.write_to_file_scotch_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            "subdomains")
        elif fileoutput == "both":
            self.write_to_file_metis_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            "subdomains")
            self.write_to_file_scotch_format(self.adjncy,
                                             self.xadj,
                                             self.vweights,
                                             self.eweights,
                                             self.edgecounter,
                                             "subdomains")

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

    def pymetis_partitioning(self, nparts):
        self.prepare_for_pymetis()

        partitioning = part_graph(nparts, xadj=self.xadj, adjncy=self.adjncy, vweights=self.vweights, eweights=self.eweights)

        with open("subdomains_pymetis.dat.part." + str(nparts), "w") as f:
            for i in partitioning[1]:
                f.write(str(i) + "\n")


class DomainDecomposition:
    def __init__(self, preprocess=True,
                 domain=None,
                 periodic=None,
                 subdivs_per_dim=None,
                 stencil_extent=None,
                 nparts=None,
                 fileoutput="",
                 fileinput=None,
                 fileinputformat=None):
        if preprocess:
            if domain is None or periodic is None or subdivs_per_dim is None or stencil_extent is None or nparts is None:
                raise ValueError("Preprocess needs: domain, periodic, subdivs_per_dim, stencil_extent, and nparts.")
            self.subdivisions = []
            preproc = DomainPreprocess(domain, periodic, subdivs_per_dim, stencil_extent, fileoutput)
            preproc.preprocess(fileoutput)
            preproc.pymetis_partitioning(nparts)
        else:
            self.subdivisions = self.load_subdivisions()

            if fileinput is None or fileinputformat is None:
                raise ValueError("Need fileinput and fileinputformat for partitioning file.")

            DomainPartitions.load_partitions(fileinput, fileinputformat)

            for s in range(len(self.subdivisions)):
                self.subdivisions[s].partitions_id = DomainPartitions.domain_partitions[s]

            # Remove subdivisions of other partitions from list
            # TODO This needs to be modified for MPI
            # this_partition = 0
            # temp_list = []
            # for sd in self.subdivisions:
            #     if sd.partitions_id == this_partition:
            #         temp_list.append(sd)
            # self.subdivisions = temp_list

    def load_subdivisions(self):
        with open("subdivisions.pkl", "rb") as f:
            return pickle.load(f)

    def register_stencil(self, **kwargs):
        for sd in self.subdivisions:
            sd.register_stencil(**kwargs)

    def compute(self):
        for sd in self.subdivisions:
            sd.compute()

    def communicate(self):
        for sd in self.subdivisions:
            sd.communicate()
