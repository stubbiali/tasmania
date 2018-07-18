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

from mpi4py import MPI
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
    def __init__(self, id, pid, size, global_coords, gridpoints, neighbors_id):
        self.id = id
        self.partitions_id = pid
        self.global_coords = global_coords
        self.size = size
        self.gridpoints = gridpoints
        self.neighbors_id = neighbors_id
        self.neighbor_list = None
        self.fields = {}
        self.halos = {}
        self.recv_slices = {}
        self.get_slices = {}

    def check_globalcoords(self, i, j, k):
        return (self.global_coords[0] <= i < self.global_coords[1]
                and self.global_coords[2] <= j < self.global_coords[3]
                and self.global_coords[4] <= k < self.global_coords[5])

    def register_field(self, fieldname, halo, field_ic_file=None):
        self.halos[fieldname] = halo
        self.fields[fieldname] = np.zeros((self.size[0] + halo[0] + halo[1],
                                           self.size[1] + halo[2] + halo[3],
                                           self.size[2] + halo[4] + halo[5]))
        if field_ic_file is not None:
            self.fields[fieldname][halo[0]:None if halo[1] == 0 else -halo[1],
                                   halo[2]:None if halo[3] == 0 else -halo[3],
                                   halo[4]:None if halo[5] == 0 else -halo[5]] = np.load(
                field_ic_file, mmap_mode='r')[self.global_coords[0]:self.global_coords[1],
                                              self.global_coords[2]:self.global_coords[3],
                                              self.global_coords[4]:self.global_coords[5]]

        self.setup_slices(fieldname)

    def setup_slices(self, fieldname):
        self.recv_slices[fieldname] = [
            # Halo region in negative x direction
            np.s_[0:self.halos[fieldname][0],
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Halo region in positive x direction
            np.s_[-self.halos[fieldname][1]:,
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Halo region in negative y direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  0:self.halos[fieldname][2],
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Halo region in positive y direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  -self.halos[fieldname][3]:,
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Halo region in negative z direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  0:self.halos[fieldname][4]],
            # Halo region in positive z direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  -self.halos[fieldname][5]:]
        ]

        self.get_slices[fieldname] = [
            # Overlap region in neighboring, positive x direction
            np.s_[-(self.halos[fieldname][1]+self.halos[fieldname][1]):-self.halos[fieldname][1],
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Overlap region in neighboring, negative x direction
            np.s_[self.halos[fieldname][0]:self.halos[fieldname][0]+self.halos[fieldname][0],
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Overlap region in neighboring, positive y direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  -(self.halos[fieldname][3]+self.halos[fieldname][3]):-self.halos[fieldname][3],
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Overlap region in neighboring, negative y direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  self.halos[fieldname][2]:self.halos[fieldname][2]+self.halos[fieldname][2],
                  self.halos[fieldname][4]:-self.halos[fieldname][5]],
            # Overlap region in neighboring, positive z direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  -(self.halos[fieldname][5] + self.halos[fieldname][5]):-self.halos[fieldname][5]],
            # Overlap region in neighboring, negative z direction
            np.s_[self.halos[fieldname][0]:-self.halos[fieldname][1],
                  self.halos[fieldname][2]:-self.halos[fieldname][3],
                  self.halos[fieldname][4]:self.halos[fieldname][4]+self.halos[fieldname][4]]
        ]

    def save_fields(self):
        for k in self.fields.keys():
            filename = (str(k) + "_from_"
                        + "x" + str(self.global_coords[0])
                        + "y" + str(self.global_coords[2])
                        + "z" + str(self.global_coords[4])
                        + "_to_"
                        + "x" + str(self.global_coords[1] - 1)
                        + "y" + str(self.global_coords[3] - 1)
                        + "z" + str(self.global_coords[5] - 1))
            np.save(filename, self.get_interior_field(k))

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

        # Use the inputs/outputs as names of the field to instantiate the stencil with subdivision fields
        fields_in = {}
        fields_out = {}

        for k, v in inputs.items():
            fields_in[k] = self.fields[v]
        for k, v in outputs.items():
            fields_out[k] = self.fields[v]

        # Change the domain to the subdivision rectangle domain with maximum halo
        ulx = uly = ulz = drx = dry = drz = 0

        for k in self.fields.keys():
            ulx = max(ulx, self.halos[k][0])
            uly = max(uly, self.halos[k][1])
            ulz = max(ulz, self.halos[k][2])
            drx = max(drx, self.halos[k][3])
            dry = max(dry, self.halos[k][4])
            drz = max(drz, self.halos[k][5])

        drx = self.size[0] + ulx - drx + 1
        dry = self.size[1] + uly - dry + 1
        drz = self.size[2] + ulz - drz + 1

        domain = gt.domain.Rectangle((ulx, uly, ulz), (drx, dry, drz))

        # Instantiate the stencil with the changed subdivision inputs
        stencil = gt.NGStencil(
            definitions_func=definitions_func,
            inputs=fields_in,
            constant_inputs=constant_inputs,
            global_inputs=global_inputs,
            outputs=fields_out,
            domain=domain,
            mode=mode,
            vertical_direction=vertical_direction)

        return stencil

    def get_local_neighbor(self, id):
        temp_sd = None
        for sd in self.neighbor_list:
            if id == sd.id:
                temp_sd = sd
        return temp_sd

    def get_interior_field(self, fieldname):
        return self.fields[fieldname][
               self.halos[fieldname][0]:-self.halos[fieldname][1],
               self.halos[fieldname][2]:-self.halos[fieldname][3],
               self.halos[fieldname][4]:-self.halos[fieldname][5]]

    def swap_fields(self, field1, field2):
        self.fields[field1], self.fields[field2] = self.fields[field2], self.fields[field1]

    def communicate(self, fieldname=None):
        if fieldname is None:
            for k in self.halos.keys():
                self.communicate_field(k)
        else:
            self.communicate_field(fieldname)

    def communicate_field(self, fieldname):
        # Iterate over all neighbors i.e. all directions:
        for d in range(len(self.neighbors_id)):
            # Check if neighbor in current direction is the global boundary:
            if self.check_global_boundary(d):
                pass
            else:
                # Check if neighbor in current direction is local or external and communicate accordingly:
                if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[d]]:
                    self.communicate_local(fieldname,
                                           self.recv_slices[fieldname][d],
                                           self.get_slices[fieldname][d],
                                           d)
                else:
                    self.communicate_external(fieldname,
                                              self.recv_slices[fieldname][d],
                                              self.get_slices[fieldname][d],
                                              d)

        # # Check if negx neighbor is local or external
        # if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[0]]:
        #     self.communicate_local_negx(fieldname)
        # else:
        #     pass
        #
        # # Check if posx neighbor is local or external
        # if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[1]]:
        #     self.communicate_local_posx(fieldname)
        # else:
        #     pass
        #
        # # Check if negy neighbor is local or external
        # if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[2]]:
        #     self.communicate_local_negy(fieldname)
        # else:
        #     pass
        #
        # # Check if posy neighbor is local or external
        # if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[3]]:
        #     self.communicate_local_posy(fieldname)
        # else:
        #     pass
        #
        # # Check if negz neighbor is local or external
        # if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[4]]:
        #     self.communicate_local_negz(fieldname)
        # else:
        #     pass
        #
        # # Check if posz neighbor is local or external
        # if self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[5]]:
        #     self.communicate_local_posz(fieldname)
        # else:
        #     pass

    def communicate_local(self, fieldname, recv_slice, get_slice, neighbor_id):
        # Numpy view of the halo region
        recv = self.fields[fieldname][recv_slice]
        # Neighboring subdivision
        neighbor_sd = self.get_local_neighbor(self.neighbors_id[neighbor_id])
        # Overlap region in neighboring subdivision
        get = neighbor_sd.fields[fieldname][get_slice]
        # Transfer overlap region into halo region
        recv[:] = get[:]

    # def communicate_local_negx(self, fieldname):
    #     recv_negx = self.fields[fieldname][
    #                 0:self.halos[fieldname][0],
    #                 self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                 self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     negx_sd = self.get_local_neighbor(self.neighbors_id[0])
    #
    #     get_posx = negx_sd.fields[fieldname][
    #                -(self.halos[fieldname][1]+self.halos[fieldname][1]):-self.halos[fieldname][1],
    #                self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     recv_negx[:] = get_posx[:]
    #
    # def communicate_local_posx(self, fieldname):
    #     recv_posx = self.fields[fieldname][
    #                 -self.halos[fieldname][1]:,
    #                 self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                 self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     posx_sd = self.get_local_neighbor(self.neighbors_id[1])
    #
    #     get_negx = posx_sd.fields[fieldname][
    #                self.halos[fieldname][0]:self.halos[fieldname][0]+self.halos[fieldname][0],
    #                self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     recv_posx[:] = get_negx[:]
    #
    # def communicate_local_negy(self, fieldname):
    #     recv_negy = self.fields[fieldname][
    #                 self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                 0:self.halos[fieldname][2],
    #                 self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     negy_sd = self.get_local_neighbor(self.neighbors_id[2])
    #
    #     get_posy = negy_sd.fields[fieldname][
    #                self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                -(self.halos[fieldname][3]+self.halos[fieldname][3]):-self.halos[fieldname][3],
    #                self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     recv_negy[:] = get_posy[:]
    #
    # def communicate_local_posy(self, fieldname):
    #     recv_posy = self.fields[fieldname][
    #                 self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                 -self.halos[fieldname][3]:,
    #                 self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     posy_sd = self.get_local_neighbor(self.neighbors_id[3])
    #
    #     get_negy = posy_sd.fields[fieldname][
    #                self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                self.halos[fieldname][2]:self.halos[fieldname][2]+self.halos[fieldname][2],
    #                self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #
    #     recv_posy[:] = get_negy[:]
    #
    # def communicate_local_negz(self, fieldname):
    #     recv_negz = self.fields[fieldname][
    #                 self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                 self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                 0:self.halos[fieldname][4]]
    #
    #     negz_sd = self.get_local_neighbor(self.neighbors_id[4])
    #
    #     get_posz = negz_sd.fields[fieldname][
    #                self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                -(self.halos[fieldname][5] + self.halos[fieldname][5]):-self.halos[fieldname][5]]
    #
    #     recv_negz[:] = get_posz[:]
    #
    # def communicate_local_posz(self, fieldname):
    #     recv_posz = self.fields[fieldname][
    #                 self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                 self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                 -self.halos[fieldname][5]:]
    #
    #     posz_sd = self.get_local_neighbor(self.neighbors_id[5])
    #
    #     get_negz = posz_sd.fields[fieldname][
    #                self.halos[fieldname][0]:-self.halos[fieldname][1],
    #                self.halos[fieldname][2]:-self.halos[fieldname][3],
    #                self.halos[fieldname][4]:self.halos[fieldname][4]+self.halos[fieldname][4]]
    #
    #     recv_posz[:] = get_negz[:]
    #
    # def communicate_field_old(self, fieldname):
    #     reqs = [None, None, None, None, None, None, None, None, None, None, None, None]
    #     original_shape = self.fields[fieldname].shape
    #     self.fields[fieldname] = self.fields[fieldname].reshape(-1)
    #
    #     halo0 = MPI.DOUBLE.Create_vector(count=original_shape[0]*self.halos[fieldname][0], blocklength=1,
    #                                      stride=original_shape[1] * original_shape[2])
    #     halo0.Commit()
    #
    #     halo1 = MPI.DOUBLE.Create_vector(count=original_shape[0]*self.halos[fieldname][1], blocklength=1,
    #                                      stride=original_shape[1] * original_shape[2])
    #     halo1.Commit()
    #
    #     halo2 = MPI.DOUBLE.Create_vector(count=original_shape[1]*self.halos[fieldname][2], blocklength=1,
    #                                      stride=original_shape[2])
    #     halo2.Commit()
    #
    #     halo3 = MPI.DOUBLE.Create_vector(count=original_shape[1]*self.halos[fieldname][3], blocklength=1,
    #                                      stride=original_shape[2])
    #     halo3.Commit()
    #
    #     halo4 = MPI.DOUBLE.Create_vector(count=original_shape[2]*self.halos[fieldname][4], blocklength=1,
    #                                      stride=1)
    #     halo4.Commit()
    #
    #     halo5 = MPI.DOUBLE.Create_vector(count=original_shape[2]*self.halos[fieldname][5], blocklength=1,
    #                                      stride=1)
    #     halo5.Commit()
    #     # Isend(self, buf, int dest, int tag=0)
    #
    #     if MPI.COMM_WORLD.Get_rank() % 2 == 0:
    #         # Send neg x halo to neighbor
    #         start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #                       + self.halos[fieldname][2]) * original_shape[2] + self.halos[fieldname][4])
    #
    #         reqs[0] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo0],
    #                                            dest=self.neighbors[0])
    #
    #         # Receive pos x halo from neighbor
    #         start_ind = (((self.halos[fieldname][0]) * original_shape[1]
    #                       + original_shape[1] - self.halos[fieldname][2])
    #                      * original_shape[2] + self.halos[fieldname][4])
    #
    #         reqs[1] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo1],
    #                                            source=self.neighbors[1])
    #
    #         # Send pos x halo to neighbor
    #         start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #                       + original_shape[1] - self.halos[fieldname][2] - self.halos[fieldname][3])
    #                      * original_shape[2] + self.halos[fieldname][4])
    #         reqs[2] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo1],
    #                                            dest=self.neighbors[0])
    #
    #         # Receive neg x halo from neighbor
    #         start_ind = ((self.halos[fieldname][0] * original_shape[1]) * original_shape[2] + self.halos[fieldname][4])
    #         reqs[3] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo0],
    #                                            source=self.neighbors[1])
    #
    #         # # Send neg y halo to neighbor
    #         # start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #         #               + self.halos[fieldname][2]) * original_shape[2] + self.halos[fieldname][4])
    #         #
    #         # reqs[4] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo2],
    #         #                                    dest=self.neighbors[0])
    #         #
    #         # # Receive pos y halo from neighbor
    #         # start_ind = (((original_shape[0] - self.halos[fieldname][0]) * original_shape[1]
    #         #               + self.halos[fieldname][2])
    #         #              * original_shape[2] + self.halos[fieldname][4])
    #         # reqs[5] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo3],
    #         #                                source=self.neighbors[1])
    #         #
    #         # # Send pos y halo to neighbor
    #         # start_ind = (((original_shape[0] - self.halos[fieldname][0] - self.halos[fieldname][1]) * original_shape[1]
    #         #               + self.halos[fieldname][2]) * original_shape[2] + self.halos[fieldname][4])
    #         #
    #         # reqs[6] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo3],
    #         #                                    dest=self.neighbors[0])
    #         #
    #         # # Receive neg y halo from neighbor
    #         # start_ind = ((0 * original_shape[1]
    #         #               + self.halos[fieldname][2])
    #         #              * original_shape[2] + self.halos[fieldname][4])
    #         # reqs[7] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo2],
    #         #                                source=self.neighbors[1])
    #         # # Send neg z halo to neighbor
    #         # start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #         #               + self.halos[fieldname][2]) * original_shape[2] + self.halos[fieldname][4])
    #         #
    #         # reqs[8] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo4],
    #         #                                    dest=self.neighbors[0])
    #     else:
    #         # Receive pos x halo from neighbor
    #         start_ind = (((self.halos[fieldname][0]) * original_shape[1]
    #                       + original_shape[1] - self.halos[fieldname][2])
    #                      * original_shape[2] + self.halos[fieldname][4])
    #         reqs[0] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo1],
    #                                        source=self.neighbors[1])
    #
    #         # Send neg x halo to neighbor
    #         start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #                       + self.halos[fieldname][2]) * original_shape[2] + self.halos[fieldname][4])
    #         reqs[1] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo0],
    #                                            dest=self.neighbors[0])
    #
    #         # Receive neg x halo from neighbor
    #         start_ind = ((self.halos[fieldname][0] * original_shape[1]) * original_shape[2] + self.halos[fieldname][4])
    #         reqs[2] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo0],
    #                                            source=self.neighbors[1])
    #
    #         # Send pos x halo to neighbor
    #         start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #                       + original_shape[1] - self.halos[fieldname][2] - self.halos[fieldname][3])
    #                      * original_shape[2] + self.halos[fieldname][4])
    #         reqs[3] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo1],
    #                                            dest=self.neighbors[0])
    #
    #         # # Receive pos y halo from neighbor
    #         # start_ind = (((original_shape[0] - self.halos[fieldname][0]) * original_shape[1]
    #         #               + self.halos[fieldname][2])
    #         #              * original_shape[2] + self.halos[fieldname][4])
    #         # reqs[4] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo3],
    #         #                                source=self.neighbors[1])
    #         #
    #         # # Send neg y halo to neighbor
    #         # start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #         #               + self.halos[fieldname][2]) * original_shape[2] + self.halos[fieldname][4])
    #         #
    #         # reqs[5] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo2],
    #         #                                    dest=self.neighbors[0])
    #         #
    #         # # Receive neg y halo from neighbor
    #         # start_ind = ((0 * original_shape[1]
    #         #               + self.halos[fieldname][2])
    #         #              * original_shape[2] + self.halos[fieldname][4])
    #         # reqs[6] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo2],
    #         #                                source=self.neighbors[1])
    #         #
    #         # # Send pos y halo to neighbor
    #         # start_ind = (((original_shape[0] - self.halos[fieldname][0] - self.halos[fieldname][1]) * original_shape[1]
    #         #               + self.halos[fieldname][2]) * original_shape[2] + self.halos[fieldname][4])
    #         #
    #         # reqs[7] = MPI.COMM_WORLD.Isend([self.fields[fieldname][start_ind:], 1, halo3],
    #         #                                    dest=self.neighbors[0])
    #
    #         # # Receive pos z halo from neighbor
    #         # start_ind = ((self.halos[fieldname][0] * original_shape[1]
    #         #               + original_shape[1] - self.halos[fieldname][2])
    #         #              * original_shape[2])
    #         # reqs[8] = MPI.COMM_WORLD.Irecv([self.fields[fieldname][start_ind:], 1, halo5],
    #         #                                source=self.neighbors[1])
    #
    #
    #     MPI.Request.waitall(reqs[:4])
    #     print(MPI.COMM_WORLD.Get_rank(), self.fields[fieldname][:].reshape(original_shape).transpose(), "\n")
    #
    #
    #     # # These are the halos to receive: neg x, pos x, neg y, pos y, neg z, pos z
    #     # self.fields[fieldname][0:self.halos[fieldname][0],
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][-self.halos[fieldname][1]:,
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       0:self.halos[fieldname][2],
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       -self.halos[fieldname][3]:,
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       0:self.halos[fieldname][4]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       -self.halos[fieldname][5]:]
    #     #
    #     # # # These are the halos to send:neg x, pos x, neg y, pos y, neg z, pos z
    #     # self.fields[fieldname][self.halos[fieldname][0]:self.halos[fieldname][0]+self.halos[fieldname][0],
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][-(self.halos[fieldname][1]+self.halos[fieldname][1]):-self.halos[fieldname][1],
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       self.halos[fieldname][2]:self.halos[fieldname][2]+self.halos[fieldname][2],
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       -(self.halos[fieldname][3]+self.halos[fieldname][3]):-self.halos[fieldname][3],
    #     #       self.halos[fieldname][4]:-self.halos[fieldname][5]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       self.halos[fieldname][4]:self.halos[fieldname][4]+self.halos[fieldname][4]]
    #     #
    #     # self.fields[fieldname][self.halos[fieldname][0]:-self.halos[fieldname][1],
    #     #       self.halos[fieldname][2]:-self.halos[fieldname][3],
    #     #       -(self.halos[fieldname][5] + self.halos[fieldname][5]):-self.halos[fieldname][5]]

    def communicate_external(self, fieldname, recv_slice, get_slice, neighbor_id):
        print("TODO comm extern")

    def check_local(self, direction):
        return self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[direction]]

    def check_global_boundary(self, direction):
        return self.neighbors_id[direction] is None

class DomainPreprocess:
    def __init__(self, domain, periodic, subdivs_per_dim, fileoutput=""):
        self.domain = domain
        self.periodic = periodic
        self.subdivs_per_dim = subdivs_per_dim
        self.fileout = fileoutput

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
                                                                        "a factor of the given domain size.")
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
                                                               pid=ind,
                                                               size=subdiv_size,
                                                               global_coords=global_range,
                                                               gridpoints=subdiv_gridpoints,
                                                               neighbors_id=nindex))

                    self.vweights.append(int(comp_cost))
                    self.xadj.append(int(self.edgecounter))
                    for e in range(len(nindex)):
                        if nindex[e] is not None:
                            self.edgecounter += 1
                            self.adjncy.append(int(nindex[e]))
                            self.eweights.append(int(comm_cost[e]))

        with open("subdivisions.pkl", "wb") as f:
            pickle.dump(self.subdivisions, f)

        if self.fileout == "metis":
            self.write_to_file_metis_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            "subdomains")
        elif self.fileout == "scotch":
            self.write_to_file_scotch_format(self.adjncy,
                                            self.xadj,
                                            self.vweights,
                                            self.eweights,
                                            self.edgecounter,
                                            "subdomains")
        elif self.fileout == "both":
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

    def pymetis_partitioning(self, nparts, verbose=False):
        self.prepare_for_pymetis()

        partitioning = part_graph(nparts,
                                  xadj=self.xadj,
                                  adjncy=self.adjncy,
                                  vweights=self.vweights,
                                  eweights=self.eweights)

        with open("subdomains_pymetis.dat.part." + str(nparts), "w") as f:
            for i in partitioning[1]:
                f.write(str(i) + "\n")

        if verbose:
            print(partitioning[1])


class DomainDecompositionStencil:
    def __init__(self):
        self.subdiv_stencil_list = []

    def compute(self):
        for sd in self.subdiv_stencil_list:
            sd.compute()


class DomainDecomposition:
    def __init__(self, fileinput=None, fileinputformat=None):
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

        for sd in self.subdivisions:
            sd.neighbor_list = self.subdivisions

    def load_subdivisions(self):
        with open("subdivisions.pkl", "rb") as f:
            return pickle.load(f)

    def register_field(self, fieldname, halo, field_ic_file=None):
        for sd in self.subdivisions:
            sd.register_field(fieldname, halo, field_ic_file)

    def register_stencil(self, **kwargs):
        dds = DomainDecompositionStencil()
        for sd in self.subdivisions:
            dds.subdiv_stencil_list.append(sd.register_stencil(**kwargs))
        return dds

    def communicate(self, fieldname=None):
        for sd in self.subdivisions:
            sd.communicate(fieldname)

    def save_fields(self):
        for sd in self.subdivisions:
            sd.save_fields()

    def swap_fields(self, field1, field2):
        for sd in self.subdivisions:
            sd.swap_fields(field1, field2)

