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
import sys


from mpi4py import MPI

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
    def __init__(self, id, pid, size, global_coords, neighbors_id):
        self.id = id
        self.partitions_id = pid
        self.global_coords = global_coords
        self.size = size
        self.neighbors_id = neighbors_id
        self.neighbor_list = None
        self.fields = {}
        self.halos = {}
        self.global_bc = {}
        self.global_bc_arr = {}
        self.recv_slices = {}
        self.send_slices = {}
        self.get_local = {}
        self.get_global = {}
        self.onesided = False
        self.onesided_buffers = {}
        self.onesided_windows = {}

    def set_boundary_condition(self, fieldname, direction, array):
        self.global_bc_arr[fieldname][direction] = array

    def register_field(self, fieldname, halo, field_ic_file=None, field_bc_file=None,
                       staggered=None, haloincluded=False):
        self.halos[fieldname] = halo
        if staggered is None:
            staggered = (0, 0, 0)
            self.fields[fieldname] = np.zeros((self.size[0] + halo[0] + halo[1],
                                               self.size[1] + halo[2] + halo[3],
                                               self.size[2] + halo[4] + halo[5]))
        else:
            self.fields[fieldname] = np.zeros((self.size[0] - staggered[0] + halo[0] + halo[1],
                                               self.size[1] - staggered[1] + halo[2] + halo[3],
                                               self.size[2] - staggered[2] + halo[4] + halo[5]))

        # print(fieldname, haloincluded)

        if field_ic_file is not None:
            if not haloincluded:
                self.fields[fieldname][halo[0]:None if halo[1] == 0 else -halo[1],
                                       halo[2]:None if halo[3] == 0 else -halo[3],
                                       halo[4]:None if halo[5] == 0 else -halo[5]] = np.load(
                    field_ic_file, mmap_mode='r')[self.global_coords[0]:self.global_coords[1] - staggered[0],
                                                  self.global_coords[2]:self.global_coords[3] - staggered[1],
                                                  self.global_coords[4]:self.global_coords[5] - staggered[2]]
                # if fieldname == "h":
                #     print(self.fields[fieldname].shape)
            else:
                print(fieldname, self.fields[fieldname].shape)
                self.fields[fieldname][:, :, :] = np.load(
                    field_ic_file, mmap_mode='r')[self.global_coords[0]:self.global_coords[1]
                                                                        + halo[0] + halo[1] - staggered[0],
                                                  self.global_coords[2]:self.global_coords[3]
                                                                        + halo[2] + halo[3] - staggered[1],
                                                  self.global_coords[4]:self.global_coords[5]
                                                                        + halo[4] + halo[5] - staggered[2]]

                # print(np.where(self.fields[fieldname][:, :, :] == 0.0))

        self.global_bc[fieldname] = field_bc_file
        self.global_bc_arr[fieldname] = [None, None, None, None, None, None]
        self.setup_slices(fieldname)

    def setup_slices(self, fieldname):
        self.recv_slices[fieldname] = [
            # Halo region in negative x direction
            np.s_[0:self.halos[fieldname][0],
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Halo region in positive x direction
            np.s_[-self.halos[fieldname][1]:,
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Halo region in negative y direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  0:None if self.halos[fieldname][2] == 0 else self.halos[fieldname][2],
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Halo region in positive y direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  -self.halos[fieldname][3]:,
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Halo region in negative z direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  0:self.halos[fieldname][4]],
            # Halo region in positive z direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  -self.halos[fieldname][5]:]
        ]

        self.send_slices[fieldname] = [
            # Overlap region in neighboring, negative x direction
            np.s_[self.halos[fieldname][0]:(None if self.halos[fieldname][0] == 0
                                            else self.halos[fieldname][0] + self.halos[fieldname][0]),
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, positive x direction
            np.s_[-(self.halos[fieldname][1] + self.halos[fieldname][1]):(None if self.halos[fieldname][1] == 0
                                                                          else -self.halos[fieldname][1]),
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, negative y direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:(None if self.halos[fieldname][2] == 0
                                            else self.halos[fieldname][2] + self.halos[fieldname][2]),
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, positive y direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  -(self.halos[fieldname][3] + self.halos[fieldname][3]):(None if self.halos[fieldname][3] == 0
                                                                          else -self.halos[fieldname][3]),
                  self.halos[fieldname][4]::None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, negative z direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:(None if self.halos[fieldname][4] == 0
                                            else self.halos[fieldname][4] + self.halos[fieldname][4])],
                        # Overlap region in neighboring, positive z direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  -(self.halos[fieldname][5] + self.halos[fieldname][5]):(None if self.halos[fieldname][5] == 0
                                                                          else -self.halos[fieldname][5])]
        ]

        self.get_local[fieldname] = [
            # Overlap region in neighboring, positive x direction
            np.s_[-(self.halos[fieldname][1] + self.halos[fieldname][1]):(None if self.halos[fieldname][1] == 0
                                                                          else -self.halos[fieldname][1]),
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, negative x direction
            np.s_[self.halos[fieldname][0]:(None if self.halos[fieldname][0] == 0
                                            else self.halos[fieldname][0] + self.halos[fieldname][0]),
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, positive y direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  -(self.halos[fieldname][3] + self.halos[fieldname][3]):(None if self.halos[fieldname][3] == 0
                                                                          else -self.halos[fieldname][3]),
                  self.halos[fieldname][4]::None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, negative y direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:(None if self.halos[fieldname][2] == 0
                                            else self.halos[fieldname][2] + self.halos[fieldname][2]),
                  self.halos[fieldname][4]:None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]],
            # Overlap region in neighboring, positive z direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  -(self.halos[fieldname][5] + self.halos[fieldname][5]):(None if self.halos[fieldname][5] == 0
                                                                          else -self.halos[fieldname][5])],
            # Overlap region in neighboring, negative z direction
            np.s_[self.halos[fieldname][0]:None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1],
                  self.halos[fieldname][2]:None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3],
                  self.halos[fieldname][4]:(None if self.halos[fieldname][4] == 0
                                            else self.halos[fieldname][4] + self.halos[fieldname][4])]
        ]

        self.get_global[fieldname] = [
            # Overlap region in global boundary, negative x direction
            np.s_[
                  self.global_coords[0]:self.global_coords[0] + self.halos[fieldname][0],
                  self.global_coords[2] + self.halos[fieldname][2]:self.global_coords[3] + self.halos[fieldname][2],
                  self.global_coords[4] + self.halos[fieldname][4]:self.global_coords[5] + self.halos[fieldname][4]
            ],
            # Overlap region in global boundary, positive x direction
            np.s_[
                  self.global_coords[1] + self.halos[fieldname][0]:(self.global_coords[1] + self.halos[fieldname][0]
                                                              + self.halos[fieldname][1]),
                  self.global_coords[2] + self.halos[fieldname][2]:self.global_coords[3] + self.halos[fieldname][2],
                  self.global_coords[4] + self.halos[fieldname][4]:self.global_coords[5] + self.halos[fieldname][4]
            ],
            # Overlap region in global boundary, negative y direction
            np.s_[
                  self.global_coords[0] + self.halos[fieldname][0]:self.global_coords[1] + self.halos[fieldname][0],
                  self.global_coords[2]:self.global_coords[2] + self.halos[fieldname][2],
                  self.global_coords[4] + self.halos[fieldname][4]:self.global_coords[5] + self.halos[fieldname][4]
            ],
            # Overlap region in global boundary, positive y direction
            np.s_[
                  self.global_coords[0] + self.halos[fieldname][0]:self.global_coords[1] + self.halos[fieldname][0],
                  self.global_coords[3] + self.halos[fieldname][2]:(self.global_coords[3] + self.halos[fieldname][2]
                                                              + self.halos[fieldname][3]),
                  self.global_coords[4] + self.halos[fieldname][4]:self.global_coords[5] + self.halos[fieldname][4]
            ],
            # Overlap region in global boundary, negative z direction
            np.s_[
                  self.global_coords[0] + self.halos[fieldname][0]:self.global_coords[1] + self.halos[fieldname][0],
                  self.global_coords[2] + self.halos[fieldname][2]:self.global_coords[3] + self.halos[fieldname][2],
                  self.global_coords[4]:self.global_coords[4] + self.halos[fieldname][4]
            ],
            # Overlap region in global boundary, positive z direction
            np.s_[
                  self.global_coords[0] + self.halos[fieldname][0]:self.global_coords[1] + self.halos[fieldname][0],
                  self.global_coords[2] + self.halos[fieldname][2]:self.global_coords[3] + self.halos[fieldname][2],
                  self.global_coords[5] + self.halos[fieldname][4]:(self.global_coords[5] + self.halos[fieldname][4]
                                                                    + self.halos[fieldname][5])
            ]
        ]

        if self.onesided:
            self.onesided_buffers[fieldname] = [
                np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][0]]),
                np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][1]]),
                np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][2]]),
                np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][3]]),
                np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][4]]),
                np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][5]])
            ]
            self.onesided_windows[fieldname] = [
                MPI.Win.Create(self.onesided_buffers[fieldname][0], comm=MPI.COMM_WORLD),
                MPI.Win.Create(self.onesided_buffers[fieldname][1], comm=MPI.COMM_WORLD),
                MPI.Win.Create(self.onesided_buffers[fieldname][2], comm=MPI.COMM_WORLD),
                MPI.Win.Create(self.onesided_buffers[fieldname][3], comm=MPI.COMM_WORLD),
                MPI.Win.Create(self.onesided_buffers[fieldname][4], comm=MPI.COMM_WORLD),
                MPI.Win.Create(self.onesided_buffers[fieldname][5], comm=MPI.COMM_WORLD)
            ]


    def save_fields(self, fieldnames=None, path="", prefix="", postfix=None):
        if fieldnames is None:
            for k in self.fields.keys():
                filename = (path + prefix + str(k) + "_from_"
                            + "x" + str(self.global_coords[0])
                            + "y" + str(self.global_coords[2])
                            + "z" + str(self.global_coords[4])
                            + "_to_"
                            + "x" + str(self.global_coords[1] - 1)
                            + "y" + str(self.global_coords[3] - 1)
                            + "z" + str(self.global_coords[5] - 1))
                if postfix is not None:
                    filename += "_" + str(postfix)
                np.save(filename, self.get_interior_field(k))
        else:
            for k in fieldnames:
                filename = (path + prefix + str(k) + "_from_"
                            + "x" + str(self.global_coords[0])
                            + "y" + str(self.global_coords[2])
                            + "z" + str(self.global_coords[4])
                            + "_to_"
                            + "x" + str(self.global_coords[1] - 1)
                            + "y" + str(self.global_coords[3] - 1)
                            + "z" + str(self.global_coords[5] - 1))
                if postfix is not None:
                    filename += "_" + str(postfix)
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

        for k, v in inputs.items():
            ulx = max(ulx, self.halos[v][0])
            drx = max(drx, self.halos[v][1])
            uly = max(uly, self.halos[v][2])
            dry = max(dry, self.halos[v][3])
            ulz = max(ulz, self.halos[v][4])
            drz = max(drz, self.halos[v][5])
            # print(self.halos[v][0], self.halos[v][1], self.halos[v][2], self.halos[v][3], self.halos[v][4], self.halos[v][5])

        # endpoint = self.size + lower halo - 1 (because index starts at 0 but size does not)
        drx = self.size[0] + ulx - 1
        dry = self.size[1] + uly - 1
        drz = self.size[2] + ulz - 1

        domain = gt.domain.Rectangle((ulx, uly, ulz), (drx, dry, drz))
        # print(ulx, uly, ulz, drx, dry, drz)
        # gt.domain.Rectangle((1, 1, 0), (89, 43, 0)),#

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
        xneg = self.halos[fieldname][0]
        xpos = None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1]
        yneg = self.halos[fieldname][2]
        ypos = None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3]
        zneg = self.halos[fieldname][4]
        zpos = None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]
        return self.fields[fieldname][xneg:xpos, yneg:ypos, zneg:zpos]

    def swap_fields(self, field1, field2):
        self.fields[field1][:], self.fields[field2][:] = self.fields[field2][:], self.fields[field1][:]

    def apply_boundary_condition(self, fieldname):
        # Iterate over all neighbors i.e. all directions:
        for d in range(len(self.neighbors_id)):
            # Check if neighbor in current direction is the global boundary:
            if self.check_global_boundary(d):
                if self.halos[fieldname][d] != 0:
                    # Load boundary condition from global file
                    if self.global_bc[fieldname] is not None:
                        self.fields[fieldname][self.recv_slices[fieldname][d]] = np.load(
                            self.global_bc[fieldname], mmap_mode='r')[self.get_global[fieldname][d]]
                    # Load boundary condition from set array
                    elif self.global_bc_arr[fieldname][d] is not None:
                        self.fields[fieldname][self.recv_slices[fieldname][d]] = self.global_bc_arr[fieldname][d]
                    else:
                        warnings.warn("No boundary condition file or set_boundary_condition() provided in direction "
                                      + str(d) + " for field " + str(fieldname), RuntimeWarning)

    def communicate(self, fieldname=None):
        if fieldname is None:
            for k in self.halos.keys():
                if self.onesided:
                    self.communicate_one_way_field(k)
                else:
                    self.communicate_field(k)
        else:
            if self.onesided:
                self.communicate_one_way_field(fieldname)
            else:
                self.communicate_field(fieldname)

    def communicate_one_way_field(self, fieldname):
        # temp_buffer = [None] * len(self.neighbors_id)
        # for d in range(len(self.neighbors_id)):
            # if self.halos[fieldname][d] != 0:
            # temp_buffer[d] = np.ones_like(self.fields[fieldname][self.recv_slices[fieldname][d]])
            # self.communicate_external_update_buffer(fieldname, d)
                # print(MPI.COMM_WORLD.Get_rank(), d, self.onesided_buffers[fieldname][d])
                # sys.stdout.flush()
        if MPI.COMM_WORLD.Get_size() > 1:
            for d in range(len(self.neighbors_id)):
                if self.halos[fieldname][d] != 0:
                    if self.neighbors_id[d] is not None:
                        fixed_neighbor = DomainPartitions.domain_partitions[self.neighbors_id[d]]
                    else:
                        fixed_neighbor = None #MPI.COMM_WORLD.Get_rank()

                    if d % 2 == 0:
                        cd = d + 1
                    else:
                        cd = d - 1

                    # print(MPI.COMM_WORLD.Get_rank(), d, fixed_neighbor)
                    # print(temp_buffer[d])
                    self.communicate_external_get(fieldname,
                                                  np.ascontiguousarray(self.fields[fieldname][self.get_local[fieldname][cd]]), #temp_buffer[d],
                                                  d,
                                                  fixed_neighbor)

                # print(self.onesided_buffers[fieldname][d])

        # Iterate over all neighbors i.e. all directions:
        for d in range(len(self.neighbors_id)):
            if self.halos[fieldname][d] != 0:
                # Check if neighbor in current direction is the global boundary:
                if not self.check_global_boundary(d):
                    # Check if neighbor in current direction is local or external and communicate accordingly:
                    if self.check_local(d) or MPI.COMM_WORLD.Get_size() == 1:
                            self.communicate_local(fieldname,
                                                   self.recv_slices[fieldname][d],
                                                   self.get_local[fieldname][d],
                                                   d)
                    else: # temp_buffer[d]
                        self.communicate_external_update_halo(fieldname, d)

    def communicate_field(self, fieldname):
        requests = [None] * 2 * len(self.neighbors_id)
        temp_buffer = [None] * len(self.neighbors_id)
        for d in range(len(self.neighbors_id)):
            temp_buffer[d] = np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][d]])
        # Iterate over all neighbors i.e. all directions:
        for d in range(len(self.neighbors_id)):
            # temp_buffer[d] = np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][d]])
            # Check if neighbor in current direction is the global boundary:
            if self.check_global_boundary(d):
                # Set the requests for the corresponding direction to null, so that the MPI waitall() works later.
                requests[2 * d] = requests[2 * d + 1] = MPI.REQUEST_NULL
            else:
                # Check if neighbor in current direction is local or external and communicate accordingly:
                if self.check_local(d) or MPI.COMM_WORLD.Get_size() == 1:
                    requests[2 * d] = requests[2 * d + 1] = MPI.REQUEST_NULL
                    if self.halos[fieldname][d] != 0:
                        self.communicate_local(fieldname,
                                               self.recv_slices[fieldname][d],
                                               self.get_local[fieldname][d],
                                               d)
                else:
                    if self.halos[fieldname][d] != 0:
                        # Communicate two-way
                        if MPI.COMM_WORLD.Get_rank() % 2 == 0:
                                requests[2 * d] = self.communicate_external_send(
                                    fieldname,
                                    self.send_slices[fieldname][d],
                                    DomainPartitions.domain_partitions[self.neighbors_id[d]]
                                )
                                requests[2 * d + 1] = self.communicate_external_recv(
                                    temp_buffer[d],
                                    DomainPartitions.domain_partitions[self.neighbors_id[d]]
                                )

                        else:
                            requests[2 * d] = self.communicate_external_recv(
                                temp_buffer[d],
                                DomainPartitions.domain_partitions[self.neighbors_id[d]]
                            )

                            requests[2 * d + 1] = self.communicate_external_send(
                                fieldname,
                                self.send_slices[fieldname][d],
                                DomainPartitions.domain_partitions[self.neighbors_id[d]]
                            )
                    else:
                        requests[2 * d] = requests[2 * d + 1] = MPI.REQUEST_NULL

        # Update halo regions after receiving all boundaries
        if MPI.COMM_WORLD.Get_size() > 1:
            MPI.Request.waitall(requests)

            for d in range(len(self.neighbors_id)):
                # Check if neighbor in current direction is the global boundary:
                if not self.check_global_boundary(d):
                    # Check if neighbor subdivision is in the same partition
                    if not self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[d]]:
                        self.fields[fieldname][self.recv_slices[fieldname][d]] = temp_buffer[d].copy()

    def communicate_local(self, fieldname, recv_slice, get_slice, neighbor_id):
        # Numpy view of the halo region
        recv = self.fields[fieldname][recv_slice]
        # Neighboring subdivision
        neighbor_sd = self.get_local_neighbor(self.neighbors_id[neighbor_id])
        # Overlap region in neighboring subdivision
        get = neighbor_sd.fields[fieldname][get_slice]
        # Transfer overlap region into halo region
        recv[:] = get[:]

    def communicate_external_send(self, fieldname, send_slice, send_id):
        req = MPI.COMM_WORLD.Isend(np.ascontiguousarray(self.fields[fieldname][send_slice]), dest=send_id)
        return req

    def communicate_external_recv(self, temp_buffer, recv_id):
        req = MPI.COMM_WORLD.Irecv(temp_buffer[:], source=recv_id)
        return req

    def communicate_external_get(self, fieldname, temp_buffer, direction, recv_id):
        # print(MPI.COMM_WORLD.Get_rank(), direction, recv_id)
        # print(temp_buffer[:])
        self.onesided_windows[fieldname][direction].Fence()
        # print(MPI.COMM_WORLD.Get_rank(), direction, recv_id)
        # sys.stdout.flush()
        if recv_id is not None:
            # print("Not None:", MPI.COMM_WORLD.Get_rank(), direction, recv_id)
            # sys.stdout.flush()
            self.onesided_windows[fieldname][direction].Put(origin=temp_buffer[:], target_rank=recv_id)
        self.onesided_windows[fieldname][direction].Fence()

        # if recv_id is not None:
        #     print(temp_buffer[:])
        #     sys.stdout.flush()


    def communicate_external_update_buffer(self, fieldname, d):
        self.onesided_buffers[fieldname][d] = self.fields[fieldname][self.get_local[fieldname][d]]

    def communicate_external_update_halo(self, fieldname, direction):
        # print(temp_buffer[:])
        # self.onesided_windows[fieldname][direction].Fence()
        if direction % 2 == 0:
            cd = direction + 1
        else:
            cd = direction - 1
        self.fields[fieldname][self.recv_slices[fieldname][direction]] = self.onesided_buffers[fieldname][cd]#temp_buffer[:]
        # self.onesided_windows[fieldname][direction].Fence()

    def check_local(self, direction):
        return self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[direction]]

    def check_global_boundary(self, direction):
        return self.neighbors_id[direction] is None


class DomainDecompositionStencil:
    def __init__(self):
        self.subdiv_stencil_list = []

    def compute(self):
        for sd in self.subdiv_stencil_list:
            sd.compute()


class DomainDecomposition:
    def __init__(self, fileinput=None, fileinputformat=None, path="", prefix="", comm_onesided=False):
        self.subdivisions = self.load_subdivisions(path=path, prefix=prefix)
        self.path = path
        self.prefix = prefix

        if fileinput is None or fileinputformat is None:
            raise ValueError("Need fileinput and fileinputformat for partitioning file.")

        DomainPartitions.load_partitions(path + prefix + fileinput, fileinputformat)

        # If there are more than one MPI processors:
        # Change the partitions id of the subdivision to the one given by the domain decomposition from the preprocess
        if MPI.COMM_WORLD.Get_size() > 1:
            for s in range(len(self.subdivisions)):
                self.subdivisions[s].partitions_id = DomainPartitions.domain_partitions[s]

        # If there are more than one MPI processors: remove subdivisions of other partitions from list
        if MPI.COMM_WORLD.Get_size() > 1:
            temp_list = []
            for sd in self.subdivisions:
                if sd.partitions_id == MPI.COMM_WORLD.Get_rank():
                    temp_list.append(sd)
            self.subdivisions = temp_list

        for sd in self.subdivisions:
            sd.neighbor_list = self.subdivisions

        if comm_onesided:
            for sd in self.subdivisions:
                sd.onesided = True

    @staticmethod
    def load_subdivisions(path="", prefix=""):
        with open(path + prefix + "subdivisions.pkl", "rb") as f:
            return pickle.load(f)

    def register_field(self, fieldname, halo, field_ic_file=None, field_bc_file=None,
                       staggered=None, haloincluded=False):
        for sd in self.subdivisions:
            sd.register_field(fieldname, halo, field_ic_file, field_bc_file, staggered, haloincluded)

    def register_stencil(self, **kwargs):
        dds = DomainDecompositionStencil()
        for sd in self.subdivisions:
            dds.subdiv_stencil_list.append(sd.register_stencil(**kwargs))
        return dds

    def communicate(self, fieldname=None):
        for sd in self.subdivisions:
            sd.communicate(fieldname)

    def set_boundary_condition(self, fieldname, direction, halo, array):
        for sd in self.subdivisions:
            if (direction == 0 or direction == 1):
                slice = np.s_[:halo, sd.global_coords[2]:sd.global_coords[3], sd.global_coords[4]:sd.global_coords[5]]
            elif (direction == 2 or direction == 3):
                slice = np.s_[sd.global_coords[0]:sd.global_coords[1], :halo, sd.global_coords[4]:sd.global_coords[5]]
            elif (direction == 4 or direction == 5):
                slice = np.s_[sd.global_coords[0]:sd.global_coords[1], sd.global_coords[2]:sd.global_coords[3], :halo]

            sd.set_boundary_condition(fieldname, direction, array[slice])

    def apply_boundary_condition(self, fieldname):
        for sd in self.subdivisions:
            sd.apply_boundary_condition(fieldname)

    def save_fields(self, fieldnames=None, postfix=None):
        for sd in self.subdivisions:
            sd.save_fields(fieldnames, path=self.path, prefix=self.prefix, postfix=postfix)

    def swap_fields(self, field1, field2):
        for sd in self.subdivisions:
            sd.swap_fields(field1, field2)

