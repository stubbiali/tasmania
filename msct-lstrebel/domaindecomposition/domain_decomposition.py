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
    """ Class containing the global partitioning array and functions to load it from the partitioning file.

    The partitioning array is one dimensional and has the size of the number of subdivisions.
    Each entry of the partitioning array contains the index of the partition
    of the corresponding array index subdivision.

    e.g. to get the partition id for the 17th subdivision would be:
    DomainPartitions.domain_partitions[16] (indexing starts at 0).
    """
    domain_partitions = None

    @staticmethod
    def load_partitions(fileinput, fileformat):
        """ Load the partitioning file depending on the fileformat ("scotch" or "metis") it was saved in.

        :param fileinput: Path and name of the partitioning file.
        :param fileformat: Fileformat of the partitioning file. Can be "scotch" or "metis".
        :return: None
        """
        if fileformat == "metis":
            DomainPartitions.domain_partitions = DomainPartitions.load_from_metis_file(fileinput)
        elif fileformat == "scotch":
            DomainPartitions.domain_partitions = DomainPartitions.load_from_scotch_file(fileinput)
        else:
            print("Only 'metis' or 'scotch' as fileformat accepted.")

    @staticmethod
    def load_from_metis_file(fileinput):
        """ Helper function to load the partitioning if it was saved in the Metis format.

        :param fileinput: Path and name of the partitioning file.
        :return: Partitioning array.
        """
        return np.loadtxt(fileinput, dtype=int)

    @staticmethod
    def load_from_scotch_file(fileinput):
        """ Helper function to load the partitioning if it was saved in the Scotch format.

        :param fileinput: Path and name of the partitioning file.
        :return: Partitioning array.
        """
        return np.loadtxt(fileinput, dtype=int, skiprows=1, usecols=1)

    @staticmethod
    def print_partitions():
        """ Small helper function to print the partitioning as loaded from the file.

        :return: None
        """
        print(DomainPartitions.domain_partitions)


class DomainSubdivision:
    """ Class containing all the information of a subdivision as well as all functions needed for the subdivisions.

    """
    def __init__(self, id, pid, size, global_coords, neighbors_id):
        """ Initialize subdivision with the input parameters.

        :param id: Subdivision identification number, generated in the pre-process.
        :param pid: Partition identification number, generated in the partitioning of the pre-process.
        :param size: Subdivision domain size: array of size 3:
        [x-direction, y-direction, and z-direction size].
        :param global_coords: Global coordinates of the subdivision, array of size 6:
        [x-direction minimum, y-direction minimum, z-direction minimum,
        x-direction maximum, y-direction maximum, z-direction maximum]
        :param neighbors_id: Subdivision identification numbers of the neighboring subdivisions: array of size 6:
        [negative x-direction, positive x-direction,
         negative y-direction, positive y-direction,
         negative z-direction, positive z-direction]
        """
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
        """ Helper function to set an input array as the global boundary condition for a specific field and direction.

        :param fieldname: Name of the field to set the global boundary condition.
        :param direction: Direction of the global boundary condition.
        :param array: Input array to be the global boundary condition.
        :return: None
        """
        self.global_bc_arr[fieldname][direction] = array

    def register_field(self, fieldname, halo, field_ic_file=None, field_bc_file=None,
                       staggered=None, haloincluded=False, singlefile=True):
        """ Register field to subdivision. Fields need to be register in order for the arrays to be allocated,
        and subdivision boundary indices to be prepared for the communication between subdivisions.

        :param fieldname: Name of the field.
        :param halo: Size of the halo in each direction: array of size 6:
        [negative x-direction, positive x-direction,
         negative y-direction, positive y-direction,
         negative z-direction, positive z-direction] halo size.
        :param field_ic_file: Optional: Initial condition / values array file.
        If none is provided initial values are zero.
        :param field_bc_file: Optional: Boundary value file. In cases were the boundary condition is provided in a file.
        If none is provided the global boundary condition needs to be registered and set manually.
        :param staggered: Optional: Parameter for staggered grids: array of size 3:
        [x-direction, y-direction, z-direction] reduces field size by array value.
        Should only have 0 or 1 in the array to be supported by GridTools.
        :param haloincluded: Optional: Binary option if the initial value field includes the global halo or not.
        :return: None
        """
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

        if field_ic_file is not None and singlefile:
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
                # print(fieldname, self.fields[fieldname].shape)
                self.fields[fieldname][:, :, :] = np.load(
                    field_ic_file, mmap_mode="r")[self.global_coords[0]:self.global_coords[1]
                                                                        + halo[0] + halo[1] - staggered[0],
                                                  self.global_coords[2]:self.global_coords[3]
                                                                        + halo[2] + halo[3] - staggered[1],
                                                  self.global_coords[4]:self.global_coords[5]
                                                                        + halo[4] + halo[5] - staggered[2]]
        if field_ic_file is not None and not singlefile:
            self.fields[fieldname][:, :, :] = np.load(
                field_ic_file + "_" + str(self.id) + ".npy", mmap_mode="r")[:, :, :]

        self.global_bc[fieldname] = field_bc_file
        self.global_bc_arr[fieldname] = [None, None, None, None, None, None]
        self.setup_slices(fieldname)

    def setup_slices(self, fieldname):
        """ Helper function to set up all indices for the inter-subdivision communication for fields.
        If the one-sided communication option is enabled also setup the one-sided windows.

        The following indices are generated:

        recv_slices: Indices to receive the halo region into i.e. the halo region of the field. Array of size 6:
        [negative x-direction, positive x-direction,
         negative y-direction, positive y-direction,
         negative z-direction, positive z-direction] halo region indices.

        send_slices: Indices to send the halo region to the neighbor i.e. the outermost part of the interior field.
        Array of size 6:
        [negative x-direction, positive x-direction,
         negative y-direction, positive y-direction,
         negative z-direction, positive z-direction] overlap region
         i.e. boundary values for neighbor and outermost interior values for this subdivision

        get_local: Indices to copy the halo region to the neighbor if they are on the same partition
        i.e. the outermost part of the interior field. Array of size 6:
        [positive x-direction, negative x-direction,
         positive y-direction, negative y-direction,
         positive z-direction, negative z-direction,] overlap region
         i.e. boundary values for neighbor and outermost interior values for this subdivision
         Differs from send_slices in order of indices only, because TODO good explanation for this

        get_global:


        :param fieldname: Name of the field.
        :return: None
        """
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
                None if self.halos[fieldname][0] == 0 else np.zeros_like(self.fields[fieldname][
                                                                             self.recv_slices[fieldname][0]]),
                None if self.halos[fieldname][1] == 0 else np.zeros_like(self.fields[fieldname][
                                                                             self.recv_slices[fieldname][1]]),
                None if self.halos[fieldname][2] == 0 else np.zeros_like(self.fields[fieldname][
                                                                             self.recv_slices[fieldname][2]]),
                None if self.halos[fieldname][3] == 0 else np.zeros_like(self.fields[fieldname][
                                                                             self.recv_slices[fieldname][3]]),
                None if self.halos[fieldname][4] == 0 else np.zeros_like(self.fields[fieldname][
                                                                             self.recv_slices[fieldname][4]]),
                None if self.halos[fieldname][5] == 0 else np.zeros_like(self.fields[fieldname][
                                                                             self.recv_slices[fieldname][5]])
            ]
            self.onesided_windows[fieldname] = [
                None if self.halos[fieldname][0] == 0 else MPI.Win.Create(self.onesided_buffers[fieldname][0],
                                                                          comm=MPI.COMM_WORLD),
                None if self.halos[fieldname][1] == 0 else MPI.Win.Create(self.onesided_buffers[fieldname][1],
                                                                          comm=MPI.COMM_WORLD),
                None if self.halos[fieldname][2] == 0 else MPI.Win.Create(self.onesided_buffers[fieldname][2],
                                                                          comm=MPI.COMM_WORLD),
                None if self.halos[fieldname][3] == 0 else MPI.Win.Create(self.onesided_buffers[fieldname][3],
                                                                          comm=MPI.COMM_WORLD),
                None if self.halos[fieldname][4] == 0 else MPI.Win.Create(self.onesided_buffers[fieldname][4],
                                                                          comm=MPI.COMM_WORLD),
                None if self.halos[fieldname][5] == 0 else MPI.Win.Create(self.onesided_buffers[fieldname][5],
                                                                          comm=MPI.COMM_WORLD)
            ]

    def save_fields(self, fieldnames=None, path="", prefix="", postfix=None):
        """ Function to save fields to file per subdivision.

        :param fieldnames: List of fieldnames to save
        :param path: Path to location where files should be saved.
        :param prefix: Prefix for the naming of all files e.g. run id.
        :param postfix: Postfix for the naming of all files e.g. time step.
        :return: None
        """
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
        """ Function to register GridTools4Py stencil to the subdivision.

        :param kwargs: keyword arguments almost the same as kwargs for usual GridTools4Py stencil instantiation.
        Accepts keywords: definitions_func, inputs, constant_inputs, global_inputs, outputs,
        domain, mode, vertical_direction, and the new keyword reductions.

        :return: Instantiated GridTools4Py stencil for the subdivision.
        """
        # Set default values
        definitions_func = inputs = outputs = domain = None
        constant_inputs = global_inputs = {}
        mode = Mode.DEBUG
        vertical_direction = VerticalDirection.PARALLEL

        rdx = [0, 0, 0, 0, 0, 0]

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
            elif key == "reductions":
                rdx = kwargs[key]
            else:
                raise ValueError("\n  NGStencil accepts the following keyword arguments: \n"
                                 "  - definitions_func, \n"
                                 "  - inputs, \n"
                                 "  - constant_inputs [default: {}], \n"
                                 "  - global_inputs [default: {}], \n"
                                 "  - outputs, \n"
                                 "  - reductions, \n"
                                 "  - mode [default: DEBUG], \n"
                                 "  - vertical_direction [default: PARALLEL]. \n"
                                 "  The order does not matter. \n"
                                 " But provided key was: " + str(key))

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

        domain = gt.domain.Rectangle((ulx + rdx[0], uly+ rdx[2], ulz+ rdx[4]),
                                     (drx - rdx[1], dry - rdx[3], drz - rdx[5]))
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
        """ Helper function to get a reference to a neighboring subdivision in the same partition from the subdivision id.

        :param id: Subdivision identification number of the neighboring subdivision.
        :return: Subdivision Object of neighbor in same partition or None if the id is not a neighbor.
        """
        temp_sd = None
        for sd in self.neighbor_list:
            if id == sd.id:
                temp_sd = sd
        return temp_sd

    def get_interior_field(self, fieldname):
        """ Helper function to get only the interior part of a field i.e. exclude the subdivision halo regions.

        :param fieldname: Name of the field to access.
        :return: Interior part of the field from the subdivision.
        """
        xneg = self.halos[fieldname][0]
        xpos = None if self.halos[fieldname][1] == 0 else -self.halos[fieldname][1]
        yneg = self.halos[fieldname][2]
        ypos = None if self.halos[fieldname][3] == 0 else -self.halos[fieldname][3]
        zneg = self.halos[fieldname][4]
        zpos = None if self.halos[fieldname][5] == 0 else -self.halos[fieldname][5]
        return self.fields[fieldname][xneg:xpos, yneg:ypos, zneg:zpos]

    def swap_fields(self, field1, field2):
        """ Function to easily swap the content of two fields.
        Used in a lot of stencil codes to swap between old and new array values between time steps / computations.

        :param field1: First field to be swapped.
        :param field2: Second field to be swapped.
        :return: None
        """
        self.fields[field1][:], self.fields[field2][:] = self.fields[field2][:], self.fields[field1][:]

    def apply_boundary_condition(self, fieldname):
        """ Function to apply the global boundary values to a field if the subdivision is at the global boundary.
        Global boundary values need to be provided by file or by set_boundary_value function beforehand.

        :param fieldname: Name of the field to apply global boundary condition.
        :return: None
        """
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
        """ Function to delegate the communication of subdivision halo regions, for specified field or all fields.

        Delegates either two-way or one-sided communication depending on binary option "onesided" of the subdivision.

        :param fieldname: Optional: Name of the field to communicate.
        If none is provided all registered fields are communicated.
        :return: None
        """
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
        """ Function to start one sided communication between subdivisions.

        :param fieldname: Name of the field to exchange halo regions.
        :return: None
        """
        # Only communicate non-locally if there are more than 1 MPI processes.
        if MPI.COMM_WORLD.Get_size() > 1:
            # Iterate over directions
            for d in range(len(self.neighbors_id)):
                # Check if the halo in the given direction is zero i.e. no halo region.
                if self.halos[fieldname][d] != 0:
                    # Check if the neighbor in the direction exists or is the global boundary.
                    if self.neighbors_id[d] is not None:
                        fixed_neighbor = DomainPartitions.domain_partitions[self.neighbors_id[d]]
                    else:
                        fixed_neighbor = None

                    # Determine counter direction number based on direction.
                    if d % 2 == 0:
                        cd = d + 1
                    else:
                        cd = d - 1

                    # Call the one sided communcication function with the local slice in the counter direction,
                    # because Put needs this subdivisions outer most values of the opposite direction
                    # to transfer into the neighbors halo region.
                    self.communicate_external_put(fieldname,
                                                  np.ascontiguousarray(self.fields[fieldname][
                                                                           self.get_local[fieldname][cd]]),
                                                  d,
                                                  fixed_neighbor)

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
                    else:
                        # If not local then halo region was Put into buffer by neighbor, need to update halo region.
                        self.communicate_external_update_halo(fieldname, d)

    def communicate_field(self, fieldname):
        """ Function to perform standard two-way communication of a fields halo regions.

        :param fieldname: Name of the field to exchange halo regions.
        :return: None
        """
        # Create MPI request and buffer arrays
        requests = [None] * 2 * len(self.neighbors_id)
        temp_buffer = [None] * len(self.neighbors_id)
        for d in range(len(self.neighbors_id)):
            temp_buffer[d] = np.zeros_like(self.fields[fieldname][self.recv_slices[fieldname][d]])

        # Iterate over all neighbors i.e. all directions:
        for d in range(len(self.neighbors_id)):
            # Check if neighbor in current direction is the global boundary:
            if self.check_global_boundary(d):
                # If at the global boundary then set the requests for the corresponding direction to null,
                # so that the MPI waitall() works later.
                requests[2 * d] = requests[2 * d + 1] = MPI.REQUEST_NULL
            else:
                # Check if neighbor in current direction is local or external and communicate accordingly:
                if self.check_local(d) or MPI.COMM_WORLD.Get_size() == 1:
                    # Set the requests for the corresponding direction to null,
                    # so that the MPI waitall() works later.
                    requests[2 * d] = requests[2 * d + 1] = MPI.REQUEST_NULL
                    if self.halos[fieldname][d] != 0:
                        self.communicate_local(fieldname,
                                               self.recv_slices[fieldname][d],
                                               self.get_local[fieldname][d],
                                               d)
                else:
                    if self.halos[fieldname][d] != 0:
                        # Communicate two-way depending on rank either send first or receive first to avoid deadlock.
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
                        # If there is no halo region in a direction,
                        # set the requests for the corresponding direction to null,
                        # so that the MPI waitall() works later.
                        requests[2 * d] = requests[2 * d + 1] = MPI.REQUEST_NULL

        # Update halo regions after receiving all boundaries
        if MPI.COMM_WORLD.Get_size() > 1:
            # Wait for all send / receives to finish
            stats = [MPI.Status()] * 2 * len(self.neighbors_id)
            try:
                MPI.Request.waitall(requests, statuses=stats)
            except MPI.MPI.Exception:
                for st in stats:
                    if st.Get_error() != 0:
                        print("MPI Error after waitall: " + str(st.Get_error()))

            for d in range(len(self.neighbors_id)):
                # Check if neighbor in current direction is the global boundary:
                if not self.check_global_boundary(d):
                    # Check if neighbor subdivision is in the same partition
                    if not self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[d]]:
                        self.fields[fieldname][self.recv_slices[fieldname][d]] = temp_buffer[d].copy()

    def communicate_local(self, fieldname, recv_slice, get_slice, neighbor_id):
        """ Function to communicate halo exchanges for subdivisions in the same partition.
        Communicate means copying in this case.

        :param fieldname: Name of the field.
        :param recv_slice: Indices to receive the halos into i.e. subdivisions halo region in this direction.
        :param get_slice: Indices of the halo region in the neighboring subdivision to get
        i.e. the outer most parts of the neighboring field in the opposite direction.
        :param neighbor_id: Identification number for the neighboring subdivision.
        :return: None
        """
        # Numpy view of the halo region
        recv = self.fields[fieldname][recv_slice]
        # Neighboring subdivision
        neighbor_sd = self.get_local_neighbor(self.neighbors_id[neighbor_id])
        # Overlap region in neighboring subdivision
        get = neighbor_sd.fields[fieldname][get_slice]
        # Transfer / copy overlap region into halo region
        recv[:] = get[:]

    def communicate_external_send(self, fieldname, send_slice, send_id):
        """ Function to communicate two-way sending a halo region to its neighboring subdivision.

        Uses numpy function "ascontiguousarray" to avoid having to copy into a buffer first
        for non-contiguous halo regions.

        :param fieldname: Name of the field
        :param send_slice: Indices to send to neighbor i.e. outer most parts of the field in the direction.
        :param send_id: Identification number of the neighboring subdivision.
        :return: MPI Request for the send communication.
        """
        req = MPI.COMM_WORLD.Isend(np.ascontiguousarray(self.fields[fieldname][send_slice]), dest=send_id)
        return req

    def communicate_external_recv(self, temp_buffer, recv_id):
        """ Function to communicate two-way receiving a halo region from a neighboring subdivision.

        :param temp_buffer: Temporary buffer array to receive halo region into.
        :param recv_id: Identification number of neighboring subdivision.
        :return: MPI Request for the receive communication.
        """
        req = MPI.COMM_WORLD.Irecv(temp_buffer[:], source=recv_id)
        return req

    def communicate_external_put(self, fieldname, temp_buffer, direction, recv_id):
        """ Function to communicate one-sided by using MPI Put to transfer
        the outer most part of the field into neighboring halo region.

        :param fieldname: Name of the field.
        :param temp_buffer: Temporary buffer to hold the contiguous boundary before transfering.
        :param direction: Direction number needed to use the corresponding MPI Window.
        :param recv_id: Identification number of the receiving neighbor subdivision.
        :return: None
        """
        self.onesided_windows[fieldname][direction].Fence()
        if recv_id is not None:
            self.onesided_windows[fieldname][direction].Put(origin=temp_buffer[:], target_rank=recv_id)
        self.onesided_windows[fieldname][direction].Fence()

    # Used before changing one sided from MPI Get() to MPI Put()
    # def communicate_external_update_buffer(self, fieldname, d):
    #     """ Function to copy the values transferred by one sided communication into the windows buffer.
    #
    #     :param fieldname: Name of the field.
    #     :param d: Direction number needed to use the corresponding MPI Window.
    #     :return: None
    #     """
    #     self.onesided_buffers[fieldname][d] = self.fields[fieldname][self.get_local[fieldname][d]]

    def communicate_external_update_halo(self, fieldname, direction):
        """ Function to copy the values transferred by one sided communication into the windows buffer.

        Notice that the one sided buffers of any direction correspond to receiving from that direction
        and therefore need to be used to update the halo of the opposite direction for correct boundary exchange.

        :param fieldname: Name of the field.
        :param direction: Direction number needed to use the corresponding MPI Window.
        :return: None
        """
        # Determine the opposite direction
        if direction % 2 == 0:
            cd = direction + 1
        else:
            cd = direction - 1
        # Copy the buffer of the opposite direction into the halo region of the current direction.
        self.fields[fieldname][self.recv_slices[fieldname][direction]] = self.onesided_buffers[fieldname][cd]

    def check_local(self, direction):
        """ Small helper function to check if the subdivision in a direction is in the same partition or not.

        :param direction: Direction to check for local neighbor.
        :return: True if the subdivision in the given direction is in the same partition. False otherwise.
        """
        return self.partitions_id == DomainPartitions.domain_partitions[self.neighbors_id[direction]]

    def check_global_boundary(self, direction):
        """ Small helper function to check if the subdivision borders the global boundary in a direction.

        :param direction: Direction to check for global boundary.
        :return: True if the subdivision borders the global boundary in the given direction. False otherwise.
        """
        return self.neighbors_id[direction] is None


class DomainDecompositionStencil:
    """ Small class to store the subdivisions instantiated stencils in a list,
    so that the partition can call the compute function of all subdivision stencils easily.

    """
    def __init__(self):
        """ Initializes with an empty list.
        DomainDecomposition.register_stencil() function adds entries into the list.
        """
        self.subdiv_stencil_list = []

    def compute(self):
        """ Function to delegate the compute() call to all subdivision stencils.

        :return: None
        """
        for sd in self.subdiv_stencil_list:
            sd.compute()


class DomainDecomposition:
    """ Class containing the partition spanning information as well as functions to manage the partitions subdivisions.

    Also used to obscure separation caused by subdivisions from users
    and make the interface simpler so that the user has to write less boiler plate code.
    """
    def __init__(self, fileinput=None, fileinputformat=None, path="", prefix="", comm_onesided=False):
        """ Only one DomainDecomposition object should be instantiated for each simulation.
        (Except each MPI process has of course it's own)

        :param fileinput: Name of the partition file to be used.
        :param fileinputformat: Format of the partition file, can be "metis" or "scotch".
        :param path: Optional: Path of the partition file and other files that will be created.
        :param prefix: Optional: Prefix for all files used.
        :param comm_onesided: Optional: Binary option to enable one-sided communication for all subdivisions.
        """
        # Load the subdivisions serialized in the pre-process.
        self.subdivisions = self.load_subdivisions(path=path, prefix=prefix)
        self.path = path
        self.prefix = prefix

        # Load the partitioning file:
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

        # Register all subdivisions to each other.
        # Needed to easily communicate / copy between the local (same partition) subdivisions.
        for sd in self.subdivisions:
            sd.neighbor_list = self.subdivisions

        # Enable one-sided communication if the option is provided.
        if comm_onesided:
            for sd in self.subdivisions:
                sd.onesided = True

    @staticmethod
    def load_subdivisions(path="", prefix=""):
        """ Small helper function to de-serialize the subdivisions created in the pre-process.

        :param path: Path to the subdivisions pickle file.
        :param prefix: Prefix of the subdivision pickle file.
        :return: List of subdivisions from the pickle file.
        """
        with open(path + prefix + "subdivisions.pkl", "rb") as f:
            return pickle.load(f)

    def register_field(self, fieldname, halo, field_ic_file=None, field_bc_file=None,
                       staggered=None, haloincluded=False, singlefile=True):
        """ Function to register fields to subdivisions. Delegates to DomainSubdivision.register_field().

        :param fieldname: Name of the field.
        :param halo: Size of the halo in each direction: array of size 6:
        [negative x-direction, positive x-direction,
         negative y-direction, positive y-direction,
         negative z-direction, positive z-direction] halo size.
        :param field_ic_file: Optional: Initial condition / values array file.
        If none is provided initial values are zero.
        :param field_bc_file: Optional: Boundary value file. In cases were the boundary condition is provided in a file.
        If none is provided the global boundary condition needs to be registered and set manually.
        :param staggered: Optional: Parameter for staggered grids: array of size 3:
        [x-direction, y-direction, z-direction] reduces field size by array value.
        Should only have 0 or 1 in the array to be supported by GridTools.
        :param haloincluded: Optional: Binary option if the initial value field includes the global halo or not.
        :return: None
        """
        for sd in self.subdivisions:
            sd.register_field(fieldname, halo, field_ic_file, field_bc_file, staggered, haloincluded, singlefile)

    def register_stencil(self, **kwargs):
        """ Function to register GridTools4Py stencils. Delegates to DomainSubdivision.register_stencil().

        :param kwargs: keyword arguments almost the same as kwargs for usual GridTools4Py stencil instantiation.
        Accepts keywords: definitions_func, inputs, constant_inputs, global_inputs, outputs,
        domain, mode, vertical_direction, and the new keyword reductions.

        :return: List of instantiated GridTools4Py stencil for all subdivisions.
        """
        dds = DomainDecompositionStencil()
        for sd in self.subdivisions:
            dds.subdiv_stencil_list.append(sd.register_stencil(**kwargs))
        return dds

    def communicate(self, fieldname=None):
        """ Function to start the boundary exchange between subdivisions. Delegates to DomainSubdivision.communicate().

        :param fieldname: Name of the field. If none is provided communicates all registered fields.
        :return: None
        """
        for sd in self.subdivisions:
            sd.communicate(fieldname)

    def set_boundary_condition(self, fieldname, direction, halo, array):
        """ Function to set an input array as the global boundary condition for a specific field and direction.
        Manages the indices of the global array for each subdivision.
        Delegates to DomainSubdivision.set_boundary_condition() with subdivision specific part of the array.

        :param fieldname: Name of the field to set the global boundary condition.
        :param direction: Direction of the global boundary condition.
        :param array: Input array to be the global boundary condition.
        :return: None
        """
        for sd in self.subdivisions:
            if (direction == 0 or direction == 1):
                slice = np.s_[:halo, sd.global_coords[2]:sd.global_coords[3], sd.global_coords[4]:sd.global_coords[5]]
            elif (direction == 2 or direction == 3):
                slice = np.s_[sd.global_coords[0]:sd.global_coords[1], :halo, sd.global_coords[4]:sd.global_coords[5]]
            elif (direction == 4 or direction == 5):
                slice = np.s_[sd.global_coords[0]:sd.global_coords[1], sd.global_coords[2]:sd.global_coords[3], :halo]

            sd.set_boundary_condition(fieldname, direction, array[slice])

    def apply_boundary_condition(self, fieldname):
        """ Function to apply the set boundary condition of a field.
        Delegates to DomainSubdivision.apply_boundary_condition().

        :param fieldname: Name of the field.
        :return: None
        """
        for sd in self.subdivisions:
            sd.apply_boundary_condition(fieldname)

    def save_fields(self, fieldnames=None, postfix=None):
        """ Function to save field into file per subdivision.

        :param fieldnames: List of names of the fields to save.
        :param postfix: Postfix to apply to all files, e.g. time step identifier.
        :return: None
        """
        for sd in self.subdivisions:
            sd.save_fields(fieldnames, path=self.path, prefix=self.prefix, postfix=postfix)

    def swap_fields(self, field1, field2):
        """ Function to easily swap the content of two fields.
        Used in a lot of stencil codes to swap between old and new array values between time steps / computations.

        :param field1: First field to be swapped.
        :param field2: Second field to be swapped.
        :return: None
        """
        for sd in self.subdivisions:
            sd.swap_fields(field1, field2)

