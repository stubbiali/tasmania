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
This module contains the domain decomposition post-process functionality.
Specifically, it contains functions to combine the output fields saved by each subdivision
into a single file per field.
"""

import numpy as np
import pickle
import os
from datetime import timedelta

from domain_decomposition import DomainSubdivision


class DomainPostprocess:
    """ Class containing all post-process functions and storing the subdivision information during post-process.
    """
    def __init__(self, path="", prefix=""):
        """ Load the subdivision information from file and store it in the class.
        Store the path and prefix parameters as well.

        :param path: Optional: Path where all the output should be saved.
        :param prefix: Optional: Prefix for all output file names.
        """
        with open(path + prefix + "subdivisions.pkl", "rb") as f:
            self.subdivisions = pickle.load(f)

        self.path = path
        self.prefix = prefix

    def create_pickle_dump(self, nx, ny, nz, nt, domain, dt, eps, save_freq, filename, cleanup=False):
        """ Helper function to create a specific pickle used in the Burger use cases.

        :param nx: Number of grid points in x-direction.
        :param ny: Number of grid points in y-direction.
        :param nz: Number of grid points in z-direction.
        :param nt: Number of time steps.
        :param domain: Extent of the domain.
        :param dt: Time step size.
        :param eps: Diffusion coefficient.
        :param save_freq: Number of time steps between save points.
        :param filename: Name of the file to create.
        :param cleanup: Optional, binary flag: If True deletes the temporary files created by each subdivision
        after processing them.
        :return: None
        """
        # Create the grid
        x = np.linspace(domain[0][0], domain[1][0], nx)
        y = np.linspace(domain[0][1], domain[1][1], ny)

        tsave = [timedelta(0), ]
        usave = self.combine_output_files(size=[nx, ny, nz], fieldname="unew",
                                          path=self.path, prefix=self.prefix,
                                          postfix="t_"+str(0), save=False, cleanup=cleanup)
        vsave = self.combine_output_files(size=[nx, ny, nz], fieldname="vnew",
                                          path=self.path, prefix=self.prefix,
                                          postfix="t_"+str(0), save=False, cleanup=cleanup)
        for n in range(1, nt+1):
            # Save
            if ((save_freq > 0) and (n % save_freq == 0)) or (n == nt):
                unew = self.combine_output_files(size=[nx, ny, nz], fieldname="unew",
                                                 path=self.path, prefix=self.prefix,
                                                 postfix="t_"+str(n), save=False, cleanup=cleanup)
                vnew = self.combine_output_files(size=[nx, ny, nz], fieldname="vnew",
                                                 path=self.path, prefix=self.prefix,
                                                 postfix="t_"+str(n), save=False, cleanup=cleanup)
                tsave.append(timedelta(seconds=n * dt))
                usave = np.concatenate((usave, unew), axis=2)
                vsave = np.concatenate((vsave, vnew), axis=2)

        # Dump solution to a binary file
        with open(self.path + self.prefix + filename, 'wb') as outfile:
            pickle.dump(tsave, outfile)
            pickle.dump(x, outfile)
            pickle.dump(y, outfile)
            pickle.dump(usave, outfile)
            pickle.dump(vsave, outfile)
            pickle.dump(eps, outfile)

    def combine_output_files(self, size, fieldname, path="", prefix="", postfix=None, save=True, cleanup=False):
        """ Main post-processing function. Collect the files created by each subdivision into a single numpy array
        and then save it as a  file.

        :param size: Size of the domain in each dimension.
        :param fieldname: Name of the field to combine.
        :param path: Optional: Path where all the output should be saved.
        :param prefix: Optional: Prefix for all output file names.
        :param postfix: Optional: Postfix for the input files e.g. time step at save point.
        :param save: Optional, binary flag: If True saves the combined array to file.
        :param cleanup: Optional, binary flag: If True deletes the input files after combining them.
        :return: None
        """
        field = np.zeros((size[0], size[1], size[2]))
        for sd in self.subdivisions:
            filename = (path + prefix + str(fieldname) + "_from_"
                        + "x" + str(sd.global_coords[0])
                        + "y" + str(sd.global_coords[2])
                        + "z" + str(sd.global_coords[4])
                        + "_to_"
                        + "x" + str(sd.global_coords[1] - 1)
                        + "y" + str(sd.global_coords[3] - 1)
                        + "z" + str(sd.global_coords[5] - 1))
            if postfix is not None:
                filename += "_" + str(postfix) + ".npy"
            else:
                filename += ".npy"

            field[sd.global_coords[0]:sd.global_coords[1],
                  sd.global_coords[2]:sd.global_coords[3],
                  sd.global_coords[4]:sd.global_coords[5]] = np.load(filename, mmap_mode='r')[:, :, :]
        if save:
            np.save(fieldname, field)

        if cleanup:
            for sd in self.subdivisions:
                filename = (path + prefix + str(fieldname) + "_from_"
                            + "x" + str(sd.global_coords[0])
                            + "y" + str(sd.global_coords[2])
                            + "z" + str(sd.global_coords[4])
                            + "_to_"
                            + "x" + str(sd.global_coords[1] - 1)
                            + "y" + str(sd.global_coords[3] - 1)
                            + "z" + str(sd.global_coords[5] - 1))
                if postfix is not None:
                    filename += "_" + str(postfix) + ".npy"
                else:
                    filename += ".npy"
                os.remove(filename)
        return field
