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
import matplotlib.pyplot as plt
import numpy as np
import argparse


class VisualizeDomainDecomposition2D:
    def __init__(self, infile, outfile, domain, partitioner, preview=False):
        self.infile = infile
        self.outfile = outfile
        self.domain = domain
        self.partitioner = partitioner
        self.values = self.read_values_from_file()
        self.preview = preview

    def read_values_from_file(self):
        if self.partitioner == 'metis':
            return self.read_values_from_metis_file()
        elif self.partitioner == 'scotch':
            return self.read_values_from_scotch_file()
        else:
            raise ValueError("Only 'metis' or 'scotch' as partitioner allowed,"
                             " received {0} instead".format(self.partitioner))

    def read_values_from_metis_file(self):
        values = np.loadtxt(self.infile)
        values = values.reshape(self.domain)
        return values

    def read_values_from_scotch_file(self):
        load = np.loadtxt(self.infile, skiprows=1)
        values = load[:, 1]
        values = values.reshape(self.domain)
        return values

    def plot(self):
        fig, ax = plt.subplots()
        image = ax.imshow(self.values.T)

        ax.set_title("Domain decomposition of \n{0:d}x{1:d} subdivisions "
                     "into {2:d} partitions".format(self.domain[0], self.domain[1],
                                                    1 + int(np.max(self.values))), loc="left")
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1.5)
        ax.set_xticks(np.arange(-.5, self.domain[0], 1))
        ax.set_xticklabels('')
        ax.set_xticks(np.arange(0, self.domain[0], 1), minor=True)
        ax.set_xticklabels(np.arange(0, self.domain[0], 1), minor=True)

        ax.set_yticks(np.arange(-.5, self.domain[1], 1))
        ax.set_yticklabels('')
        ax.set_yticks(np.arange(0, self.domain[1], 1), minor=True)
        ax.set_yticklabels(np.arange(0, self.domain[1], 1), minor=True)

        if self.outfile is not None:
            plt.savefig(self.outfile + ".png")
        if self.preview:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize domain decomposition in a colored 2D grid format.")

    parser.add_argument("-i", required=True, type=str,
                        help="Graph partitioning file (metis or scotch format).")
    parser.add_argument("-gp", required=True, type=str,
                        help="Graph partitioner: metis or scotch")
    parser.add_argument('-d', nargs='+', type=int, required=True)
    parser.add_argument("-o", default=None, type=str,
                        help="Optional: Name for the output file.")
    parser.add_argument("-v", action="store_true",
                        help="Optional: Enable gui preview output")

    args = parser.parse_args()

    if len(args.d) != 2:
        raise ValueError("Domain needs to be 2D")

    vis = VisualizeDomainDecomposition2D(args.i, args.o, args.d, args.gp, args.v)
    vis.plot()

