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
import matplotlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA
from matplotlib.cbook import _backports
from collections import defaultdict
import types

# BEGIN: monkeypatch for internal faces in 3D plot:
# Source: https://stackoverflow.com/questions/48672663/matplotlib-render-all-internal-voxels-with-alpha
def voxels(self, *args, **kwargs):

    if len(args) >= 3:
        # underscores indicate position only
        def voxels(__x, __y, __z, filled, **kwargs):
            return (__x, __y, __z), filled, kwargs
    else:
        def voxels(filled, **kwargs):
            return None, filled, kwargs

    xyz, filled, kwargs = voxels(*args, **kwargs)

    # check dimensions
    if filled.ndim != 3:
        raise ValueError("Argument filled must be 3-dimensional")
    size = np.array(filled.shape, dtype=np.intp)

    # check xyz coordinates, which are one larger than the filled shape
    coord_shape = tuple(size + 1)
    if xyz is None:
        x, y, z = np.indices(coord_shape)
    else:
        x, y, z = (_backports.broadcast_to(c, coord_shape) for c in xyz)

    def _broadcast_color_arg(color, name):
        if np.ndim(color) in (0, 1):
            # single color, like "red" or [1, 0, 0]
            return _backports.broadcast_to(
                color, filled.shape + np.shape(color))
        elif np.ndim(color) in (3, 4):
            # 3D array of strings, or 4D array with last axis rgb
            if np.shape(color)[:3] != filled.shape:
                raise ValueError(
                    "When multidimensional, {} must match the shape of "
                    "filled".format(name))
            return color
        else:
            raise ValueError("Invalid {} argument".format(name))

    # intercept the facecolors, handling defaults and broacasting
    facecolors = kwargs.pop('facecolors', None)
    if facecolors is None:
        facecolors = self._get_patches_for_fill.get_next_color()
    facecolors = _broadcast_color_arg(facecolors, 'facecolors')

    # broadcast but no default on edgecolors
    edgecolors = kwargs.pop('edgecolors', None)
    edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')

    # include possibly occluded internal faces or not
    internal_faces = kwargs.pop('internal_faces', False)

    # always scale to the full array, even if the data is only in the center
    self.auto_scale_xyz(x, y, z)

    # points lying on corners of a square
    square = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0]
    ], dtype=np.intp)

    voxel_faces = defaultdict(list)

    def permutation_matrices(n):
        """ Generator of cyclic permutation matices """
        mat = np.eye(n, dtype=np.intp)
        for i in range(n):
            yield mat
            mat = np.roll(mat, 1, axis=0)

    for permute in permutation_matrices(3):
        pc, qc, rc = permute.T.dot(size)
        pinds = np.arange(pc)
        qinds = np.arange(qc)
        rinds = np.arange(rc)

        square_rot = square.dot(permute.T)

        for p in pinds:
            for q in qinds:
                p0 = permute.dot([p, q, 0])
                i0 = tuple(p0)
                if filled[i0]:
                    voxel_faces[i0].append(p0 + square_rot)

                # draw middle faces
                for r1, r2 in zip(rinds[:-1], rinds[1:]):
                    p1 = permute.dot([p, q, r1])
                    p2 = permute.dot([p, q, r2])
                    i1 = tuple(p1)
                    i2 = tuple(p2)
                    if filled[i1] and (internal_faces or not filled[i2]):
                        voxel_faces[i1].append(p2 + square_rot)
                    elif (internal_faces or not filled[i1]) and filled[i2]:
                        voxel_faces[i2].append(p2 + square_rot)

                # draw upper faces
                pk = permute.dot([p, q, rc-1])
                pk2 = permute.dot([p, q, rc])
                ik = tuple(pk)
                if filled[ik]:
                    voxel_faces[ik].append(pk2 + square_rot)

    # iterate over the faces, and generate a Poly3DCollection for each voxel
    polygons = {}
    for coord, faces_inds in voxel_faces.items():
        # convert indices into 3D positions
        if xyz is None:
            faces = faces_inds
        else:
            faces = []
            for face_inds in faces_inds:
                ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                face = np.empty(face_inds.shape)
                face[:, 0] = x[ind]
                face[:, 1] = y[ind]
                face[:, 2] = z[ind]
                faces.append(face)

        poly = art3d.Poly3DCollection(faces,
            facecolors=facecolors[coord],
            edgecolors=edgecolors[coord],
            **kwargs
        )
        self.add_collection3d(poly)
        polygons[coord] = poly

    return polygons
# END: monkeypatch for internal faces in 3D plot:

class VisualizeDomainDecomposition:
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

    def plot2D(self):
        fig, ax = plt.subplots()
        image = ax.imshow(self.values.T, cmap='Set1')

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

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cmap = matplotlib.cm.get_cmap('Set1')
        filled = np.ones(self.domain, dtype=np.bool)
        colors = np.empty(self.domain + [4], dtype=np.float32)
        line_color = [0.0, 0.0, 0.0, 0.3]
        for i in range(self.domain[0]):
            for j in range(self.domain[1]):
                for k in range(self.domain[2]):
                    partition = self.values[i, j, k]
                    colors[i, j, k] = cmap(partition / self.values.max())
                    if partition == 1:
                        colors[i, j, k, 3] = 0.8
                    else:
                        colors[i, j, k, 3] = 0.8

        ax.voxels = types.MethodType(voxels, ax)
        ax.voxels(filled, facecolors=colors, edgecolors='k', internal_faces=True, linewidth=0.5)

        ax.set_title("Domain decomposition of \n{0:d}x{1:d}x{2:d} subdivisions "
                     "into {3:d} partitions".format(self.domain[0], self.domain[1], self.domain[2],
                                                    1 + int(np.max(self.values))), loc="left")

        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1.5)

        # ax.set_xticks(np.arange(0.0, self.domain[0], 1))
        # ax.set_xticklabels(np.arange(0, self.domain[0], 1), size='small')
        #
        # ax.set_yticks(np.arange(0.0, self.domain[1], 1))
        # ax.set_yticklabels(np.arange(0, self.domain[1], 1), size='small')
        #
        ax.set_zticks(np.arange(0, self.domain[2]+1, 1))
        ax.set_zticklabels(np.arange(0, self.domain[2]+1, 1))
        ax.set_zlim(0, 1.5)

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

    vis = VisualizeDomainDecomposition(args.i, args.o, args.d, args.gp, args.v)

    if len(args.d) == 2:
        vis.plot2D()
    elif len(args.d) == 3:
        vis.plot3D()
    else:
        raise ValueError("Domain needs to be 2D or 3D")

