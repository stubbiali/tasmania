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
import unittest
import os
import numpy as np
import pickle
from mpi4py import MPI

import gridtools as gt
from domain_decomposition import DomainDecomposition, DomainSubdivision, DomainPartitions
from dd_postprocess import DomainPostprocess
from dd_preprocess import DomainPreprocess


def test_stencil_mult(const, in_u):
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = in_u[i, j, k] * const

    return out_u


def test_stencil_minus(const, in_u):
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = in_u[i, j, k] - const

    return out_u


def test_stencil_5point(in_u):
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = (in_u[i, j, k]
                      + in_u[i - 1, j, k] + in_u[i + 1, j, k]
                      + in_u[i, j - 1, k] + in_u[i, j + 1, k]
                      + in_u[i, j, k - 1] + in_u[i, j, k + 1])

    return out_u


def test_stencil_9point(in_u):
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = (in_u[i, j, k]
                      + in_u[i - 1, j, k] + in_u[i + 1, j, k]
                      + in_u[i, j - 1, k] + in_u[i, j + 1, k]
                      + in_u[i, j, k - 1] + in_u[i, j, k + 1]
                      + in_u[i - 2, j, k] + in_u[i + 2, j, k]
                      + in_u[i, j - 2, k] + in_u[i, j + 2, k]
                      + in_u[i, j, k - 2] + in_u[i, j, k + 2]
                      )

    return out_u


class TestDDPreprocess(unittest.TestCase):
    def setUp(self):
        pass

    def test_register_field(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            domain = np.array([100, 50, 4])
            slices = np.array([4, 2, 2])
            periodic = np.array([1, 0, 0])
            ddc = DomainPreprocess(domain=domain,
                                   periodic=periodic,
                                   subdivs_per_dim=slices,
                                   path="",
                                   prefix="",
                                   fileoutput="both")

            ddc.add_stencil({"unow": [[2], [2], [2], [2], [0], [0]]})
            ddc.add_stencil({"vnow": [[2], [2], [2], [2], [0], [0]]})
            ddc.preprocess()
            ddc.pymetis_partitioning(2)

            prepared_domain = DomainDecomposition("subdomains_pymetis.dat.part." + str(2), "metis",
                                                  path="", prefix="", comm_onesided=False)
            # Generate initial conditions file:
            ic = np.linspace(0,
                             domain[0] * domain[1] * domain[2] - 1,
                             domain[0] * domain[1] * domain[2]).reshape((domain[0],
                                                                         domain[1],
                                                                         domain[2]))
            np.save("test_initial_conditions", ic)

            halo = [2, 2, 2, 2, 0, 0]
            prepared_domain.register_field(fieldname="unow",
                                           halo=halo,
                                           field_ic_file="test_initial_conditions.npy")

            if MPI.COMM_WORLD.Get_size() == 1:
                values = [3041, 3043, 3141, 3143, 8041, 8043, 8141, 8143, 13041,
                          13043, 13141, 13143, 18041, 18043, 18141, 18143]
            elif MPI.COMM_WORLD.Get_size() == 2:
                values = [3141, 3143, 8141, 8143, 13141, 13143, 18141, 18143]
            i = 0
            for sd in prepared_domain.subdivisions:
                # print(int(sd.get_interior_field("unow")[15, 10, 1]))
                assert int(sd.get_interior_field("unow")[15, 10, 1]) == values[i]
                i += 1

    def test_register_stencil(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            dummy_subdiv = DomainSubdivision(id=0, pid=0, size=np.array([16, 8, 1]),
                                             global_coords=np.array([0, 16, 0, 8, 0, 1]),
                                             neighbors_id=[2, 1, 6, 3, None, None])

            dummy_subdiv.register_field(fieldname="unow", halo=[0, 0, 0, 0, 0, 0])
            dummy_subdiv.register_field(fieldname="vnow", halo=[0, 0, 0, 0, 0, 0])
            dummy_subdiv.register_field(fieldname="unew", halo=[0, 0, 0, 0, 0, 0])
            dummy_subdiv.register_field(fieldname="vnew", halo=[0, 0, 0, 0, 0, 0])

            test_stencil = dummy_subdiv.register_stencil(
                definitions_func="definitions_func_",
                inputs={"in_u": "unow", "in_v": "vnow"},
                global_inputs={"dt": "dt_", "dx": "dx_", "dy": "dy_", "eps": "eps_"},
                outputs={"out_u": "unew", "out_v": "vnew"},
                domain="domain_",
                mode=gt.mode.NUMPY
            )

            assert type(test_stencil) == gt.user_interface.ngstencil.NGStencil

    def test_compute_local_subdivs(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([8, 8, 1]),
                                        global_coords=np.array([0, 8, 0, 8, 0, 1]),
                                        neighbors_id=np.array([None, 1, None, None, None]))

            subdiv1 = DomainSubdivision(id=1, pid=0, size=np.array([8, 8, 1]),
                                        global_coords=np.array([8, 16, 8, 16, 0, 1]),
                                        neighbors_id=np.array([0, None, None, None, None]))

            DomainPartitions.domain_partitions = np.array([0, 0])

            slist = [subdiv0, subdiv1]
            subdiv0.register_field(fieldname="unow", halo=[0, 0, 0, 0, 0, 0])
            subdiv0.register_field(fieldname="unew", halo=[0, 0, 0, 0, 0, 0])
            subdiv1.register_field(fieldname="unow", halo=[0, 0, 0, 0, 0, 0])
            subdiv1.register_field(fieldname="unew", halo=[0, 0, 0, 0, 0, 0])
            subdiv0.fields["unow"] = np.linspace(0, 63, 64).reshape((8, 8, 1))
            subdiv0.fields["unew"] = np.zeros(64).reshape((8, 8, 1))
            subdiv1.fields["unow"] = np.linspace(64, 127, 64).reshape((8, 8, 1))
            subdiv1.fields["unew"] = np.zeros(64).reshape((8, 8, 1))
            subdiv0.register_field(fieldname="vnow", halo=[0, 0, 0, 0, 0, 0])
            subdiv0.register_field(fieldname="vnew", halo=[0, 0, 0, 0, 0, 0])
            subdiv1.register_field(fieldname="vnow", halo=[0, 0, 0, 0, 0, 0])
            subdiv1.register_field(fieldname="vnew", halo=[0, 0, 0, 0, 0, 0])
            subdiv0.fields["vnow"] = np.linspace(0, 63, 64).reshape((8, 8, 1))
            subdiv0.fields["vnew"] = np.zeros(64).reshape((8, 8, 1))
            subdiv1.fields["vnow"] = np.linspace(64, 127, 64).reshape((8, 8, 1))
            subdiv1.fields["vnew"] = np.zeros(64).reshape((8, 8, 1))

            domain_ = gt.domain.Rectangle((0, 0, 0), (64, 64, 0))
            # Convert global inputs to GT4Py Global"s
            test_const = 2.0
            test_const = gt.Global(test_const)

            st_list = []

            for sd in slist:
                st_list.append(sd.register_stencil(definitions_func=test_stencil_mult,
                                                   inputs={"in_u": "unow"},
                                                   global_inputs={"const": test_const},
                                                   outputs={"out_u": "unew"},
                                                   domain=domain_,
                                                   mode=gt.mode.NUMPY))

            for sd in st_list:
                sd.compute()

            values = [2 * np.linspace(0, 63, 64).reshape(8, 8, 1), 2 * np.linspace(64, 127, 64).reshape(8, 8, 1)]
            i = 0
            for sd in slist:
                # sd.save_fields()
                assert (sd.get_interior_field("unew") == values[i]).all()
                i += 1

    def test_communicate_locally(self):
        if MPI.COMM_WORLD.Get_rank() == 0:
            DomainPartitions.domain_partitions = np.array([0, 0])

            subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([5, 5, 2]),
                                        global_coords=np.array([0, 5, 0, 5, 0, 1]),
                                        neighbors_id=np.array([1, 1, 1, 1, 1, 1]))
            subdiv0.register_field(fieldname="unow", halo=[2, 2, 2, 2, 2, 2])
            subdiv0.register_field(fieldname="unew", halo=[2, 2, 2, 2, 2, 2])
            subdiv0.fields["unow"][2:-2, 2:-2, 2:-2] = np.linspace(0, 49, 50).reshape((5, 5, 2))
            subdiv0.fields["unow"][0, :, :] = -1
            subdiv0.fields["unow"][1, :, :] = -2
            subdiv0.fields["unow"][-2, :, :] = -3
            subdiv0.fields["unow"][-1, :, :] = -4
            subdiv0.fields["unow"][:, 0, :] = -5
            subdiv0.fields["unow"][:, 1, :] = -6
            subdiv0.fields["unow"][:, -2, :] = -7
            subdiv0.fields["unow"][:, -1, :] = -8
            subdiv0.fields["unow"][:, :, 0] = -9
            subdiv0.fields["unow"][:, :, 1] = -10
            subdiv0.fields["unow"][:, :, -2] = -11
            subdiv0.fields["unow"][:, :, -1] = -12
            subdiv0.fields["unew"][2:-2, 2:-2, 2:-2] = np.zeros(50).reshape((5, 5, 2))
            subdiv1 = DomainSubdivision(id=1, pid=0, size=np.array([5, 5, 2]),
                                        global_coords=np.array([5, 10, 5, 10, 0, 1]),
                                        neighbors_id=np.array([0, 0, 0, 0, 0, 0]))
            subdiv1.register_field(fieldname="unow", halo=[2, 2, 2, 2, 2, 2])
            subdiv1.register_field(fieldname="unew", halo=[2, 2, 2, 2, 2, 2])
            subdiv1.fields["unow"][2:-2, 2:-2, 2:-2] = np.linspace(50, 99, 50).reshape((5, 5, 2))
            subdiv1.fields["unew"][2:-2, 2:-2, 2:-2] = np.zeros(50).reshape((5, 5, 2))

            slist = [subdiv0, subdiv1]
            subdiv0.neighbor_list = slist
            subdiv1.neighbor_list = slist

            domain_ = gt.domain.Rectangle((2, 2, 2), (10, 10, 4))

            st_list = []
            for sd in slist:
                st_list.append(sd.register_stencil(definitions_func=test_stencil_9point,
                                                   inputs={"in_u": "unow"},
                                                   outputs={"out_u": "unew"},
                                                   domain=domain_,
                                                   mode=gt.mode.NUMPY))

            for sd in slist:
                sd.communicate(fieldname="unow")

            values = np.array([[[-9., -10., -5., -5., -11., -12.],
                                 [-9., -10.,  -6., -6., -11., -12.],
                                 [-9., -10.,  80., 81., -11., -12.],
                                 [-9., -10.,  82., 83., -11., -12.],
                                 [-9., -10.,  84., 85., -11., -12.],
                                 [-9., -10.,  86., 87., -11., -12.],
                                 [-9., -10.,  88., 89., -11., -12.],
                                 [-9., -10.,  -7., -7., -11., -12.],
                                 [-9., -10., -8., -8., -11., -12.]],
                                 [[-9., -10.,  -5.,  -5., -11., -12.],
                                  [-9., -10.,  -6.,  -6., -11., -12.],
                                  [-9., -10.,  90.,  91., -11., -12.],
                                  [-9., -10.,  92.,  93., -11., -12.],
                                  [-9., -10.,  94.,  95., -11., -12.],
                                  [-9., -10.,  96.,  97., -11., -12.],
                                  [-9., -10.,  98.,  99., -11., -12.],
                                  [-9., -10.,  -7.,  -7., -11., -12.],
                                  [-9., -10.,  -8.,  -8., -11., -12.]],
                                 [[-9., -10.,  56.,   0., -11., -12.],
                                  [-9., -10.,  58.,   0., -11., -12.],
                                  [50.,  51.,   0.,   1.,  50.,  51.],
                                  [52.,  53.,   2.,   3.,  52.,  53.],
                                  [54.,  55.,   4.,   5.,  54.,  55.],
                                  [56.,  57.,   6.,   7.,  56.,  57.],
                                  [58.,  59.,   8.,   9.,  58.,  59.],
                                  [-9., -10.,  50.,  51., -11., -12.],
                                  [-9., -10.,  52.,  53., -11., -12.]],
                                 [[-9., -10.,  66.,   0., -11., -12.],
                                  [-9., -10.,  68.,   0., -11., -12.],
                                  [60.,  61.,  10.,  11.,  60.,  61.],
                                  [62.,  63.,  12.,  13.,  62.,  63.],
                                  [64.,  65.,  14.,  15.,  64., 65.],
                                  [66.,  67.,  16.,  17.,  66.,  67.],
                                  [68.,  69.,  18.,  19.,  68.,  69.],
                                  [-9., -10.,  60.,  61., -11., -12.],
                                  [-9., -10.,  62.,  63., -11., -12.]],
                                 [[-9., -10.,  76.,   0., -11., -12.],
                                  [-9., -10.,  78.,   0., -11., -12.],
                                  [70.,  71.,  20.,  21.,  70.,  71.],
                                  [72.,  73.,  22.,  23.,  72.,  73.],
                                  [74.,  75.,  24.,  25.,  74.,  75.],
                                  [76.,  77.,  26.,  27.,  76.,  77.],
                                  [78.,  79.,  28.,  29.,  78.,  79.],
                                  [-9., -10.,  70.,  71., -11., -12.],
                                  [-9., -10.,  72.,  73., -11., -12.]],
                                 [[-9., -10.,  86.,   0., -11., -12.],
                                  [-9., -10.,  88.,   0., -11., -12.],
                                  [80.,  81.,  30.,  31.,  80.,  81.],
                                  [82.,  83.,  32.,  33.,  82.,  83.],
                                  [84.,  85.,  34.,  35.,  84.,  85.],
                                  [86.,  87.,  36.,  37.,  86.,  87.],
                                  [88.,  89.,  38.,  39.,  88.,  89.],
                                  [-9., -10.,  80.,  81., -11., -12.],
                                  [-9., -10.,  82.,  83., -11., -12.]],
                                 [[-9., -10.,  96.,   0., -11., -12.],
                                  [-9., -10.,  98.,   0., -11., -12.],
                                  [90.,  91.,  40.,  41.,  90.,  91.],
                                  [92.,  93.,  42.,  43.,  92.,  93.],
                                  [94.,  95.,  44.,  45.,  94.,  95.],
                                  [96.,  97.,  46.,  47.,  96.,  97.],
                                  [98.,  99.,  48.,  49.,  98.,  99.],
                                  [-9., -10.,  90.,  91., -11., -12.],
                                  [-9., -10.,  92.,  93., -11., -12.]],
                                 [[-9., -10.,  -5.,  -5., -11., -12.],
                                  [-9., -10.,  -6.,  -6., -11., -12.],
                                  [-9., -10.,  50.,  51., -11., -12.],
                                  [-9., -10.,  52.,  53., -11., -12.],
                                  [-9., -10.,  54.,  55., -11., -12.],
                                  [-9., -10.,  56.,  57., -11., -12.],
                                  [-9., -10.,  58.,  59., -11., -12.],
                                  [-9., -10.,  -7.,  -7., -11., -12.],
                                  [-9., -10.,  -8.,  -8., -11., -12.]],
                                 [[-9., -10.,  -5.,  -5., -11., -12.],
                                  [-9., -10.,  -6.,  -6., -11., -12.],
                                  [-9., -10.,  60.,  61., -11., -12.],
                                  [-9., -10.,  62.,  63., -11., -12.],
                                  [-9., -10.,  64.,  65., -11., -12.],
                                  [-9., -10.,  66.,  67., -11., -12.],
                                  [-9., -10.,  68.,  69., -11., -12.],
                                  [-9., -10.,  -7.,  -7., -11., -12.],
                                  [-9., -10.,  -8.,  -8., -11., -12.]]])
            for sd in slist:
                for k, v in sd.fields.items():
                    if k == "unow":
                        if sd.id == 0:
                            assert (v[:] == values[:]).all()

    def test_communicate_two_way(self):
        assert MPI.COMM_WORLD.Get_size() == 2

        DomainPartitions.domain_partitions = np.array([0, 1])

        size_x = 2
        size_y = 2
        size_z = 1
        tot_size = size_x * size_y * size_z
        hxm = 1
        hxp = 1
        hym = 1
        hyp = 1
        hzm = 1
        hzp = 1

        if MPI.COMM_WORLD.Get_rank() % 2 == 0:
            # Prepare boundary condition file:
            bc = -1.0 * np.arange((size_x + size_x + hxm + hxp) * (size_y + hym + hyp) * (size_z + hzm + hzp)).reshape(
                (size_x + size_x + hxm + hxp), (size_y + hym + hyp), (size_z + hzm + hzp)
            )
            np.save("test_boundary_condition.npy", bc)

        MPI.COMM_WORLD.Barrier()

        if MPI.COMM_WORLD.Get_rank() % 2 == 0:
            subdiv0 = DomainSubdivision(id=0,
                                        pid=0,
                                        size=np.array([size_x, size_y, size_z]),
                                        global_coords=np.array([0, size_x, 0, size_y, 0, size_z]),
                                        neighbors_id=np.array([None, 1, None, None, None, None]))
            subdiv0.onesided = False
            slist = [subdiv0]
            subdiv0.register_field(fieldname="unow", halo=[hxm, hxp, hym, hyp, hzm, hzp])
            subdiv0.register_field(fieldname="unew", halo=[hxm, hxp, hym, hyp, hzm, hzp])
            subdiv0.fields["unow"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.linspace(0, tot_size - 1, tot_size).reshape(
                (size_x, size_y, size_z))
            subdiv0.fields["unew"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.zeros(tot_size).reshape((size_x, size_y, size_z))
        else:
            subdiv1 = DomainSubdivision(id=1,
                                        pid=1,
                                        size=np.array([size_x, size_y, size_z]),
                                        global_coords=np.array([size_x, size_x + size_x, 0, size_y, 0, size_z]),
                                        neighbors_id=np.array([0, None, None, None, None, None]))
            subdiv1.onesided = False
            slist = [subdiv1]
            subdiv1.register_field(fieldname="unow", halo=[hxm, hxp, hym, hyp, hzm, hzp])  # ,
            # field_bc_file="test_boundary_condition.npy")
            subdiv1.register_field(fieldname="unew", halo=[hxm, hxp, hym, hyp, hzm, hzp])
            subdiv1.fields["unow"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.linspace(tot_size,
                                                                               tot_size + tot_size - 1,
                                                                               tot_size).reshape(
                (size_x, size_y, size_z)
            )
            subdiv1.fields["unew"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.zeros(tot_size).reshape((size_x, size_y, size_z))

        domain_ = gt.domain.Rectangle((hxm, hym, hzm), (size_x + size_x, size_y, size_z))

        st_list = []
        for sd in slist:
            st_list.append(sd.register_stencil(definitions_func=test_stencil_5point,
                                               inputs={"in_u": "unow"},
                                               outputs={"out_u": "unew"},
                                               domain=domain_,
                                               mode=gt.mode.NUMPY))

        westboundary = -1.0 * np.ones((hxm, size_y, size_z))
        eastboundary = -2.0 * np.ones((hxp, size_y, size_z))
        northboundary = -3.0 * np.ones((size_x, hym, size_z))
        southboundary = -4.0 * np.ones((size_x, hyp, size_z))
        lowerboundary = -5.0 * np.ones((size_x, size_y, hzm))
        upperboundary = -6.0 * np.ones((size_x, size_y, hzp))

        for sd in slist:
            sd.set_boundary_condition("unow", 0, westboundary)
            sd.set_boundary_condition("unow", 1, eastboundary)
            sd.set_boundary_condition("unow", 2, northboundary)
            sd.set_boundary_condition("unow", 3, southboundary)
            sd.set_boundary_condition("unow", 4, lowerboundary)
            sd.set_boundary_condition("unow", 5, upperboundary)

        for sd in slist:
            sd.apply_boundary_condition("unow")

        for sd in slist:
            sd.communicate(fieldname="unow")

        for sd in st_list:
            sd.compute()

        unow_values = [np.array([[[0.,0.,0.,0.], [0.,-5.,-5.,0.], [0.,-5.,-5.,0.], [0.,0.,0.,0.]],
                                 [[0.,-3.,-3.,0.], [-1.,0.,2.,4.], [-1.,1.,3.,5.], [0.,-4.,-4.,0.]],
                                 [[0.,0.,0.,0.], [0.,-6.,-6.,0.], [0.,-6.,-6.,0.], [0.,0.,0.,0.]]]),
                       np.array([[[[0.,0.,0.,0.], [0.,-5.,-5.,0.], [0.,-5.,-5.,0.], [0.,0.,0.,0.]],
                                  [[0.,-3.,-3.,0.], [2.,4.,6.,-2.], [3.,5.,7.,-2.], [0.,-4.,-4.,0.]],
                                  [[0.,0.,0.,0.], [0.,-6.,-6.,0.], [0.,-6.,-6.,0.], [0.,0.,0.,0.]]]])]
        unew_values = [np.array([[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
                                 [[0., 0., 0., 0.], [0., -12.,-5., 0.], [0., -12.,-4., 0.], [0., 0., 0., 0.]],
                                 [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]),
                       np.array([[[0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.]],
                                 [[0.,0.,0.,0.], [0.,3.,1.,0.], [0.,4.,1.,0.], [0.,0.,0.,0.]],
                                 [[0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.]]])]

        for sd in slist:
            for k, v in sd.fields.items():
                if k == "unow":
                    assert (v.transpose()[:] == unow_values[sd.id][:]).all()
                if k == "unew":
                    assert (v.transpose()[:] == unew_values[sd.id][:]).all()
                # print(sd.id, k, v.transpose())

    def test_communicate_one_way(self):
        assert MPI.COMM_WORLD.Get_size() == 2

        DomainPartitions.domain_partitions = np.array([0, 1])

        size_x = 2
        size_y = 2
        size_z = 1
        tot_size = size_x * size_y * size_z
        hxm = 1
        hxp = 1
        hym = 1
        hyp = 1
        hzm = 1
        hzp = 1

        if MPI.COMM_WORLD.Get_rank() % 2 == 0:
            # Prepare boundary condition file:
            bc = -1.0 * np.arange((size_x + size_x + hxm + hxp) * (size_y + hym + hyp) * (size_z + hzm + hzp)).reshape(
                (size_x + size_x + hxm + hxp), (size_y + hym + hyp), (size_z + hzm + hzp)
            )
            np.save("test_boundary_condition.npy", bc)

        MPI.COMM_WORLD.Barrier()

        if MPI.COMM_WORLD.Get_rank() % 2 == 0:
            subdiv0 = DomainSubdivision(id=0,
                                        pid=0,
                                        size=np.array([size_x, size_y, size_z]),
                                        global_coords=np.array([0, size_x, 0, size_y, 0, size_z]),
                                        neighbors_id=np.array([None, 1, None, None, None, None]))
            subdiv0.onesided = True
            slist = [subdiv0]
            subdiv0.register_field(fieldname="unow", halo=[hxm, hxp, hym, hyp, hzm, hzp])
            subdiv0.register_field(fieldname="unew", halo=[hxm, hxp, hym, hyp, hzm, hzp])
            subdiv0.fields["unow"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.linspace(0, tot_size - 1, tot_size).reshape(
                (size_x, size_y, size_z))
            subdiv0.fields["unew"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.zeros(tot_size).reshape((size_x, size_y, size_z))
        else:
            subdiv1 = DomainSubdivision(id=1,
                                        pid=1,
                                        size=np.array([size_x, size_y, size_z]),
                                        global_coords=np.array([size_x, size_x + size_x, 0, size_y, 0, size_z]),
                                        neighbors_id=np.array([0, None, None, None, None, None]))
            subdiv1.onesided = True
            slist = [subdiv1]
            subdiv1.register_field(fieldname="unow", halo=[hxm, hxp, hym, hyp, hzm, hzp])  # ,
            # field_bc_file="test_boundary_condition.npy")
            subdiv1.register_field(fieldname="unew", halo=[hxm, hxp, hym, hyp, hzm, hzp])
            subdiv1.fields["unow"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.linspace(tot_size,
                                                                               tot_size + tot_size - 1,
                                                                               tot_size).reshape(
                (size_x, size_y, size_z)
            )
            subdiv1.fields["unew"][hxm:-hxp, hym:-hyp, hzm:-hzp] = np.zeros(tot_size).reshape((size_x, size_y, size_z))

        domain_ = gt.domain.Rectangle((hxm, hym, hzm), (size_x + size_x, size_y, size_z))

        st_list = []
        for sd in slist:
            st_list.append(sd.register_stencil(definitions_func=test_stencil_5point,
                                               inputs={"in_u": "unow"},
                                               outputs={"out_u": "unew"},
                                               domain=domain_,
                                               mode=gt.mode.NUMPY))

        westboundary = -1.0 * np.ones((hxm, size_y, size_z))
        eastboundary = -2.0 * np.ones((hxp, size_y, size_z))
        northboundary = -3.0 * np.ones((size_x, hym, size_z))
        southboundary = -4.0 * np.ones((size_x, hyp, size_z))
        lowerboundary = -5.0 * np.ones((size_x, size_y, hzm))
        upperboundary = -6.0 * np.ones((size_x, size_y, hzp))

        for sd in slist:
            sd.set_boundary_condition("unow", 0, westboundary)
            sd.set_boundary_condition("unow", 1, eastboundary)
            sd.set_boundary_condition("unow", 2, northboundary)
            sd.set_boundary_condition("unow", 3, southboundary)
            sd.set_boundary_condition("unow", 4, lowerboundary)
            sd.set_boundary_condition("unow", 5, upperboundary)

        for sd in slist:
            sd.apply_boundary_condition("unow")

        for sd in slist:
            sd.communicate(fieldname="unow")

        for sd in st_list:
            sd.compute()

        unow_values = [np.array([[[0.,0.,0.,0.], [0.,-5.,-5.,0.], [0.,-5.,-5.,0.], [0.,0.,0.,0.]],
                                 [[0.,-3.,-3.,0.], [-1.,0.,2.,4.], [-1.,1.,3.,5.], [0.,-4.,-4.,0.]],
                                 [[0.,0.,0.,0.], [0.,-6.,-6.,0.], [0.,-6.,-6.,0.], [0.,0.,0.,0.]]]),
                       np.array([[[[0.,0.,0.,0.], [0.,-5.,-5.,0.], [0.,-5.,-5.,0.], [0.,0.,0.,0.]],
                                  [[0.,-3.,-3.,0.], [2.,4.,6.,-2.], [3.,5.,7.,-2.], [0.,-4.,-4.,0.]],
                                  [[0.,0.,0.,0.], [0.,-6.,-6.,0.], [0.,-6.,-6.,0.], [0.,0.,0.,0.]]]])]
        unew_values = [np.array([[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
                                 [[0., 0., 0., 0.], [0., -12.,-5., 0.], [0., -12.,-4., 0.], [0., 0., 0., 0.]],
                                 [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]),
                       np.array([[[0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.]],
                                 [[0.,0.,0.,0.], [0.,3.,1.,0.], [0.,4.,1.,0.], [0.,0.,0.,0.]],
                                 [[0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,0.]]])]

        for sd in slist:
            for k, v in sd.fields.items():
                if k == "unow":
                    assert (v.transpose()[:] == unow_values[sd.id][:]).all()
                if k == "unew":
                    assert (v.transpose()[:] == unew_values[sd.id][:]).all()
                # print(sd.id, k, v.transpose())
