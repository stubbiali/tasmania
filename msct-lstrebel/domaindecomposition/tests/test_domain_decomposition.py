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
import numpy as np

from mpi4py import MPI

import gridtools as gt
from domain_decomposition import DomainDecomposition, DomainSubdivision, DomainPartitions, DomainPreprocess


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


def test_stencil_move_x(in_u):
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = in_u[i - 1, j, k]

    return out_u


def test_stencil_move_y(in_u):
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = in_u[i, j - 1, k]

    return out_u


def test_stencil_move_z(in_u):
    # Declare the indices
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the output field
    out_u = gt.Equation()

    # Step the solution
    out_u[i, j, k] = in_u[i, j, k - 1]

    return out_u


class TestDD(unittest.TestCase):
    def setUp(self):
        pass

    def test_partitioning(self):
        print("---------------TEST partitioning----------------------")
        domain = np.array([2048, 1024, 40])
        slices = np.array([16, 8, 1])
        periodic = np.array([1, 0, 0])
        nparts = 5

        ddc = DomainPreprocess(domain=domain, periodic=periodic, subdivs_per_dim=slices, fileoutput="both")
        ddc.add_stencil({"in": [[1], [1], [1], [1], [0], [0]]})
        ddc.preprocess()
        ddc.pymetis_partitioning(nparts, verbose=True)

    def test_add_stencil_patterns(self):
        print("---------------TEST add stencil patterns----------------------")
        domain = np.array([2048, 1024, 40])
        subdivs_per_dim = np.array([16, 8, 1])
        periodic = np.array([1, 0, 0])
        fileoutput = ""
        preproc = DomainPreprocess(domain, periodic, subdivs_per_dim, fileoutput)
        stencil1 = [[1], [1], [1], [1], [1], [1]]
        stencil2 = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        stencil3 = [[1, 4], [1, 4], [1, 4], [1, 4], [], []]

        preproc.add_stencil({"unow": stencil1, "vnow": stencil1})
        # print(preproc.stencil_field_patterns)
        # print(preproc.stencil_field_accesses)
        preproc.add_stencil({"unow": stencil2})
        # print(preproc.stencil_field_patterns)
        # print(preproc.stencil_field_accesses)
        preproc.add_stencil({"unow": stencil3})
        # print(preproc.stencil_field_patterns)
        # print(preproc.stencil_field_accesses)

        # preproc.add_stencil({"vnow": stencil1})
        # print(preproc.stencil_field_patterns)
        # preproc.add_stencil({"vnow": stencil2})
        # print(preproc.stencil_field_patterns)


        # print("stencils")
        # print(stencil1, stencil2, stencil3)
        # print(preproc.stencil_field_accesses)

        print(preproc.combined_accesses())
        print(preproc.halo_maximum_extent())

    def test_check_globalcoords(self):
        print("---------------TEST check global coords----------------------")
        dsubdiv = DomainSubdivision(id=0, pid=0, size=np.array([16, 8, 1]),
                                    global_coords=np.array([1, 8, 1, 8, 1, 8]), gridpoints=128,
                                    neighbors_id=[2, 1, 6, 3, None, None])

        assert (dsubdiv.check_globalcoords(2, 2, 2))
        assert not (dsubdiv.check_globalcoords(8, 2, 2))
        assert not (dsubdiv.check_globalcoords(2, 8, 2))
        assert not (dsubdiv.check_globalcoords(2, 2, 8))
        assert not (dsubdiv.check_globalcoords(0, 2, 2))
        assert not (dsubdiv.check_globalcoords(2, 0, 2))
        assert not (dsubdiv.check_globalcoords(2, 2, 0))

    def test_register_field(self):
        print("---------------TEST register field----------------------")
        # Generate initial conditions file:
        ic = np.linspace(0, 999, 1000).reshape((50, 20, 1))
        np.save("test_initial_conditions", ic)

        # Generate dummy classes
        subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([25, 20, 1]),
                                    global_coords=np.array([0, 25, 0, 20, 0, 1]), gridpoints=500,
                                    neighbors_id=[1, 1, None, None, None, None])

        subdiv1 = DomainSubdivision(id=1, pid=0, size=np.array([25, 20, 1]),
                                    global_coords=np.array([25, 50, 0, 20, 0, 1]), gridpoints=500,
                                    neighbors_id=[0, 0, None, None, None, None])

        # register test field with ic
        for sd in [subdiv0, subdiv1]:
            sd.register_field(fieldname="unow", halo=[1, 1, 1, 1, 0, 0], field_ic_file="test_initial_conditions.npy")

        # register test field without ic
        for sd in [subdiv0, subdiv1]:
            sd.register_field(fieldname="unew", halo=[1, 1, 1, 1, 0, 0])
        #
        # for sd in [subdiv0, subdiv1]:
        #     print(sd.fields["unow"])
        #     print("===================================")

        assert int(subdiv0.fields["unow"][21, 5, 0]) == 404
        assert int(subdiv0.fields["unew"][21, 5, 0]) == 0
        assert int(subdiv1.fields["unow"][21, 5, 0]) == 904
        assert int(subdiv1.fields["unew"][21, 5, 0]) == 0

    def test_register_stencil(self):
        print("---------------TEST register stencil----------------------")
        dummy_subdiv = DomainSubdivision(id=0, pid=0, size=np.array([16, 8, 1]),
                                         global_coords=np.array([0, 16, 0, 8, 0, 1]), gridpoints=128,
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

        print(test_stencil)

    def test_domain_partitions(self):
        print("---------------TEST domain partitions----------------------")

        fileinput = "../subdomains_pymetis.dat.part.5"
        DomainPartitions.load_partitions(fileinput, fileformat="metis")
        DomainPartitions.print_partitions()

        fileinput = "../subdomains_scotch.map"
        DomainPartitions.load_partitions(fileinput, fileformat="scotch")
        DomainPartitions.print_partitions()

    def test_communicate_partitions(self):
        print("---------------TEST communicate partitions----------------------")
        pass

    def test_subdivision_distribution(self):
        print("---------------TEST subdivision distribution--------------------")
        ddc = DomainDecomposition(fileinput="subdomains_pymetis.dat.part.5", fileinputformat="metis")

        for sd in ddc.subdivisions:
            print(sd.id, sd.partitions_id)

    def test_compute_local_subdivs(self):
        print("---------------TEST compute local subdivs----------------------")
        subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([8, 8, 1]),
                                    global_coords=np.array([0, 8, 0, 8, 0, 1]), gridpoints=64,
                                    neighbors_id=np.array([None, 1, None, None, None]))

        subdiv1 = DomainSubdivision(id=1, pid=0, size=np.array([8, 8, 1]),
                                    global_coords=np.array([8, 16, 8, 16, 0, 1]), gridpoints=64,
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

        for sd in slist:
            st_list.append(sd.register_stencil(definitions_func=test_stencil_minus,
                                               inputs={"in_u": "vnow"},
                                               global_inputs={"const": test_const},
                                               outputs={"out_u": "vnew"},
                                               domain=domain_,
                                               mode=gt.mode.NUMPY))

        # print(st_list)
        for sd in st_list:
            sd.compute()

        for sd in slist:
            # for k, v in sd.fields.items():
            #     print(k, v.reshape((64)))
            sd.save_fields()

    def test_communicate_locally(self):
        print("---------------TEST communicate locally----------------------")

        DomainPartitions.domain_partitions = np.array([0, 0])

        # if MPI.COMM_WORLD.Get_rank() % 2 == 0:
        #     subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([5, 5, 2]),
        #                                 global_coords=np.array([0, 5, 0, 5, 0, 1]), gridpoints=50,
        #                                 neighbors_id=np.array([1, 1, None, None, None, None]))
        #     slist = [subdiv0]
        #     subdiv0.register_field(fieldname="unow", halo=[2, 2, 2, 2, 2, 2])
        #     subdiv0.register_field(fieldname="unew", halo=[2, 2, 2, 2, 2, 2])
        #     subdiv0.fields["unow"][2:-2, 2:-2, 2:-2] = np.linspace(0, 49, 50).reshape((5, 5, 2))
        #     subdiv0.fields["unow"][0, :, :] = -1
        #     subdiv0.fields["unow"][1, :, :] = -2
        #     subdiv0.fields["unow"][-2, :, :] = -3
        #     subdiv0.fields["unow"][-1, :, :] = -4
        #     subdiv0.fields["unow"][:, 0, :] = -5
        #     subdiv0.fields["unow"][:, 1, :] = -6
        #     subdiv0.fields["unow"][:, -2, :] = -7
        #     subdiv0.fields["unow"][:, -1, :] = -8
        #     subdiv0.fields["unow"][:, :, 0] = -9
        #     subdiv0.fields["unow"][:, :, 1] = -10
        #     subdiv0.fields["unow"][:, :, -2] = -11
        #     subdiv0.fields["unow"][:, :, -1] = -12
        #     subdiv0.fields["unew"][2:-2, 2:-2, 2:-2] = np.zeros(50).reshape((5, 5, 2))
        # else:
        #     subdiv1 = DomainSubdivision(id=1, pid=0, size=np.array([5, 5, 2]),
        #                                 global_coords=np.array([5, 10, 5, 10, 0, 1]), gridpoints=50,
        #                                 neighbors_id=np.array([0, 0, None, None, None, None]))
        #     slist = [subdiv1]
        #     subdiv1.register_field(fieldname="unow", halo=[2, 2, 2, 2, 2, 2])
        #     subdiv1.register_field(fieldname="unew", halo=[2, 2, 2, 2, 2, 2])
        #     subdiv1.fields["unow"][2:-2, 2:-2, 2:-2] = np.linspace(50, 99, 50).reshape((5, 5, 2))
        #     subdiv1.fields["unew"][2:-2, 2:-2, 2:-2] = np.zeros(50).reshape((5, 5, 2))

        subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([5, 5, 2]),
                                    global_coords=np.array([0, 5, 0, 5, 0, 1]), gridpoints=50,
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
                                    global_coords=np.array([5, 10, 5, 10, 0, 1]), gridpoints=50,
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

        # for sd in slist:
        #     for k, v in sd.fields.items():
        #         print(k, v.reshape((7, 7, 3)).transpose())
        #
        # for sd in st_list:
        #     sd.compute()

        for sd in slist:
            sd.communicate(fieldname="unow")

        # for sd in st_list:
        #     sd.compute()
        #
        # for sd in slist:
        #     sd.swap_fields("unow", "unew")
        #
        # subdiv0.fields["unow"][:], subdiv1.fields["unow"][:] = subdiv0.fields["unew"][:], subdiv1.fields["unew"][:]
        #
        # for sd in slist:
        #     sd.communicate(new_fieldname="unow", old_fieldname="unow")
        #
        # for sd in st_list:
        #     sd.compute()

        # print("-------------------------------------")
        #
        # subdiv0.fields["unow"][:], subdiv1.fields["unow"][:] = subdiv0.fields["unew"][:], subdiv1.fields["unew"][:]
        #
        # for sd in st_list:
        #     sd.compute()
        #
        for sd in slist:
            print("id = " + str(sd.id))
            for k, v in sd.fields.items():
                if k == "unow":
                    print(k, v.transpose())

        # for sd in slist:
        #     print("id = " + str(sd.id))
        #     for k in sd.fields.keys():
        #         print(k, sd.get_interior_field(k).transpose())

    def test_communicate_two_way(self):
        print("---------------TEST communicate two way----------------------")
        DomainPartitions.domain_partitions = np.array([0, 1])

        if MPI.COMM_WORLD.Get_rank() % 2 == 0:
            subdiv0 = DomainSubdivision(id=0,
                                        pid=0,
                                        size=np.array([5, 5, 2]),
                                        global_coords=np.array([0, 5, 0, 5, 0, 1]),
                                        gridpoints=50,
                                        neighbors_id=np.array([None, 1, 1, 1, 1, 1]))
            slist = [subdiv0]
            subdiv0.register_field(fieldname="unow", halo=[2, 2, 2, 2, 2, 2])#,
                                   # field_bc_file="test_boundary_condition_12x12x6.npy")
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
        else:
            subdiv1 = DomainSubdivision(id=1,
                                        pid=1,
                                        size=np.array([5, 5, 2]),
                                        global_coords=np.array([5, 10, 5, 10, 0, 1]),
                                        gridpoints=50,
                                        neighbors_id=np.array([0, None, 0, 0, 0, 0]))
            slist = [subdiv1]
            subdiv1.register_field(fieldname="unow", halo=[2, 2, 2, 2, 2, 2])#,
                                   # field_bc_file="test_boundary_condition_12x12x6.npy")
            subdiv1.register_field(fieldname="unew", halo=[2, 2, 2, 2, 2, 2])
            subdiv1.fields["unow"][2:-2, 2:-2, 2:-2] = np.linspace(50, 99, 50).reshape((5, 5, 2))
            subdiv1.fields["unew"][2:-2, 2:-2, 2:-2] = np.zeros(50).reshape((5, 5, 2))

        domain_ = gt.domain.Rectangle((2, 2, 2), (10, 10, 4))

        st_list = []
        for sd in slist:
            st_list.append(sd.register_stencil(definitions_func=test_stencil_9point,
                                               inputs={"in_u": "unow"},
                                               outputs={"out_u": "unew"},
                                               domain=domain_,
                                               mode=gt.mode.NUMPY))

        # for sd in slist:
        #     for k, v in sd.fields.items():
        #         print(k, v.reshape((7, 7, 3)).transpose())
        #
        # for sd in st_list:
        #     sd.compute()

        for sd in slist:
            sd.communicate(fieldname="unow")

        for sd in st_list:
            sd.compute()
        #
        # for sd in slist:
        #     sd.swap_fields("unow", "unew")
        #
        # subdiv0.fields["unow"][:], subdiv1.fields["unow"][:] = subdiv0.fields["unew"][:], subdiv1.fields["unew"][:]
        #
        # for sd in slist:
        #     sd.communicate(new_fieldname="unow", old_fieldname="unow")
        #
        # for sd in st_list:
        #     sd.compute()

        # print("-------------------------------------")
        #
        # subdiv0.fields["unow"][:], subdiv1.fields["unow"][:] = subdiv0.fields["unew"][:], subdiv1.fields["unew"][:]
        #
        # for sd in st_list:
        #     sd.compute()
        #
        for sd in slist:
            for k, v in sd.fields.items():
                print(sd.id, k, v.transpose())

        # for sd in slist:
        #     print("id = " + str(sd.id))
        #     for k in sd.fields.keys():
        #         print(k, sd.get_interior_field(k).transpose())

if __name__ == "__main__":
    unittest.main()