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

import gridtools as gt
from domain_decomposition import DomainDecomposition, DomainSubdivision, DomainPartitions


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
        self.dummy_subdiv = DomainSubdivision(0, 0, np.array([16, 8, 1]),
                                              np.array([8, 8, 16, 16, 0, 0]), 128, [2, 1, 6, 3, None, None])

    def test_partitioning(self):
        domain = np.array([2048, 1024, 40])
        slices = np.array([16, 8, 1])
        stencil = np.array([1, 1, 1, 1, 0, 0])
        periodic = np.array([1, 0, 0])

        ddc = DomainDecomposition(domain=domain, periodic=periodic, subdivs_per_dim=slices,
                                  stencil_extent=stencil, preprocess=True, fileoutput="both", nparts=5)

    def test_register_stencil(self):
        self.dummy_subdiv.fields = {"unow": np.zeros(3), "vnow": np.zeros(3), "unew": np.zeros(3), "vnew": np.zeros(3)}
        self.dummy_subdiv.register_stencil(definitions_func="definitions_func_",
                                           inputs={"in_u": "unow", "in_v": "vnow"},
                                           global_inputs={"dt": "dt_", "dx": "dx_", "dy": "dy_", "eps": "eps_"},
                                           outputs={"out_u": "unew", "out_v": "vnew"},
                                           domain="domain_",
                                           mode=gt.mode.NUMPY)

        print(self.dummy_subdiv.sub_stencils)

    def test_domain_partitions(self):
        fileinput = "../subdomains_pymetis.dat.part.5"
        DomainPartitions.load_partitions(fileinput, fileformat="metis")
        DomainPartitions.print_partitions()

        fileinput = "../subdomains_scotch.map"
        DomainPartitions.load_partitions(fileinput, fileformat="scotch")
        DomainPartitions.print_partitions()

    def test_communicate_partitions(self):
        pass

    def test_subdivision_distribution(self):
        ddc = DomainDecomposition(preprocess=False, fileinput="subdomains_pymetis.dat.part.5", fileinputformat="metis")

        for sd in ddc.subdivisions:
            print(sd.id, sd.partitions_id)

    def test_compute_local_subdivs(self):
        subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([8, 8, 1]),
                                    halo=np.array([8, 8, 8, 8, 0, 0]), gridpoints=64,
                                    neighbors=np.array([None, 1, None, None, None]))
        subdiv1 = DomainSubdivision(id=1, pid=0, size=np.array([8, 8, 1]),
                                    halo=np.array([8, 8, 8, 8, 0, 0]), gridpoints=64,
                                    neighbors=np.array([0, None, None, None, None]))

        DomainPartitions.domain_partitions = np.array([0, 0])

        slist = [subdiv0, subdiv1]

        subdiv0.fields["unow"] = np.linspace(0, 63, 64).reshape((8, 8, 1))
        subdiv0.fields["unew"] = np.zeros(64).reshape((8, 8, 1))
        subdiv1.fields["unow"] = np.linspace(64, 127, 64).reshape((8, 8, 1))
        subdiv1.fields["unew"] = np.zeros(64).reshape((8, 8, 1))

        subdiv0.fields["vnow"] = np.linspace(0, 63, 64).reshape((8, 8, 1))
        subdiv0.fields["vnew"] = np.zeros(64).reshape((8, 8, 1))
        subdiv1.fields["vnow"] = np.linspace(64, 127, 64).reshape((8, 8, 1))
        subdiv1.fields["vnew"] = np.zeros(64).reshape((8, 8, 1))

        domain_ = gt.domain.Rectangle((0, 0, 0), (64, 64, 0))
        # Convert global inputs to GT4Py Global"s
        test_const = 2.0
        test_const = gt.Global(test_const)

        for sd in slist:
            sd.register_stencil(definitions_func=test_stencil_mult,
                                inputs={"in_u": "unow"},
                                global_inputs={"const": test_const},
                                outputs={"out_u": "unew"},
                                domain=domain_,
                                mode=gt.mode.NUMPY)


        for sd in slist:
            sd.register_stencil(definitions_func=test_stencil_minus,
                                inputs={"in_u": "vnow"},
                                global_inputs={"const": test_const},
                                outputs={"out_u": "vnew"},
                                domain=domain_,
                                mode=gt.mode.NUMPY)

        for sd in slist:
            sd.compute()

        for sd in slist:
            for k, v in sd.fields.items():
                print(k, v.reshape((64)))

    def test_communicate_locally(self):
        subdiv0 = DomainSubdivision(id=0, pid=0, size=np.array([5, 5, 1]),
                                    halo=np.array([5, 5, 5, 5, 0, 0]), gridpoints=25,
                                    neighbors=np.array([None, 1, None, None, None]))
        subdiv1 = DomainSubdivision(id=1, pid=0, size=np.array([5, 5, 1]),
                                    halo=np.array([5, 5, 5, 5, 0, 0]), gridpoints=25,
                                    neighbors=np.array([0, None, None, None, None]))

        DomainPartitions.domain_partitions = np.array([0, 0])

        slist = [subdiv0, subdiv1]

        subdiv0.fields["unow"] = np.linspace(0, 24, 25).reshape((5, 5, 1))
        subdiv0.fields["unew"] = np.zeros(25).reshape((5, 5, 1))
        subdiv1.fields["unow"] = np.linspace(25, 49, 25).reshape((5, 5, 1))
        subdiv1.fields["unew"] = np.zeros(25).reshape((5, 5, 1))

        domain_ = gt.domain.Rectangle((0, 1, 0), (5, 4, 0))
        # Convert global inputs to GT4Py Global"s

        for sd in slist:
            sd.register_stencil(definitions_func=test_stencil_move_y,
                                            inputs={"in_u": "unow"},
                                            outputs={"out_u": "unew"},
                                            domain=domain_,
                                            mode=gt.mode.NUMPY)


        for sd in slist:
            sd.compute()

        for sd in slist:
            for k, v in sd.fields.items():
                print(k, v.reshape((5, 5)).transpose())

        print("-------------------------------------")
        subdiv0.fields["unow"][:], subdiv1.fields["unow"][:] = subdiv0.fields["unew"][:], subdiv1.fields["unew"][:]

        for sd in slist:
            sd.compute()

        for sd in slist:
            for k, v in sd.fields.items():
                print(k, v.reshape((5, 5)).transpose())


if __name__ == "__main__":
    unittest.main()