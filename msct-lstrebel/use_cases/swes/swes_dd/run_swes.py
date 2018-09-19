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
import math
import sys
import argparse
import copy
from mpi4py import MPI

import gridtools as gt
import stencils
import timer as ti

from domain_decomposition import DomainDecomposition
from dd_postprocess import DomainPostprocess


class LaxWendroffSWES:
    """
    Implementation of the finite difference Lax-Wendroff scheme
    for the shallow water equations defined on a sphere.
    """

    def __init__(self, planet, t_final, m, n, ic, cfl, diff, backend, nparts,
                 dtype=np.float64, path="", prefix=""):
        """
        Constructor.

        Parameters
        ----------
        planet : int
            Integer denoting the planet on which set the equations.
            Available options are:

                * 0, for the Earth;
                * 1, for Saturn.

        t_final : float
            Simulation length in days.
        m : int
            Number of grid points in the longitude direction.
        n : int
            Number of grid points in the latitude direction.
        ic : tuple
            Tuple storing the identifier of the initial condition,
            followed by optional parameters. Available options for
            the identifier are:

                * 0, for the test case #1 by Williamson et al.;
                * 1, for the test case #2 by Williamson et al.;
                * 2, for the test case #6 by Williamson et al..

        cfl : float
            CFL number.
        diff : bool
            :obj:`True` to switch on numerical diffusion, :obj:`False` otherwise.
        backend : obj
            GT4Py backend.
        dtype : `obj`, float
            Type to be used for all NumPy arrays used with this object.
            Defaults tp :obj:`numpy.float64`.
        """
        # Set planet constants
        # For Saturn, all the values are retrieved from:
        # http://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html

        timer.start(name="Initialization", level=2)

        self.planet = planet
        if self.planet == 0:  # Earth
            self.g = 9.80616
            self.a = 6.37122e6
            self.omega = 7.292e-5
            self.scale_height = 8e3
            self.nu = 5e5
        elif self.planet == 1:  # Saturn
            self.g = 10.44
            self.a = 5.8232e7
            self.omega = 2. * math.pi / (10.656 * 3600.)
            self.scale_height = 60e3
            self.nu = 5e6
        else:
            raise ValueError("Unknown planet {}.".format(planet))
        #
        # Lat-lon grid
        #
        assert (m > 1) and (n > 1), \
            "Number of grid points along each direction must be greater than one."
        # Discretize longitude
        self.m = m
        # Discretize latitude
        self.n = n

        # Time discretization
        #
        assert (t_final >= 0.), "Final time must be non-negative."

        # Convert simulation length from days to seconds
        self.t_final = 24. * 3600. * t_final

        # Set maximum number of iterations
        if self.t_final == 0.:
            self.nmax = 50
            self.t_final = 3600.
        else:
            self.nmax = sys.maxsize

        # CFL number
        self.cfl = cfl

        # Initialize timestep size
        self.dt = gt.Global(0.)

        self.g = gt.Global(self.g)
        self.a = gt.Global(self.a)
        self.nu = gt.Global(self.nu)

        #
        # Numerical settings
        #
        assert ic[0] in range(3), "Invalid problem identifier. " \
                                  "See code documentation for supported initial conditions."

        self.ic = ic
        self.diffusion = diff

        if self.ic[0] == 0:
            self.only_advection = True
        else:
            self.only_advection = False

        self.backend = backend

        # Register fields and stencils to the DomainDecomposition class:
        self.prepared_domain = DomainDecomposition("subdomains_pymetis.dat.part." + str(nparts), "metis",
                                              path=path, prefix=prefix)

        if self.only_advection:
            self.halo = [1, 1, 1, 1, 0, 0]
        else:
            self.halo = [1, 1, 1, 1, 0, 0]

        self.prepared_domain.register_field(fieldname="h",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_h.npy",
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="dx",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dx.npy",
                                            staggered=(1, 0, 0),
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="dxc",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dxc.npy",
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="dy",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dy.npy",
                                            staggered=(0, 1, 0),
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="dy1",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dy1.npy",
                                            staggered=(0, 1, 0),
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="dy1c",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dy1c.npy",
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="c",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_c.npy",
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="c_midy",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_c_midy.npy",
                                            staggered=(0, 1, 0),
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="u",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u.npy",
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="v",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v.npy",
                                            haloincluded=True)

        self.prepared_domain.register_field(fieldname="h_new",
                                            halo=self.halo,
                                            field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_h.npy",
                                            haloincluded=True)

        if self.only_advection:
            self.prepared_domain.register_field(fieldname="v_midy",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v_midy.npy",
                                                staggered=(0, 1, 0),
                                                haloincluded=True)

            self.prepared_domain.register_field(fieldname="u_midx",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u_midx.npy",
                                                staggered=(1, 0, 0),
                                                haloincluded=True)

        if not self.only_advection:
            self.prepared_domain.register_field(fieldname="u_new",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="v_new",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="dyc",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dyc.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="f",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_f.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="hs",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_hs.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="tg",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_tg.npy")
            self.prepared_domain.register_field(fieldname="tg_midx",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_tg_midx.npy",
                                                staggered=(1, 0, 0),
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="tg_midy",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_tg_midy.npy",
                                                staggered=(0, 1, 0),
                                                haloincluded=True)
        if self.diffusion:
            self.prepared_domain.register_field(fieldname="Ax",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_ax.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="Bx",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_bx.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="Cx",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_cx.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="Ay",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_ay.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="By",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_by.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="Cy",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_cy.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="h_tmp",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_h.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="u_tmp",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="v_tmp",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="h_tmp2",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_h.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="u_tmp2",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="v_tmp2",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="h_tmp3",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_h.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="u_tmp3",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u.npy",
                                                haloincluded=True)
            self.prepared_domain.register_field(fieldname="v_tmp3",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v.npy",
                                                haloincluded=True)

        # Compute minimum longitudinal and latitudinal distance between
        # adjacent grid points, needed to compute time step size through
        # CFL condition
        dx_min = []
        dy_min = []
        for sd in self.prepared_domain.subdivisions:
            dx_min.append(sd.get_interior_field("dx")[:, 1:-1].min())
            dy_min.append(sd.get_interior_field("dy").min())
        self.dx_min = min(dx_min)
        self.dy_min = min(dy_min)

        #
        # Initialize stencils
        #
        if self.only_advection:
            self.stencil = self.prepared_domain.register_stencil(
                definitions_func=stencils.definitions_advection,
                inputs={"in_h": "h", "dx": "dx", "dxc": "dxc",
                        "dy1": "dy1", "dy1c": "dy1c",
                        "c": "c", "c_midy": "c_midy",
                        "u": "u", "u_midx": "u_midx",
                        "v": "v", "v_midy": "v_midy"},
                global_inputs={"dt": self.dt},
                outputs={"out_h": "h_new"},
                mode=self.backend,
            )
        else:
            self.stencil_lw = self.prepared_domain.register_stencil(
                definitions_func=stencils.definitions_lax_wendroff,
                inputs={"in_h": "h", "in_u": "u", "in_v": "v",
                        "dx": "dx", "dxc": "dxc",
                        "dy": "dy", "dyc": "dyc",
                        "dy1": "dy1", "dy1c": "dy1c",
                        "c": "c", "c_midy": "c_midy",
                        "f": "f", "hs": "hs", "tg": "tg",
                        "tg_midx": "tg_midx", "tg_midy": "tg_midy"},
                global_inputs={"dt": self.dt, "a": self.a, "g": self.g},
                outputs={"out_h": "h_new", "out_u": "u_new", "out_v": "v_new"},
                mode=self.backend,
            )
            if self.diffusion:
                rdx = [1, 1, 1, 1, 0, 0]
                self.stencil_hdiff = self.prepared_domain.register_stencil(
                    definitions_func=stencils.definitions_diffusion,
                    inputs={"in_q": "h_tmp2", "tmp_q": "h_tmp",
                            "Ax": "Ax", "Ay": "Ay",
                            "Bx": "Bx", "By": "By",
                            "Cx": "Cx", "Cy": "Cy"},
                    global_inputs={"dt": self.dt, "nu": self.nu},
                    outputs={"out_q": "h_tmp3"},
                    mode=self.backend,
                    reductions=rdx,
                )
                self.stencil_udiff = self.prepared_domain.register_stencil(
                    definitions_func=stencils.definitions_diffusion,
                    inputs={"in_q": "u_tmp2", "tmp_q": "u_tmp",
                            "Ax": "Ax", "Ay": "Ay",
                            "Bx": "Bx", "By": "By",
                            "Cx": "Cx", "Cy": "Cy"},
                    global_inputs={"dt": self.dt, "nu": self.nu},
                    outputs={"out_q": "u_tmp3"},
                    mode=self.backend,
                    reductions=rdx,
                )
                self.stencil_vdiff = self.prepared_domain.register_stencil(
                    definitions_func=stencils.definitions_diffusion,
                    inputs={"in_q": "v_tmp2", "tmp_q": "v_tmp",
                            "Ax": "Ax", "Ay": "Ay",
                            "Bx": "Bx", "By": "By",
                            "Cx": "Cx", "Cy": "Cy"},
                    global_inputs={"dt": self.dt, "nu": self.nu},
                    outputs={"out_q": "v_tmp3"},
                    mode=self.backend,
                    reductions=rdx,
                )

        timer.stop(name="Initialization")

    def solve(self, verbose, save, fixed_ts=None):
            """
            Perform the time marching.

            Parameters
            ----------
            verbose : int
                If positive, print to screen information about the solution
                every :obj:`verbose` timesteps. If negative, print to screen
                information only about the initial and final state.
            save : int
                If positive, store the solution every :obj:`save` timesteps.
                If negative, store only the initial and final state.

            Returns
            -------
            t_save : array_like
                1-D :class:`numpy.ndarray` collecting the time instants (in seconds)
                at which solution has been stored.
            phi : array_like
                2-D :class:`numpy.ndarray` collecting the longitudinal coordinates
                of all grid points.
            theta : array_like
                2-D :class:`numpy.ndarray` collecting the latitudinal coordinates
                of all grid points.
            h_save : array_like
                3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
                fluid height at time :obj:`t_save[k]`.
            u_save : array_like
                3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
                u-velocity at time :obj:`t_save[k]`.
            v_save : array_like
                3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
                v-velocity at time :obj:`t_save[k]`.
            """
            verbose = int(verbose)
            save = int(save)
            #
            # Time marching
            #
            n, t = 0, 0.

            # Communicate partition boundaries
            # self.prepared_domain.communicate("h")
            # self.prepared_domain.communicate("h_new")
            # self.prepared_domain.communicate("dx")
            # self.prepared_domain.communicate("dxc")
            # self.prepared_domain.communicate("dy")
            # self.prepared_domain.communicate("dy1")
            # self.prepared_domain.communicate("dy1c")
            # self.prepared_domain.communicate("c")
            # self.prepared_domain.communicate("c_midy")
            # self.prepared_domain.communicate("u")
            # self.prepared_domain.communicate("v")
            # if self.only_advection:
            #     self.prepared_domain.communicate("u_midx")
            #     self.prepared_domain.communicate("v_midy")
            # if not self.only_advection:
            #     self.prepared_domain.communicate("u_new")
            #     self.prepared_domain.communicate("v_new")
            #     self.prepared_domain.communicate("dyc")
            #     self.prepared_domain.communicate("f")
            #     self.prepared_domain.communicate("hs")
            #     self.prepared_domain.communicate("tg")
            #     self.prepared_domain.communicate("tg_midx")
            #     self.prepared_domain.communicate("tg_midy")
            #
            # hnew_north = np.zeros((self.m, self.halo[2], 1))
            # hnew_south = np.zeros((self.m, self.halo[3], 1))
            #
            # if not self.only_advection:
            #     unew_north = np.zeros((self.m, self.halo[2], 1))
            #     unew_south = np.zeros((self.m, self.halo[3], 1))
            #
            #     vnew_north = np.zeros((self.m, self.halo[2], 1))
            #     vnew_south = np.zeros((self.m, self.halo[3], 1))

            if MPI.COMM_WORLD.Get_rank() == 0:
                tsave = [0.0]

            timer.start(name="Time integration", level=2)
            self.prepared_domain.save_fields(["h"], postfix="t_" + str(0))

            while t < self.t_final and n < self.nmax:
                n += 1

                if fixed_ts is None:
                    # Compute timestep through CFL condition
                    dtnew = []
                    # Compute flux Jacobian eigenvalues
                    for sd in self.prepared_domain.subdivisions:
                        eigenx = (np.maximum(
                            np.absolute(sd.get_interior_field("u")[1:-1, :]
                                        - np.sqrt(self.g.value * np.absolute(sd.get_interior_field("h")[1:-1, :]))),
                            np.maximum(np.absolute(
                                sd.get_interior_field("u")[1:-1, :]), np.absolute(
                                sd.get_interior_field("u")[1:-1, :]
                                + np.sqrt(self.g.value * np.absolute(sd.get_interior_field("h")[1:-1, :])))))).max()
                        eigeny = (np.maximum(
                            np.absolute(sd.get_interior_field("v")[1:-1, :]
                                        - np.sqrt(self.g.value * np.absolute(sd.get_interior_field("h")[1:-1, :]))),
                                             np.maximum(np.absolute(
                                                 sd.get_interior_field("v")[1:-1, :]), np.absolute(
                                                 sd.get_interior_field("v")[1:-1, :]
                                                 + np.sqrt(self.g.value
                                                           * np.absolute(sd.get_interior_field("h")[1:-1, :])))))).max()

                        # Compute timestep
                        dtmax = np.minimum(self.dx_min / eigenx, self.dy_min / eigeny)
                        dtnew.append(self.cfl * dtmax)
                    # Select local minimum time step
                    self.dt.value = min(dtnew)
                    # If run with mpi collect all local minimum, choose the global minimum and send it to everybody
                    if MPI.COMM_WORLD.Get_size() > 1:
                        self.dt.value = MPI.COMM_WORLD.allreduce(sendobj=self.dt.value, op=MPI.MIN)
                    # if n % 100 == 0:
                    #     print(self.dt.value)
                else:
                    self.dt.value = fixed_ts

                # If needed, adjust time step
                if t + self.dt > self.t_final:
                    self.dt = gt.Global(self.t_final - t)
                    t = self.t_final
                else:
                    t += self.dt.value

                # Communicate partition boundaries
                timer.start(name="Communication during time integration", level=3)
                self.prepared_domain.communicate("h")
                if not self.only_advection:
                    self.prepared_domain.communicate("u")
                    self.prepared_domain.communicate("v")
                timer.stop(name="Communication during time integration")

                #
                # Apply boundary conditions
                #
                hnew_north = {}
                hnew_south = {}
                for sd in self.prepared_domain.subdivisions:
                    hnew_north[sd] = sd.get_interior_field("h")[:, 1, 0]
                    hnew_south[sd] = sd.get_interior_field("h")[:, -2, 0]
                    hnew_north[sd] = hnew_north[sd].reshape((sd.size[0], self.halo[2], sd.size[2]))
                    hnew_south[sd] = hnew_south[sd].reshape((sd.size[0], self.halo[3], sd.size[2]))

                    sd.set_boundary_condition("h", 2, hnew_north[sd])
                    sd.set_boundary_condition("h", 3, hnew_south[sd])

                if not self.only_advection:
                    unew_south = {}
                    unew_north = {}
                    vnew_south = {}
                    vnew_north = {}

                    for sd in self.prepared_domain.subdivisions:
                        unew_north[sd] = sd.get_interior_field("u")[:, 1, 0]
                        unew_south[sd] = sd.get_interior_field("u")[:, -2, 0]
                        unew_north[sd] = unew_north[sd].reshape((sd.size[0], self.halo[2], sd.size[2]))
                        unew_south[sd] = unew_south[sd].reshape((sd.size[0], self.halo[3], sd.size[2]))

                        sd.set_boundary_condition("u", 2, unew_north[sd])
                        sd.set_boundary_condition("u", 3, unew_south[sd])

                        vnew_north[sd] = sd.get_interior_field("v")[:, 1, 0]
                        vnew_south[sd] = sd.get_interior_field("v")[:, -2, 0]
                        vnew_north[sd] = vnew_north[sd].reshape((sd.size[0], self.halo[2], sd.size[2]))
                        vnew_south[sd] = vnew_south[sd].reshape((sd.size[0], self.halo[3], sd.size[2]))

                        sd.set_boundary_condition("v", 2, vnew_north[sd])
                        sd.set_boundary_condition("v", 3, vnew_south[sd])

                self.prepared_domain.apply_boundary_condition("h")
                if not self.only_advection:
                    self.prepared_domain.apply_boundary_condition("u")
                    self.prepared_domain.apply_boundary_condition("v")


                #
                # Update solution at the internal grid points
                #
                if self.only_advection:
                    self.stencil.compute()
                else:

                    self.stencil_lw.compute()

                    if self.diffusion:
                        for sd in self.prepared_domain.subdivisions:
                            sd.fields["h_tmp"] = sd.fields["h_new"].copy()
                            sd.fields["u_tmp"] = sd.fields["u_new"].copy()
                            sd.fields["v_tmp"] = sd.fields["v_new"].copy()
                            sd.fields["h_tmp3"] = sd.fields["h_new"].copy()
                            sd.fields["u_tmp3"] = sd.fields["u_new"].copy()
                            sd.fields["v_tmp3"] = sd.fields["v_new"].copy()
                            sd.fields["h_tmp2"] = sd.fields["h"].copy()
                            sd.fields["u_tmp2"] = sd.fields["u"].copy()
                            sd.fields["v_tmp2"] = sd.fields["v"].copy()

                            # print("before", (sd.fields["h_new"] == sd.fields["h_tmp3"]).all())

                        self.stencil_hdiff.compute()
                        self.stencil_udiff.compute()
                        self.stencil_vdiff.compute()

                        # for sd in self.prepared_domain.subdivisions:
                        #     print("after", (sd.fields["h_tmp"] == sd.fields["h_tmp3"]).all())

                if self.only_advection:
                    self.prepared_domain.swap_fields("h", "h_new")
                else:
                    self.prepared_domain.swap_fields("h", "h_tmp3")
                    self.prepared_domain.swap_fields("u", "u_tmp3")
                    self.prepared_domain.swap_fields("v", "v_tmp3")

                # self.prepared_domain.swap_fields("h", "h_new")
                # if not self.only_advection:
                #     self.prepared_domain.swap_fields("u", "u_new")
                #     self.prepared_domain.swap_fields("v", "v_new")

                if n % 100 == 0 and MPI.COMM_WORLD.Get_rank() == 0:
                    umax = []
                    for sd in self.prepared_domain.subdivisions:
                        # print("time step " + str(n) + " time " + str(t) + " sd " + str(sd.id) + " u max " + str(np.max(sd.get_interior_field("u_new")[:])))
                        norm = np.sqrt(sd.get_interior_field("u")[1:-1, :] * sd.get_interior_field("u")[1:-1, :] +
                                       sd.get_interior_field("v")[1:-1, :] * sd.get_interior_field("v")[1:-1, :])
                        umax.append(norm.max())

                    print('%7.2f (out of %i) hours: max(|u|) = %13.8f'
                          % (t / 3600., int(self.t_final / 3600.), max(umax)))

                timer.start(name="Saving fields during time integration", level=3)
                if (save > 0 and n % save == 0) or t == self.t_final:
                    self.prepared_domain.save_fields(["h"], postfix="t_" + str(n))

                if MPI.COMM_WORLD.Get_rank() == 0:
                    if (save > 0 and n % save == 0) or t == self.t_final:
                        tsave.append(t)

                    if t == self.t_final:
                        tsave.append(n)
                        np.save(path + prefix + "t.npy", tsave)
                timer.stop(name="Saving fields during time integration")

            timer.stop(name="Time integration")


def postprocess_swes(nx, ny, nz, nic, save_freq, path="", prefix=""):
    postproc = DomainPostprocess(path=path, prefix=prefix)

    hsave = postproc.combine_output_files(size=[nx, ny, nz], fieldname="h",
                                      path=path, prefix=prefix,
                                      postfix="t_" + str(0), save=False, cleanup=True)
    hsave = np.append(hsave, hsave[0, :].reshape((1, hsave.shape[1], hsave.shape[2])), axis=0)

    # htemp = np.load("swes_ic{}_h.npy".format(nic))
    # hsave = htemp[1:-1, 1:-1, :]
    # hsave = np.append(hsave, hsave[0, :].reshape((1, hsave.shape[1], hsave.shape[2])), axis=0)

    tsave = np.load(path + prefix + "t.npy")
    nt = int(tsave[-1])
    for n in range(1, nt + 1):
        # Save
        if ((save_freq > 0) and (n % save_freq == 0)) or (n == nt):
            hnew = postproc.combine_output_files(size=[nx, ny, nz], fieldname="h",
                                             path=path, prefix=prefix,
                                             postfix="t_" + str(n), save=False, cleanup=True)

            hnew = np.append(hnew, hnew[0, :].reshape((1, hnew.shape[1], hnew.shape[2])), axis=0)
            # print(hnew.shape, hsave.shape)
            hsave = np.concatenate((hsave, hnew), axis=2)

    phi = np.load(path + prefix + "swes_ic{}_phi.npy".format(nic))
    phi = np.append(phi, 2.0 * math.pi * np.ones((1, phi.shape[1])), axis=0)

    theta = np.load(path + prefix + "swes_ic{}_theta.npy".format(nic))
    theta = np.append(theta, theta[0, :].reshape((1, theta.shape[1])), axis=0)

    usave = np.load(path + prefix + "swes_ic{}_u.npy".format(nic))

    vsave = np.load(path + prefix + "swes_ic{}_v.npy".format(nic))

    filename = path + prefix + "swes_no_poles_ic{}.npz".format(nic)
    np.savez(filename, t=tsave[:-1], phi=phi[1:nx+2, 1:ny+1], theta=theta[1:nx+2, 1:ny+1],
             h=hsave, u=usave[1:nx+2, 1:ny+1], v=vsave[1:nx+2, 1:ny+1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Shallow Water Equation on a Sphere")
    parser.add_argument("-nx", default=360, type=int,
                        help="Number of grid points in x direction.")
    parser.add_argument("-ny", default=180, type=int,
                        help="Number of grid points in y direction.")
    parser.add_argument("-nz", default=1, type=int,
                        help="Number of grid points in z direction.")
    parser.add_argument("-ic", default=0, type=int,
                        help="Initial condition either 0 or 2.")
    parser.add_argument("-nt", default=12, type=int,
                        help="Number of days the simulation should run.")
    parser.add_argument("-sf", default=100, type=int,
                        help="Save frequency: Number of time steps between fields are saved to file.")
    parser.add_argument("-np", default=2, type=int,
                        help="Number of partitions.")
    parser.add_argument("-loc", default="", type=str,
                        help="Path to location where files should be saved to.")
    parser.add_argument("-pf", default="", type=str,
                        help="Prefix for file names.")
    parser.add_argument("-ft", default=None, type=float,
                        help="Optional: If set use as fixed time step instead of adaptive time stepping.")
    args = parser.parse_args()

    nx = args.nx
    ny = args.ny
    nz = args.nz
    aic = args.ic
    days = args.nt
    sf = args.sf
    nparts = args.np

    path = args.loc
    prefix = args.pf


    timer = ti.Timings(name="Shallow Water Equation on a Sphere")
    timer.start(name="Overall SWES time", level=1)

    # Suggested values for $\alpha$ for first and second
    # test cases from Williamson"s suite:
    # * 0
    # * 0.05
    # * pi/2 - 0.05
    # * pi/2
    ic = (aic, 0) #math.pi / 2)

    # Suggested simulation"s length for Williamson"s test cases:
    # * IC 0: 12 days
    # * IC 1: 14 days
    # t_final = 12
    # t_final = 3
    # nx = 360
    # ny = 180
    # nz = 1

    # Let"s go!
    solver = LaxWendroffSWES(planet=0, t_final=days, m=nx, n=ny, ic=ic,
                             cfl=1, diff=True, backend=gt.mode.NUMPY, nparts=nparts,
                             dtype=np.float64, path=path, prefix=prefix)
    # save_freq = 100 #25
    solver.solve(verbose=100, save=sf, fixed_ts=args.ft)

    timer.stop(name="Overall SWES time")

    timer.start(name="Post processing time", level=1)

    if MPI.COMM_WORLD.Get_rank() == 0:
        # Save data
        postprocess_swes(nx, ny, nz, ic[0], sf, path=path, prefix=prefix)

    timer.stop(name="Post processing time")
    if MPI.COMM_WORLD.Get_rank() == 0:
        timer.list_timings()